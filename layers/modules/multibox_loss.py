# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import config as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
           ***SSD的match策略，groundtruth和prior框做匹配，IOU大于0.5即为正样本
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
           难例挖掘部分
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,预测类别置信度
            l: predicted boxes,预测的回归框
            g: ground truth boxes GT框
            N: number of matched default boxes 匹配的默认框数目
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    # 构造函数参数：类别数，阈值，是否用prior框匹配（bool）,背景的标签值, 是否难例挖掘（bool）,
    # 负例和正例的比例, 确定为困难负例的IUO最小值, 编码对象（bool）,默认使用GPU
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap  # 难例挖掘中确认为困难负例的iou最小值
        self.variance = cfg.voc['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            输入参数1：网络结构net输出的out：[loc conf priors]
            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
            loc_t,conf_t是由target产生的标签数据
            loc_data,conf_data是feature map计算出来的预测数据
        """
        loc_data, conf_data, priors = predictions  # 包括net预测的loc，conf，所有的prior box
        num = loc_data.size(0)  # batch_size每次输入的图片数
        priors = priors[:loc_data.size(1), :]  # 这部分没啥用，priors里面包括所有的先验框[8732, 4]
        num_priors = (priors.size(0))  # 8732 anchor数量
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)  # 初始化每张图片的8732个先验框[batch, 8732, 4]，每个先验框[中心点xy, w, h]
        conf_t = torch.LongTensor(num, num_priors)  # 每张图片生成8732个先验框，每个框有一个置信度值
        for idx in range(num):  # 对batch内每一张图片进行遍历
            # target里面是5维tensor，最后维度是label
            truths = targets[idx][:, :-1].data  # position真实的GT方框信息 [前4维表示位置信息 最后一维表示类别]
            labels = targets[idx][:, -1].data  # labels真实回归框标签信息
            defaults = priors.data  # [8732, 4] default box在同一尺度下的坐标是不变的，与batch无关

            # 匹配函数，参数输入[阈值，ground_truth,设置的先验框prior,variance方差？,真实标签，位置预测，类别预测，遍历每个batch中的图片顺序]
            # match这个函数给每个ground truth匹配了最好的priors，给每个priors匹配最好的ground truth
            # 经过encode后的offset([g_cx cy, g_wh])->loc_t,top class label for each prior->conf_t
            # match函数最后更新 loc_t, conf_t [编码之后的位置信息和类别信息]
            # loc_t [batch_size, 8732, 4]
            # conf_t [batch_size, 8732]
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)
            # 经过match之后对default boxes进行了匹配，得到了正负样本的信息以及偏移量的标签
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)  # 转换tensor为variable，默认是不要求梯度的
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0  # 只有大于0的才被认为不是背景，而是存在目标pos=bool型pos=Tensor:[batch, 8732]
        num_pos = pos.sum(dim=1, keepdim=True)  # num_pos记录的是8732个框中是存在目标的方框 选择为正样本的数量

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # loc_loss是只考虑正样本的  loc_data是预测的tensor
        # pos_idx是bool型[batch, 8732, 4]，记录的是每张图片中生成的prior中目标是True背景是False
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # 首先将pos的最后维度添加个‘1’再将bool型的pos [batch, 8732]->[batch, 8732, 4]
        # 由net预测的存在目标的区域目标 loc_p（p即positive）
        loc_p = loc_data[pos_idx].view(-1, 4)
        # 由实际的GT得出位置
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        # 难例挖掘部分
        # conf_data:[batch, num_piror, num_classes]
        batch_conf = conf_data.view(-1, self.num_classes) # [batch*8732, num_class] 一个batch内所有prior的数量
        # 参考论文中conf计算方式
        # conf.view() [batch*8732, 1] 与GT匹配之后的置信度的值
        # batch_conf [batch*8732, num_classes] 每个prior中N类别的置信度
        # 得到的loss_c [batch*8732, 1]
        # loss_c是在难负样本挖掘中用来给默认框排序的，后续分类损失会改变对应的计算
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)  # [batch, 8732]
        loss_c[pos] = 0  # filter out pos boxes for now  # 正例样本的损失置为0，背景样本的loss不是0 pos（bool）=Tensor:[batch, 8732]
        _, loss_idx = loss_c.sort(1, descending=True)  # _ 存放每行由大到小的数列，loss_idx降序后的元素在原本每行中的index
        _, idx_rank = loss_idx.sort(1)  # idx_rank [batch, 8732]
        # ## 第一次sort：得到的index是按顺序排的索引   第两次sort：得到原Tensor的损失从大到小的映射，排第几的数字变为排名
        # ## 总结：正样本为默认框与真实框根据iou匹配得到，负样本为分类loss值排序得到。
        # ## 先将 pos bool型（True，False）转化为（1，0） num_pos：【batch_size, 1】 每一行记录的是batch中 每一张图片中有目标的prior数量

        num_pos = pos.long().sum(1, keepdim=True)
        # max=pos.size(1)-1表示最多有多少个prior，每张图片中的负样本数不能超过每张图片中最大的prior数
        # ngpos_ratio*num_pos 表示负样本数量是正样本的3倍
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)  # num_neg返回的是 torch.Size([batch_size, 1])
        # ## 【num_pos，num_neg】均为【batch_size, 1】 分别记录了每张图片中正样本和负样本的数目 比例 1:3
        # neg(bool) [batch, 8732] 选取了每张图片中排名靠前的（对应负样本数量）的设置为True
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # 置信度的损失包括 正/负样本都包括损失
        # 因为pos 和 neg 都是bool型 因此 pos_idx 和 neg_idx 也是bool型
        # ## pos_idx 和 neg_idx 均为[batch_size, 8732 ,num_classes]

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # ## conf_p：【batch_size*8732 , num_classes】
        # ## conf_p  包括 正/负样本都要算入损失
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        # ## Net在每个prior框中分别预测每一类别结果【batch_size*8732 , num_classes】
        targets_weighted = conf_t[(pos + neg).gt(0)]  # ## 含有GT信息【batch_size,8732】

        # ***分类损失函数，这里其实和上面的loss_c不是一个
        # 参数1：conf_p是Net在每个prior框中分别预测每一类别结果
        # 参数2：是存储的标签值long形式
        # 如果想对这部分进行改进的话就直接改损失函数即可
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()  # N:一个batch中的所有图片的目标总数
        loss_l = loss_l.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
