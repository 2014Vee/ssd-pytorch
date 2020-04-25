# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import config as cfg  # ##将原本的coco换成了voc0712
from ..box_utils import match, log_sum_exp
import focal_loss  # ##引入了FocalLoss


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
           ###ground——trut和自定义的prior框做匹配，IOU大于0.5就默认为正样本
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
           ###编码过程
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
           ###难例挖掘部分（正负样本的比值为1：3）
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        损失部分有两部分组成（类别的交叉熵loss和位置的smoothL2损失）
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences, （预测的类别置信度）
            l: predicted boxes,（预测的回归框）
            g: ground truth boxes （ground——truth框）
            N: number of matched default boxes （匹配的默认框数目）
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    # 输入参数：类别数，阈值，是否用prior框匹配（bool）,背景的标签值, 是否难例挖掘（bool）,负例和正例的比例, 确定为困难负例的IUO最小值, 编码对象（bool）,默认使用GPU
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh  # ##阈值
        self.background_label = bkg_label  # ##背景的标签值*****但是后面好像没使用
        self.encode_target = encode_target  # ##不明白，后面好像也没用
        self.use_prior_for_matching = prior_for_matching # ##是否用prior框匹配（bool）
        self.do_neg_mining = neg_mining  # ##是否难例挖掘（bool）
        self.negpos_ratio = neg_pos  # ##负例和正例的比例
        self.neg_overlap = neg_overlap  # ##确定为困难负例的IUO最小值
        self.variance = cfg.voc['variance']

    # ##forward里面包括了【难例挖掘】
    # ##输入参数1：网络结构net输出的out:[loc conf priors]
    # ##输入参数2：targets:真实目标的位置标签值
    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size【batch_size, num_priors, num_classes】 3维度
                loc shape: torch.size【batch_size, num_priors, 4】 3维度
                priors shape: torch.size【num_priors,4】

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions  # 【prediction包括net预测的位置信息，预测的类别，所有的先验框】
        num = loc_data.size(0)  # batch_size每次输入的图片数
        priors = priors[:loc_data.size(1), :]  # priors里面包括所有的先验prior框[8732,4] # feel no use
        num_priors = (priors.size(0))  # 8732 anchors的数量
        num_classes = self.num_classes  # 类别数

        # match priors (default boxes) and ground truth boxes
        # ##下面的loc_t和conf_t是生成的随机的
        loc_t = torch.Tensor(num, num_priors, 4) # [batch_size,8732,4] 每张图片有8732个先验框，每个先验框有四个数值[中心点xy，高，宽]
        # 用来记录每一个default box的类别，0类就是负样本
        conf_t = torch.LongTensor(num, num_priors)  # [batch_size,8732] 每张图片生成8732个先验框 每个先验框有一个置信度的的值
        for idx in range(num):  # 对每个batch_size里每一张图进行遍历
            # target里面是五维度tensor，最后个维度是label
            truths = targets[idx][:, :-1].data  # position 真实的ground_truth方框信息 targets是5维数据【前4维表示位置信息，最后1维表示类别】
            labels = targets[idx][:, -1].data  # labels 真实的回归框标签信息
            defaults = priors.data  # [8732,4] default box在同一尺度下的坐标是不变的，与batch无关

            # 【MATCH函数】参数输入【阈值，ground_truth,设置的先验框prior,variance方差？,真实标签，位置预测，类别预测，遍历每个batch中的图片顺序】
            match(self.threshold, truths, defaults, self.variance, labels,loc_t, conf_t, idx)
            # match这个函数给每个ground truth匹配了最好的priors，给每个priors匹配最好的ground truth
            # 经过encode后的offset([g_cx cy, g_wh])->loc_t,top class label for each prior->conf_t
            # match函数最后更新 loc_t, conf_t 【编码之后的位置信息和类别信息】
            # loc_t 【batch_size, 8732, 4】
            # conf_t【batch_size, 8732】
        if self.use_gpu:  # 将编码后的位置信息和类别信息放在GPU上
            loc_t = loc_t.cuda()  # 【loc_t里面是一个batch中所有图片的位置信息，每张图片有（8732,4）】 Tensor:【batch_size,7843,4】
            conf_t = conf_t.cuda()  # Tensor: 【batch_size,8732】
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)  # #Tensor:【batch_size,7843,4】 encoded offsets to learn
        conf_t = Variable(conf_t, requires_grad=False)
        # #Tensor: 【batch_size,8732】 top class label for each prior conf_t是标签值

        pos = conf_t > 0  # 只有大于0的才被认为不是背景，而是存在目标 pos=bool型 pos=Tensor:【batch_size,8732】
        num_pos = pos.sum(dim=1, keepdim=True)  # num_pos记录的是8732个框中是存在目标的方框 选择为正样本的数量？？？

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # loc_loss是只考虑正样本的  loc_data是预测的tensor
        # ## pos_idx是bool型【batch_size,8732,4】，记录的是每张图片中生成的prior中是目标是True 背景是False
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # 首先将pos的最后个维度添加个'1' 再将bool型的pos【batch_size,8732】->【batch_size,8732,4】
        loc_p = loc_data[pos_idx].view(-1, 4)  # ## 由net预测的存在目标的区域目标 loc_p (p代表positive) 【前景目标区域的个数，4】
        loc_t = loc_t[pos_idx].view(-1, 4)  # ## 由实际GT 编码出来的loc_t
        # 输入的loc_p是指真实编码后的ground_truth 和 网络的预测位置结果 通过L1函数计算损失
        '''
        【loss_l】即为位置损失值
        '''
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)  # ##输入参数1：网络net的位置预测 输入参数2：真实GT编码后的位置信息
# ############################################################################################################################################
# ############################################################################################################################################
        '''【难例挖掘】'''
        # 【conf_data】: torch.size(batch_size,num_priors,num_classes)
        batch_conf = conf_data.view(-1, self.num_classes)  # 【batch_size*8732行,num_classes列】  一个batch_size中所有prior的数量
        # 【参照论文中conf计算方式】
        # ## conf_t.view(-1, 1) 【batch_size*8732行, 1列】 与GT匹配之后的置信度的值
        # ## batch_conf 【batch_size*8732行,num_classes列】 每个prior中N类别的置信度
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))  # ##将预测信息按照论文中的公式编码【难懂】
        # 得到的loss_c  torch.Size([batch_size*8732, 1])

        # 【Hard Negative Mining】
        # loss_c[pos.view(-1, 1)] = 0###上面两行被同时注释掉
        loss_c = loss_c.view(num, -1)  # ##这里和下面一行调换了 loss=【torch.Size([batch_size, 8732])】
        loss_c[pos] = 0  # ##将正例样本的损失置为0，背景样本的loss不是0 pos(bool型)=Tensor:【batch_size,8732】
        _, loss_idx = loss_c.sort(1, descending=True)  # _ 里面存 放每行由大到小的数列， loss_idx 降序后的元素在原本每行中的index
        _, idx_rank = loss_idx.sort(1)  # ##idx_rank [batch_size ,8732]
        # ## 第一次sort：得到的index是按顺序排的索引   第两次sort：得到原Tensor的损失从大到小的映射，排第几的数字变为排名【难懂但看懂了】
        # ## 总结：正样本为默认框与真实框根据iou匹配得到，负样本为分类loss值排序得到。
        # ## 先将 pos bool型（True，False）转化为（1，0） num_pos：【batch_size, 1】 每一行记录的是batch中 每一张图片中有目标的prior数量
        num_pos = pos.long().sum(1, keepdim=True)
        # ## max=pos.size(1)-1 表示最多有多少个prior，每张图片中的负样本数不能超过每张图片中最大的prior数
        # ## negpos_ratio*num_pos 表示负样本数是正样本数的3倍
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)  # num_neg返回的是 torch.Size([batch_size, 1])
        # ## 【num_pos，num_neg】均为【batch_size, 1】 分别记录了每张图片中正样本和负样本的数目 比例 1:3

        # ## neg(bool型)【batch_size, 8732】 选取了每张图片中 排名前（对应负样本数量）的 设置为True
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # 置信度的损失包括 正/负样本都包括损失
        # 因为pos 和 neg 都是bool型 因此 pos_idx 和 neg_idx 也是bool型
        # ## pos_idx 和 neg_idx 均为【batch_size, 8732 ,num_classes】
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # ## conf_p：【batch_size*8732 , num_classes】
        # ## conf_p  包括 正/负样本都要算入损失
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        # ## Net在每个prior框中分别预测每一类别结果【batch_size*8732 , num_classes】
        targets_weighted = conf_t[(pos+neg).gt(0)]  # ## 含有GT信息【batch_size,8732】
        '''
        【loss_c】即为类别损失值
        '''
        # ##参数1：conf_p 是Net在每个prior框中分别预测每一类别结果
        # ##参数2：targets_weighted 是存储的标签值long形式
        # ##【FocalLoss函数是针对类别损失部分 【问题1】：正样本/负样本不均衡 【问题2】：难易样本本身对损失函数的贡献不一样】
        # ##-------------------------------------------------------------------------------------------------
        compute_c_loss = focal_loss.FocalLoss(alpha=None, gamma=2, class_num=num_classes, size_average=False)
        loss_c = compute_c_loss(conf_p, targets_weighted)
        # ##下面是原本的损失函数 若引入FocalLoss那么就注释掉这一行
        # loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)  ###【难懂没懂】  ************
        # ## Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        # ##-------------------------------------------------------------------------------------------------
        
        N = num_pos.data.sum()  # ## N：一个batch中的所有图片的目标总数
        N=N.double()
        loss_l = loss_l.double()  # 上面加入double()下面也添加了一行
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
