import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase  # train/test
        self.num_classes = num_classes
        self.cfg = voc  # (coco, )[num_classes == 2] # voc和coco都是字典 找到num_classes对应键值，这里返回voc字典
        self.priorbox = PriorBox(self.cfg)  # 实例化PriorBox类，类实现的功能时生成所有的prior anchors
        # 结合生成先验框的操作，priors保存的是【tensor 8760行4列】
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)  # vgg16层
        # Layer learns to scale the l2 normalized features from conv4_3
        # conv4-3需要做L2归一化操作
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)  # SSD后面添加的额外层
        # head中包含两个list，第一个是loc预测，第二个是类别预测
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':  # 测试阶段
            self.softmax = nn.Softmax(dim=-1)  # 最后一个维度是预测分类信息，需要进行softmax来做类别判断
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    default box对应每个分类的confidence
                    2: localization layers, Shape: [batch,num_priors*4]
                    每个default box的4个坐标信息
                    3: priorbox layers, Shape: [2,num_priors*4]
                    计算每个default box在同一尺度下的坐标，用于后面IoU、offset的计算
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu激活后再进行L2Norm操作后输出的tensor
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7也就是vgg基础层最后一层 relu激活层操作后的输出tensor
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs额外添加的4个tensor提取出
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # sources里面包括了6个特征层
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf): # sources loc conf都是6个元素的list
            # 通道变换[batch, C, H, W]->[batch, H, W, C]，前面介绍的在C通道进行softmax分类
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())  # 这个就是通道维度变换类似于transpose
            # view只能作用在contiguous的variable上，如果在view之前调用了transpose、permute等，就需要调用contiguous()来返回一个contiguous copy
            # 判断ternsor是否为contiguous，可以调用torch.Tensor.is_contiguous()函数 返回为bool值
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# 注释部分参考https://blog.csdn.net/zxd52csx/article/details/82795104
def vgg(cfg, i, batch_norm=False):
    layers = []  # 用于存放vgg网络的list
    in_channels = i  # 最前面那层的维度--300*300*3，因此i=3
    for v in cfg:  # 代码厉害的地方，循环建立多层，数据信息存放在一个字典中
        if v == 'M':  # maxpooling 时边缘不补
            # 窗口大小；窗口移动步长默认是kernel_size；ceil_mode向上取整，反之向下取整
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':  # maxpooling 时边缘补NAN
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:  # 卷积前后维度可以通过字典中数据设置好
            # 输入维度；输出维度；kernel_size卷积核大小；stride步长大小；补零；dilation，kernel间距；group分组卷积
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # dilation=卷积核元素之间的间距,扩大卷积感受野的范围，没有增加卷积size
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        # S代表stride，为2时久相当于缩小feature map，即下采样
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


# 部分参考https://blog.csdn.net/qq_18315295/article/details/104095555
def multibox(vgg, extra_layers, cfg, num_classes):
    """
    :param vgg: 没有全连接层的vgg网络
    :param extra_layers: vgg网络后面新增的额外层
    :param cfg: '300': [4, 6, 6, 6, 4, 4], 不同部分的feature map上一个网格预测多少框
    :param num_classes: 20分类+1背景，共21类
    :return: 返回vgg，extra_layers结构，6个特整层提取的 loc层+conf层
    """
    loc_layers = []  # loc_layers的输出维度是default box的种类(4/6)*4
    conf_layers = [] # conf_layers的输出维度是default box的种类(4/6)*num_class
    vgg_source = [21, -2]  # 第21层和倒数第2层
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]  # 特征图尺寸没动，通道改变为[4/6*4]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]  # 只改变通道[4/6*num_class]
    for k, v in enumerate(extra_layers[1::2], 2):  # 找到对应的层
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]  # 只改变通道为[4/6*4]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]  # 只改变通道[4/6*num_class]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    # 当前的代码只支持300*300
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),  # 网络结构是先经过vgg+额外层 这里保留原始vgg的输出1024通道，额外层输入为1024
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
