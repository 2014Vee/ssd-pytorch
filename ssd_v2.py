# -*- coding: utf-8 -*-
# @Time    : 2020/11/18 21:20
# @Author  : 2014Vee
# @Email   : 1976535998@qq.com
# @File    : ssd_v2.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
from layers import *
from data import voc


# 参考https://github.com/zigangzhao-ai/ssd.pytorch
#############################################
# 加入SE
class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


#####################################################
#ECA模块
class ECAModule(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

#############################################
# Upsample--->deconvd
class Upsample(nn.Module):
    """ nn.Upsample is deprecated
    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.
    The modes available for upsampling are: `nearest`, `linear` (3D-only),
    `bilinear` (4D-only), `trilinear` (5D-only)
    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
        mode (string): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'trilinear'. Default: 'nearest'
        align_corners (bool, optional): if True, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is `linear`,
            `bilinear`, or `trilinear`. Default: False
    warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.
    """

    def __init__(self, size, scale_factor=None, mode="nearest"): #bilinear，nearest
        super(Upsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        return x

#############################【SSD改进】######################
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
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        # SSD network
        self.vgg = nn.ModuleList(base)  # vgg16层,modulelist变成了可迭代的module子类
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

    # =====se_fpn新增==================
        # pool2到conv4_3  扩张卷积，尺度少一半
        self.DilationConv_128_128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2,
                                              stride=2)
        # conv4_3到conv4_3  尺度不变
        self.conv_512_256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        # fc7 到 conv4_3    反卷积上采样，尺度大一倍
        # self.DeConv_1024_128 = nn.ConvTranspose2d(in_channels=1024, out_channels=128, kernel_size=2, stride=2)
        self.upsample_1024_1024 = Upsample(38)
        self.conv_1024_128 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1, stride=1)

        # conv4_3 到FC7  扩张卷积，尺度少一半
        self.DilationConv_512_256 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=2, dilation=2,
                                              stride=2)
        # FC7到FC7 尺度不变
        self.conv_1024_512 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1)
        # conv8_2 到 FC7    反卷积上采样，尺度大一倍  10->19
        # self.DeConv_512_128 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.upsample_512_512 = Upsample(19)
        self.conv_512_256_fc7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)

        # conv5_3到conv8_2,扩张卷积，尺度少一半
        self.DilationConv_512_128_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=2, dilation=2,
                                                stride=2)
        # conv8_2到conv8_2 尺度不变
        self.conv_512_256_2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        # conv9_2到conv8_2
        # self.DeConv_256_128_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upsample_256_256_2 = Upsample(10)
        self.conv_256_128_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)

        # conv9_2到conv9_2 尺度不变
        self.conv_256_128_9_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.upsample_256_256_10_2 = Upsample(5)
        self.conv_256_128_9_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)

        # conv10_2到conv10_2 尺度不变
        self.conv_256_128_10_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=1)
        self.upsample_256_256_11_2 = Upsample(3)
        self.conv_256_128_10_2 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1)

        # 平滑层
        # self.smooth = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        # self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        # 通道数BN层的参数是输出通道数out_channels
        self.bn = nn.BatchNorm2d(128)
        self.bn1 = nn.BatchNorm2d(256)

        # CBAM模块【6个特征层：512 512 512 256 256 256 】
        # self.CBAM1 = Bottleneck(512)
        # self.CBAM2 = Bottleneck(512)
        # self.CBAM3 = Bottleneck(512)
        # self.CBAM4 = Bottleneck(256)
        # self.CBAM5 = Bottleneck(256)
        # self.CBAM6 = Bottleneck(256)

        # SE模块【6个特征层：512 512 512 256 256 256 】
        # SE+SSD模块 6个特整层的channel【512 1024
        self.SE1 = SEModule(512)
        self.SE2 = SEModule(1024)
        self.SE3 = SEModule(512)
        self.SE4 = SEModule(256)
        self.SE5 = SEModule(256)
        self.SE6 = SEModule(256)

        # ECA模块【6个特征层：512 512 512 256 256 256 】
        self.ECA1 = ECAModule(512)
        self.ECA2 = ECAModule(1024)
        self.ECA3 = ECAModule(512)
        self.ECA4 = ECAModule(256)
        self.ECA5 = ECAModule(256)
        self.ECA6 = ECAModule(256)





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
        for k in range(10):
            x = self.vgg[k](x)
        # sources.append(x)

        # apply vgg up to conv4_3 relu
        for k in range(10, 23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7也就是vgg基础层最后一层 relu激活层操作后的输出tensor
        for k in range(23, 30):
            x = self.vgg[k](x)
        x = self.ECA1(x)

        s = self.L2Norm(x)
        # sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        x = self.ECA2(x)
        sources.append(x)


        # apply extra layers and cache source layer outputs额外添加的4个tensor提取出
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                if k == 1:
                    x = self.ECA3(x)
                if k == 3:
                    x = self.ECA4(x)
                if k == 5:
                    x = self.ECA5(x)
                if k == 7:
                    x = self.ECA6(x)
                sources.append(x)
        # 此时sources保存了所有中间结果，论文中的pool2、conv4_3、conv5_3、fc7、conv8_2、conv9_2、conv10_2、conv11_2


        # # sources_final保存各层融合之后的最终结果
        # sources_final = list()
        # # con4_3层融合结果  self.bn1(self.conv1(x)) 在通道维度上融合
        # # conv4_fp = torch.cat((F.relu(self.bn(self.DilationConv_128_128(sources[0])), inplace=True),
        # #                       F.relu(self.conv_512_256(sources[1]), inplace=True),
        # #                       F.relu(self.DeConv_1024_128(sources[3]), inplace=True)), 1)
        #
        # conv4_fp = torch.cat((F.relu(self.bn(self.DilationConv_128_128(sources[0])), inplace=True),
        #                       F.relu(self.conv_512_256(sources[1]), inplace=True),
        #                       F.relu(self.conv_1024_128(self.upsample_1024_1024(sources[3])), inplace=True)), 1)
        #
        # conv4_fp = F.relu(conv4_fp, inplace=True)
        # sources_final.append(self.SE1(conv4_fp))
        # # FC7层融合结果
        # # fc7_fp = torch.cat((F.relu(self.bn(self.DilationConv_512_128(sources[1])), inplace=True),
        # #                     F.relu(self.conv_1024_256(sources[3]), inplace=True),
        # #                     F.relu(self.DeConv_512_128(sources[4]), inplace=True)), 1)
        #
        # fc7_fp = torch.cat((F.relu(self.bn1(self.DilationConv_512_256(sources[1])), inplace=True),
        #                     F.relu(self.conv_1024_512(sources[3]), inplace=True),
        #                     F.relu(self.conv_512_256_fc7(self.upsample_512_512(sources[4])), inplace=True)), 1)
        #
        # # sources_final.append(F.relu(self.smooth(fc7_fp) , inplace=True))
        # fc7_fp = F.relu(fc7_fp, inplace=True)
        # sources_final.append(self.SE2(fc7_fp))
        # # conv8_2层融合结果
        # # conv8_fp = torch.cat((F.relu(self.bn(self.DilationConv_512_128_2(sources[2])), inplace=True),
        # #                       F.relu(self.conv_512_256_2(sources[4]), inplace=True),
        # #                       F.relu(self.DeConv_256_128_2(sources[5]), inplace=True)), 1)
        #
        # conv8_fp = torch.cat((F.relu(self.bn(self.DilationConv_512_128_2(sources[2])), inplace=True),
        #                       F.relu(self.conv_512_256_2(sources[4]), inplace=True),
        #                       F.relu(self.conv_256_128_2(self.upsample_256_256_2(sources[5])), inplace=True)),
        #                      1)
        # # sources_final.append(F.relu(self.smooth(conv8_fp) , inplace=True))
        # conv8_fp = F.relu(conv8_fp, inplace=True)
        # sources_final.append(self.SE3(conv8_fp))
        #
        # # conv9_2层融合
        # conv9_fp = torch.cat((F.relu(self.conv_256_128_9_2(sources[5]), inplace=True),
        #                       F.relu(self.conv_256_128_9_2(self.upsample_256_256_10_2(sources[6])),
        #                              inplace=True)), 1)
        #
        # conv9_fp = F.relu(conv9_fp, inplace=True)
        # sources_final.append(self.SE4(conv9_fp))
        #
        # # conv10_2层融合
        # conv10_fp = torch.cat((F.relu(self.conv_256_128_10_2(sources[6]), inplace=True),
        #                        F.relu(self.conv_256_128_10_2(self.upsample_256_256_11_2(sources[7])),
        #                               inplace=True)), 1)
        # conv9_fp = F.relu(conv10_fp, inplace=True)
        # sources_final.append(self.SE5(conv10_fp))
        #
        # # conv11_2
        # sources_final.append(self.SE6(sources[7]))





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
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]  # 300-》150， 150-》75，38-》19
        elif v == 'C':  # maxpooling 时边缘补NAN
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]  # 75-》38
        else:  # 卷积前后维度可以通过字典中数据设置好
            # 输入维度；输出维度；kernel_size卷积核大小；stride步长大小；补零；dilation，kernel间距；group分组卷积
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 20-》18
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # dilation=卷积核元素之间的间距,扩大卷积感受野的范围，没有增加卷积size
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


# def add_extras(cfg, i, batch_norm=False):
#     # Extra layers added to VGG for feature scaling
#     layers = []
#     in_channels = i
#     flag = False
#     for k, v in enumerate(cfg):
#         # S代表stride，为2时久相当于缩小feature map，即下采样
#         if in_channels != 'S':
#             if v == 'S':
#                 layers += [nn.Conv2d(in_channels, cfg[k + 1],
#                            kernel_size=(1, 3)[flag], stride=2, padding=1)]
#             else:
#                 layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
#             flag = not flag
#         in_channels = v
#     return layers

# 替换add_extras函数，改进为结构更加清晰的
# 参考https://zhuanlan.zhihu.com/p/76050558
def add_extras():
    exts1_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
    exts1_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
    exts2_1 = nn.Conv2d(512, 128, 1, 1, 0)
    exts2_2 = nn.Conv2d(128, 256, 3, 2, 1)
    exts3_1 = nn.Conv2d(256, 128, 1, 1, 0)
    exts3_2 = nn.Conv2d(128, 256, 3, 1, 0)
    exts4_1 = nn.Conv2d(256, 128, 1, 1, 0)
    exts4_2 = nn.Conv2d(128, 256, 3, 1, 0)

    return [exts1_1, exts1_2, exts2_1, exts2_2, exts3_1, exts3_2, exts4_1, exts4_2]


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
        # 按照fp-ssd坤问，将1024改为512通道
        if k == 1:
            in_channels = 1024
        else:
            in_channels = vgg[v].out_channels

        loc_layers += [nn.Conv2d(in_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(in_channels,
                                  cfg[k] * num_classes, kernel_size=3, padding=1)]
        # loc_layers += [nn.Conv2d(vgg[v].out_channels,
        #                          cfg[k] * 4, kernel_size=3, padding=1)]  # 特征图尺寸没动，通道改变为[4/6*4]
        # conf_layers += [nn.Conv2d(vgg[v].out_channels,
        #                 cfg[k] * num_classes, kernel_size=3, padding=1)]  # 只改变通道[4/6*num_class]
    for k, v in enumerate(extra_layers[1::2], 2):  # 找到对应的层
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]  # 只改变通道为[4/6*4]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]  # 只改变通道[4/6*num_class]
    return vgg, extra_layers, (loc_layers, conf_layers)



# # ssd.py
# def multibox(vgg, extras, num_classes):
#     loc_layers = []
#     conf_layers = []
#     #vgg_source=[21, -2] # 21 denote conv4_3, -2 denote conv7
#
#     # 定义6个坐标预测层, 输出的通道数就是每个像素点上会产生的 default box 的数量
#     loc1 = nn.Conv2d(vgg[21].out_channels, 4*4, 3, 1, 1) # 利用conv4_3的特征图谱, 也就是 vgg 网络 List 中的第 21 个元素的输出(注意不是第21层, 因为这中间还包含了不带参数的池化层).
#     loc2 = nn.Conv2d(vgg[-2].out_channels, 6*4, 3, 1, 1) # Conv7
#     loc3 = nn.Conv2d(vgg[1].out_channels, 6*4, 3, 1, 1) # exts1_2
#     loc4 = nn.Conv2d(extras[3].out_channels, 6*4, 3, 1, 1) # exts2_2
#     loc5 = nn.Conv2d(extras[5].out_channels, 4*4, 3, 1, 1) # exts3_2
#     loc6 = nn.Conv2d(extras[7].out_channels, 4*4, 3, 1, 1) # exts4_2
#     loc_layers = [loc1, loc2, loc3, loc4, loc5, loc6]
#
#     # 定义分类层, 和定位层差不多, 只不过输出的通道数不一样, 因为对于每一个像素点上的每一个default box,
#     # 都需要预测出属于任意一个类的概率, 因此通道数为 default box 的数量乘以类别数.
#     conf1 = nn.Conv2d(vgg[21].out_channels, 4*num_classes, 3, 1, 1)
#     conf2 = nn.Conv2d(vgg[-2].out_channels, 6*num_classes, 3, 1, 1)
#     conf3 = nn.Conv2d(extras[1].out_channels, 6*num_classes, 3, 1, 1)
#     conf4 = nn.Conv2d(extras[3].out_channels, 6*num_classes, 3, 1, 1)
#     conf5 = nn.Conv2d(extras[5].out_channels, 4*num_classes, 3, 1, 1)
#     conf6 = nn.Conv2d(extras[7].out_channels, 4*num_classes, 3, 1, 1)
#     conf_layers = [conf1, conf2, conf3, conf4, conf5, conf6]
#
#     # loc_layers: [b×w1×h1×4*4, b×w2×h2×6*4, b×w3×h3×6*4, b×w4×h4×6*4, b×w5×h5×4*4, b×w6×h6×4*4]
#     # conf_layers: [b×w1×h1×4*C, b×w2×h2×6*C, b×w3×h3×6*C, b×w4×h4×6*C, b×w5×h5×4*C, b×w6×h6×4*C] C为num_classes
#     # 注意pytorch中卷积层的输入输出维度是:[N×C×H×W], 上面的顺序有点错误, 不过改起来太麻烦
#     return vgg, extras, loc_layers, conf_layers

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
                                     # add_extras(extras[str(size)], 1024),
                                     add_extras(),
                                     mbox[str(size)],
                                     num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
