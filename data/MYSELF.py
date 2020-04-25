# # import os.path as osp
# # import sys
# # import torch
# # import torch.utils.data as data
# # import cv2
# # import numpy as np
# # if sys.version_info[0] == 2:
# #     import xml.etree.cElementTree as ET
# # else:
# #     import xml.etree.ElementTree as ET
# # image_sets=['2007', 'trainval'],#,('2012', 'trainval') 要选用的数据集
# # root="D:/Deep_learning/ssd.pytorch-master/data/VOCdevkit/"
# # ids = list()
# # for (year, name) in image_sets:
# #     rootpath = osp.join(root, 'VOC' + year)
# #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
# #         ids.append((rootpath, line.strip()))
# # print(ids[0])
# #
# # img_id = ids[927] #('D:/Deep_learning/ssd.pytorch-master/data/VOCdevkit/VOC2007', '000001')
# # anno = osp.join('%s', 'Annotations', '%s.xml')
# # img = osp.join('%s', 'JPEGImages', '%s.jpg')
# # target = ET.parse(anno % img_id).getroot() #读取xml文件
# # img = cv2.imread(img % img_id)#获取图像
# # cv2.imshow('pwn',img)
# # height, width, channels = img.shape
# # print(height)
# # print(width)
# # print(channels)
# # cv2.waitKey (0)
# #
# # VOC_CLASSES1 = (  # always index 0
# #     'aeroplane', 'bicycle', 'bird', 'boat',
# #     'bottle', 'bus', 'car', 'cat', 'chair',
# #     'cow', 'diningtable', 'dog', 'horse',
# #     'motorbike', 'person', 'pottedplant',
# #     'sheep', 'sofa', 'train', 'tvmonitor')
# # VOC_CLASSES2=('ship','pwn')
# #
# # what=dict(zip(VOC_CLASSES1, range(len(VOC_CLASSES1))))
# # what2=dict(zip(VOC_CLASSES2, range(len(VOC_CLASSES2))))
# # print(what)
# # print(what2)
# #######################################################################################################################
# # from __future__ import division
# # from math import sqrt as sqrt
# # from itertools import product as product
# # import torch
# # mean = []
# # clip=True
# # for i, j in product(range(5), repeat=2):  # 生成平面的网格位置坐标 i=[0 0 0 0 0 1 1 1 1 1 2 2 2 2 2 3 3 3 3 3 4 4 4 4 4]
# #     f_k = 300 / 64 #37.5                                           j=[0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4 0 1 2 3 4]
# #     cx = (j + 0.5) / f_k #
# #     cy = (i + 0.5) / f_k #
# #     s_k =162 / 300#0.1
# #     mean += [cx, cy, s_k, s_k]
# #     # aspect_ratio: 1
# #     # rel size: sqrt(s_k * s_(k+1))
# #     s_k_prime = sqrt(s_k * (213/300))#0.14
# #     mean += [cx, cy, s_k_prime, s_k_prime]
# #
# #     # rest of aspect ratios
# #     for ar in [2,3]:  # 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
# #         mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
# #         mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
# #
# # output = torch.Tensor(mean).view(-1, 4)
# # if clip:
# #     output.clamp_(max=1, min=0)
# import torch as t
# list1=[t.full([2,2,2],1),t.full([2,2,2],2)]
# list2=[t.full([2,2,2],3),t.full([2,2,2],4)]
# list3=[t.full([2,2,2],5),t.full([2,2,2],6)]
# loc=[]
# conf=[]
# pwn=zip(list1,list2,list3)
# print(pwn)
#
# # for (x,l,c) in zip(list1,list2,list3):
# #     loc.append(l(x))
# #     conf.append(c(x))
#
# # import torch
# # x = torch.tensor([[1,2,3],[4,5,6]])
# # x.is_contiguous()  # True
# # print(x)
# # print(x.transpose(0,1))
# # print(x.transpose(0, 1).is_contiguous()) # False
# # print(x.transpose(0, 1).contiguous().is_contiguous())  # True

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
import visdom as viz

list1=torch.arange(0,8)
x = torch.Tensor([[1], [2], [3]])
y = x.expand(3, 4)
print("x.size():", x.size())
print("y.size():", y.size())

print(x)
print(y)