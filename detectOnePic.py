# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 9:19
# @Author  : 2014Vee
# @Email   : 1976535998@qq.com
# @File    : detectOnePic.py
# @Software: PyCharm

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
import torch.utils.data as data

from ssd import SSD
from  data.config import voc

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

cfg = voc
# 实例化模型
net = SSD(cfg)
# 使用cpu或gpu
net.to('cuda')
# 模型从权重文件中加载权重
net.load_pretrained_weight('weights/ssd300_COCO_55000.pth')
# 打开图片
image = cv2.imread("./data/VOCdevkit/VOC2007/JPEGImage/000059.jpg")
# 进行检测, 分别返回 绘制了检测框的图片数据/回归框/标签/分数.
drawn_image, boxes, labels, scores = net.Detect_single_img(image=image, score_threshold=0.5)

cv2.imwrite("./data/VOCdevkit/VOC2007/PREDECTION/"+'000059'+'.jpg', drawn_image)
# cv2.imshow(drawn_image)
# cv2.show()