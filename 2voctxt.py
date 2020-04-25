#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   2voctxt.py    
@Version :   1.0 
@Author  :   2014Vee
@Contact :   1976535998@qq.com
@License :   (C)Copyright 2014Vee From UESTC
@Modify Time :   2020/4/17 15:37
@Desciption  :   None
'''
import os
import random

# https://blog.csdn.net/duanyajun987/article/details/81507656
#*这里小心一下，里面的训练集和数据集文件的比例如下
#这里由于数据集NWPU里已经有测试集所以不需要在把数据留一部分测试
trainval_percent = 0.2
train_percent = 0.8
xmlfilepath = './data/VOCdevkit/VOC2007/Annotations'
txtsavepath = './data/VOCdevkit/VOC2007/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + '/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftest.write(name)
        else:
            fval.write(name)
    else:
        ftrain.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
