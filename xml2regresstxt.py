# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
# '''
# @File    :   bbox-regress.py
# @Version :   1.0
# @Author  :   2014Vee
# @Contact :   1976535998@qq.com
# @License :   (C)Copyright 2014Vee From UESTC
# @Modify Time :   2020/4/14 9:44
# @Desciption  :   生成回归框训练的数据文件
# '''
# import os
# import random
#
# xmlfilepath = r'/data/lp/project/ssd.pytorch/xml_zc_fz'
# saveBasePath = r'/data/lp/project/ssd.pytorch/txtsave'
#
# trainval_percent = 1.0
# train_percent = 0.9
# total_xml = os.listdir(xmlfilepath)
# num = len(total_xml)
# list = range(num)
# tv = int(num * trainval_percent)
# tr = int(tv * train_percent)
# trainval = random.sample(list, tv)
# train = random.sample(trainval, tr)
#
# print("train and val size", tv)
# print("train size", tr)
# ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
# ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
# ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
# fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')
#
# for i in list:
#     name = total_xml[i][:-4] + '\n'
#     if i in trainval:
#         ftrainval.write(name)
#         if i in train:
#             ftrain.write(name)
#         else:
#             fval.write(name)
#     else:
#         ftest.write(name)
#
# ftrainval.close()
# ftrain.close()
# fval.close()
# ftest.close()
# # test


tensors_list = [[[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]], [[[25,26],[27,28],[29,30]],[[31,32],[33,34],[35,36]],[[37,38],[39,40],[41,42]],[[43,44],[45,46],[47,48]]], [[[49,50],[51,52],[53,54]],[[55,56],[57,58],[59,60]],[[61,62],[63,64],[65,66]],[[67,68],[69,70],[71,72]]], [[[73,74],[75,76],[77,78]],[[79,80],[81,82],[83,84]],[[85,86],[87,88],[89,90]],[[91,92],[93,94],[95,96]]], [[[97,98],[99,100],[101,102]],[[103,104],[105,106],[107,108]],[[109,110],[111,112],[113,114]],[[115,116],[117,118],[119,120]]]]
print(tensors_list)