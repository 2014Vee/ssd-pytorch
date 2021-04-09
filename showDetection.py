# -*- coding: utf-8 -*-
# @Time    : 2020/11/11 9:00
# @Author  : 2014Vee
# @Email   : 1976535998@qq.com
# @File    : showDetection.py
# @Software: PyCharm

import cv2
import os
# 检测结果地址
detection_results = './test_v2_14000'
if not os.path.exists(detection_results):
    os.mkdir(detection_results)
with open('./eval/test_v2_14000.txt', 'r') as f:
    lines = f.readlines()
    pic = list()
    loc = list()
    match = {}
    print(len(lines))
    for i in range(0, len(lines)):
        if 'GROUND TRUTH FOR:' in lines[i]:
            pic.append(lines[i][-7:-1])
            print(str(pic))
        if 'island score' in lines[i]:
            location=lines[i].split(' ')[5:12:2]
            conf = lines[i].split()[4][7:-1]
            location=[float(x) for x in location]
            location.append(float(conf))
            # print(location)
            loc.append(location)
        if (len(lines[i]) == 1 and i != 0) or i == len(lines)-1:
            match[pic[0]] = loc
            print(pic[0]+"**"+str(match[pic[0]]))
            pic=list()
            loc=list()
    f.close()
label = "island"
for i in match.keys():
    #print('D:/Deep_learning/ssd.pytorch-master/data/VOCdevkit/VOC2007/ground_truth/'+i+'.jpg.jpg')
    img=cv2.imread('/usr/idip/idip/liuping/dataset/VOCdevkit/VOC2007/JPEGImages/'+i+'.jpg')

    new_f = open(detection_results+'/'+i+'.txt', "a")
    # print(match[i],'每一幅图的框个数： ', len(match[i]))
    for num in range(len(match[i])):
        x1=int(match[i][num][0])
        y1=int(match[i][num][1])
        x2=int(match[i][num][2])
        y2=int(match[i][num][3])
        confidence = float(match[i][num][-1])
        new_f.write("%s %s %s %s %s %s\n" % (label, confidence, x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
        cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=1)

    cv2.imwrite(detection_results+'/'+i+'.jpg', img)