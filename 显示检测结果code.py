import cv2
with open('D:/Deep_learning/ssd.pytorch-master/eval/test1.txt','r') as f:
    line=f.readline()
    pic=list()
    loc=list()
    match={}
    while line:
        if 'GROUND TRUTH FOR:' in line:
            pic.append(line[-7:-1])
        if 'ship score' in line:
            location=line.split(' ')[5:12:2]
            location=[float(x) for x in location]
            loc.append(location)
        if  len(line)==1: 
            match[pic[0]]=loc
            pic=list()
            loc=list()
        line=f.readline()
    f.close()
for i in match.keys():  
    #print('D:/Deep_learning/ssd.pytorch-master/data/VOCdevkit/VOC2007/ground_truth/'+i+'.jpg.jpg')
    img=cv2.imread('D:/Deep_learning/ssd.pytorch-master/data/VOCdevkit/VOC2007/ground_truth/'+i+'.jpg.jpg')
    #print(match[i],'每一幅图的框个数： ',len(match[i]))
    for num in range(len(match[i])):
        x1=int(match[i][num][0])
        y1=int(match[i][num][1])
        x2=int(match[i][num][2])
        y2=int(match[i][num][3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
    cv2.imwrite("D:/Deep_learning/ssd.pytorch-master/data/VOCdevkit/VOC2007/PREDECTION/"+i+'.jpg.jpg', img)    
