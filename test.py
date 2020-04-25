# test.py 用于测试单张图片的效果
from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import VOC_ROOT, VOC_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
import torch.utils.data as data
from ssd import build_ssd

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_VOC_10000.pth',  # #########修改检测模型的路径
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default='data/VOCdevkit', help='Location of VOC root directory')  # ###修改读取图片的路径【VOC_ROOT】
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


# 输入参数：【VOC数据集root，网络，cuda，输入的测试数据，预处理函数， 阈值】
def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'   # 保存的txt文件名
    num_images = len(testset)    # 测试的数据数量
    for i in range(num_images):  # 依次遍历每一张图片
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))  # 索引加1才是加载显示的图片数从1开始

        img = testset.pull_image(i)  # pull_image的功能是cv2.imread读取某张图片 放到img中
        img_id, annotation = testset.pull_anno(i)  # pull_anno的功能是读取标签信息 img_id

        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)  # #先去掉第一个维度，然后将最后个通道的提前
        x = Variable(x.unsqueeze(0))  # #然后在将第一维度添加1维度还原成4维度

        with open(filename, mode='a') as f:   # 下面是添加GT的信息 真实目标的标签
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()  # 将单张的图片转化为GPU上的数据

        y = net(x)      # forward pass 将数据放在网上前向传播
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):  #
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1  # +1 background【这里我觉得也不应该加1】
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))  # 在创建的网络中添加前面训练过的权重系数
    net.eval()  # 开始机进行eval()模式
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())  # 将 第二个参数的默认值改变成【要选用的数据集】
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    # #输入参数：【VOC数据集root，网络，cuda，输入的测试数据，预处理函数， 阈值】
    test_net(args.save_folder,  # VOC数据集root
             net,               # 网络
             args.cuda,         # cuda
             testset,           # 输入的测试数据
             BaseTransform(net.size, (104, 117, 123)),  # 预处理函数
             thresh=args.visual_threshold               # 阈值
             )


if __name__ == '__main__':
    test_voc()
