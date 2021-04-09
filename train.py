from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd_v2 import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 指定GPU训练


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],  # 数据集格式VOC
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,    # 数据集目录
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',    # 与训练网络权重系数
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,   # batchsize大小
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,   # 设置从某一结点开始训练，记住保存中间权重
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,  # 继续训练的起始节点
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,   # 学习率设置
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,  # 可视化功能这部分还没有用到
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights-eca/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():


    cfg = voc # config中配置信息
    dataset = VOCDetection(
        root=args.dataset_root,
        transform=SSDAugmentation(cfg['min_dim'], MEANS)) # 实例化图片预处理的类，MEANS参数在构造函数中并没出现，不知道咋接

    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])  # 先进行实例化并完成构造函数的赋初值
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')  # 如果不是接着某个断点继续训练，那么其余的extrs loc con都会按照xavier的方法初始化
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init) # 具体模块由xavier方法默认初始化data和bias
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # 优化器参数配置
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # 损失函数的计算部分
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate, # 这里shuffle改了改为false
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader) # batch迭代器 依次迭代batch
    for iteration in range(args.start_iter, cfg['max_iter']):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                            'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:  # 通过多少次epoch来调节一次学习率
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        # targets 和image都是读取的训练数据
        # images=[batch_size,3,300,300]
        # targets=[batch_size,num_object,5]
        # num_object代表一张图里面有几个ground truth，5代表四个位置信息和一个label
        try:
            images, targets = next(batch_iterator)
        except StopIteration:  # 避免迭代器没有更多值的报错
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        # ## criterion是nn.Module形式，下面是调用它的forward模式【重点看，里面包括难例挖掘的内容】
        # ###################################【【【训练阶段的损失！！！】】】######################################
        # ##输入参数1：网络结构net输出的out:[loc conf priors]
        # ##输入参数2：targets:真实目标的位置标签值

        loss_l, loss_c = criterion(out, targets) # 返回的损失函数即位置损失和类别损失
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

        # if args.visdom:
        #     update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
        #                     iter_plot, epoch_plot, 'append')

        if args.visdom:
            # train_loss += loss.data[0] 是pytorch0.3.1版本代码,在0.4-0.5版本的pytorch会出现警告,不会报错,
            # 但是0.5版本以上的pytorch就会报错,总的来说是版本更新问题.
            # 直接把data[0]换成item()即可
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, epoch_plot, 'append')
        # 设置迭代多少次保存一次模型。 在尝试阶段可以设置低一点，观察loss
        # repr是将对象转化供解释器读取的形式。
        if iteration != 0 and iteration % 1000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_eca_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step): # 这个模块根据训练步骤进行学习率调整的经常用，记住
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
