# encoding: utf-8
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU做训练


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default="data/VOCdevkit/",   # 修改【dataset_root】
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',  # 【预训练好的权重系数】
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=4, type=int,  # 【修改batch_size】
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,  # 【是否从某节点开始训练】没有就是None
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=2, type=int,  # 【num_workers】
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,  # 【修改学习率】
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,   # 可视化 这次设置为【【】可视化】】】
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
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
    cfg = voc  # voc是一个字典 里面包括网络的一系列参数信息
    dataset = VOCDetection(  # 是一个VOC数据的类
        root=args.dataset_root,  # 数据集的根目录
        transform=SSDAugmentation(cfg['min_dim'], MEANS))  # 图片的预处理方法(输入图片的尺寸和均值) 原本类中定义为None 后面的MEANS我人为可以删除

    if args.visdom:  # 这里是可视化工具，不用管###################
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    # 阶段【train or test】 输入图片尺寸大小 类别数
    # build_ssd是一个放在ssd.py的函数
    # return是一个类的对象，也就是class SSD(nn.Module)，ssd_net也就是SSD类的一个对象
    # ssd_net拥有所有class SSD继承于nn.Module以及作者增加方法的所有属性
    # 在SSD这个类中就定义了网络的base部分（修改全连接层后的VGG16）和extras部分（论文作者加入的多尺度feature map）和head部分
    # 对选定的6个尺度下的feature map进行卷积操作得到的每个default box 的每一个分类类别的confidence以及位置坐标的信息
    net = ssd_net  # 到这里class类SSD只完成了__init__()并没有执行__forward__() net是一个类

    if args.cuda:  # 是否将模型放到多个个GPU上运行{我认为在我的任务中不要放在多线程GPU中}
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True
    if args.resume:  # 【resume】的默认值是None,表示不是接着某个断点来继续训练这个模型 【其实checkpoint里面最好还要加上优化器的保存】
        # 【model_state_dict,optimizer_state_dict,epoch】 见深度之眼
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:  # 那么就从weights文件夹下面直接加载预训练好vgg基础网络预训练权重
        vgg_weights = torch.load(args.save_folder + args.basenet)  # 整个ssd_net中vgg基础网络的权重
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)  # 只在整个ssd_net中的vgg模块中加载预训练好的权重，其余的extra，特征融合，CBAM模块没有加载预训练权重
    if args.cuda:  # 将模型结构放在GPU上训练
        net = net.cuda()
    if not args.resume:  # ######################################################################
        print('Initializing weights...')  # 如果不是接着某个断点接着训练，那么其余extras loc con都会xavier方法初始化
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)  # extras 模块由 xavier 方法默认初始化data和bias
        ssd_net.loc.apply(weights_init)  # loc 模块由 xavier 方法默认初始化data和bias
        ssd_net.conf.apply(weights_init)  # conf 模块由 xavier 方法默认初始化data和bias

    # 【优化器】net.parameters()是网络结构中的参数，学习率，动量，权重衰减率
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # 定义损失函数部分【MultiBoxesLoss是一个类用于计算网络的损失，criterion是一个对象】
    # 【损失函数】 关键！！！ criterion是个nn.Moudule的形式 里面包括两部分loss_c 和 loss_l
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)
    # 前向传播
    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')
    epoch_size = len(dataset) // args.batch_size  # 每个epoch中有多少个batch
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)  # 讲设定的参数打印出来

    step_index = 0
    # 可视化部分
    if args.visdom:  # 默认值为False
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,  # 默认值我修改成了0
                                  shuffle=True,
                                  collate_fn=detection_collate,  # collate_fn将一个batch_size数目的图片进行合并成batch
                                  pin_memory=True)
    batch_iterator = iter(data_loader)  # batch迭代器 依次迭代batch
    for iteration in range(args.start_iter, cfg['max_iter']):  # 由最大迭代次数来迭代训练
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):   # 因为args.visdom一直设置为False因此没有被调用
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None, 'append', epoch_size)
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:  # 通过多少次epoch调节一次学习率
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
            # targets 和image都是读取的训练数据
        except StopIteration:
            bath_iterator = iter(data_loader)
            images, targets = next(bath_iterator)
        # images=【batch_size,3,300,300】
        # targets=【batch_size,num_object,5】
        # num_object代表一张图里面有几个ground truth，5代表四个位置信息和一个label
        if args.cuda:  # 将数据放在cuda上
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        # ##out是netforward的输出：是个元组，里面包括3个部分[loc conf  priors]
        out = net(images)
        # ## backprop 优化器梯度清零
        optimizer.zero_grad()
        # ## criterion是nn.Module形式，下面是调用它的forward模式【重点看，里面包括难例挖掘的内容】
        # ###################################【【【训练阶段的损失！！！】】】######################################
        # ##输入参数1：网络结构net输出的out:[loc conf priors]
        # ##输入参数2：targets:真实目标的位置标签值
        loss_l, loss_c = criterion(out, targets)  # criterion就是MultiBoxLoss类定义的对象，forward前传播返回的结果是【loss_l, loss_c】
        loss = loss_l + loss_c  # 总loss
        loss.backward()
        optimizer.step()
        t1 = time.time()
        # 下面两行好像没有使用
        loc_loss += loss_l.data  # ###到底是改成item()还是data
        conf_loss += loss_c.data  # ###到底是改成item()还是data

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % loss.data, end=' ')  # 到底是改成item()还是data

        if args.visdom:
            update_vis_plot(iteration, loss_l.data, loss_c.data, iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 10000 == 0:
            # 迭代多少次保存一次模型。 在尝试阶段，为了节省时间，建议将根据迭代次数保存模型的参数调低，例如调节到500
            print('Saving state, iter:', iteration) # 保存的checkpoint
            torch.save(ssd_net.state_dict(), 'weights/ssd300_VOC_' + repr(iteration) + '.pth')  # 保存模型的路径
    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.dataset + '.pth')  # 最后的保存：不是保存整个模型，只是保存了参数


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
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
