"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME, MEANS, voc, COLORS   # ################################3
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# VOC_CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')

# ##VOC_CLASSES=( 'None','ship')#这里如果只写一个名字的话就有很严重的问题
# ##VOC_CLASSES=('ship')###就会出现严重的错误 需要在ship后面添加‘，’
VOC_CLASSES=('face',
             'face_mask')  # ##**************************************************************
# note: if you used our download scripts, this should be right
VOC_ROOT = "/data/lp/project/ssd.pytorch/data/VOCdevkit/" # 个人感觉路径应该自己设定一下
# VOC_ROOT = osp.join(HOME, "data/VOCdevkit/")  # ###去掉了HOME####我认为应该修改成自己的数据位子 【这里是数据默认的读取路径】


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(  # class_to_ind将标签信息转化为字典形式{'aeroplane': 0, 'bicycle': 1}
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()  # ##strip()返回移除字符串头尾指定的字符生成的新字符串。 # 去除首尾空格
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height  # 获得该目标在这张图向上的相对坐标位置【0-1】
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self,
                 root,  # VOCdevkit folder的根目录
                 image_sets=[('2007', 'trainval')],  # ('2012', 'trainval') 要选用的数据集 是字符串的格式
                 transform=None,  # 图片的预处理方法
                 target_transform=VOCAnnotationTransform(),  # 标签的预处理方法
                 dataset_name='VOC0712'):  # 数据集的名字
        self.root = root  # 设置数VOCdevkit folder的根目录
        self.image_set = image_sets  # 设置要选用的数据集
        self.transform = transform  # 定义图像转换方法
        self.target_transform = target_transform  # 定义标签的转换方法
        self.name = dataset_name  # 定义数据集名称
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')  # 记录标签的位置 留下了两个【%s】的部分没有填写
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')  # 记录图像的位置 留下了两个【%s】的部分没有填写
        self.ids = list()  # 记录数据集中的所有图像的名字,没有后缀名
        for (year, name) in image_sets:   # 【image_sets】就是我们要用训练好的模型测试的test数据集('2007', 'trainval')这样的形式
            # 读入数据集中的图像名称，可以依照该名称和_annopath、_imgpath推断出图片、描述文件存储的位置
            rootpath = osp.join(self.root, 'VOC' + year) # ...../VOC2007
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                # ...../VOC2007/ImageSets/Main/(test val train).txt
                self.ids.append((rootpath, line.strip()))
                # 将这个测试集的txt文本打开后，读取每一行的数据，注意去除前后的空格
                # 【（ids）存放每一张图片的信息 （rootpath 和 去后缀的图片名 没有.jpg）为一个元组】ids是个list里面的每个元素都是一个元组

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]  # ('D:/Deep_learning/ssd.pytorch-master/data/VOCdevkit/VOC2007', '000001')

        target = ET.parse(self._annopath % img_id).getroot()  # 将self._annopath空缺的两个部分用img_id补全。  获得需要读取xml的对象
        img = cv2.imread(self._imgpath % img_id)  # 将self._imgpath的空缺的两个部分用img_id补全      获取对应的图像
        height, width, channels = img.shape  # 获取图像的尺寸  高宽通道数

        if self.target_transform is not None:  # 对读入的测试的标签进行处理
            target = self.target_transform(target, width, height)  # 返回的是【xmin，xmax，ymin，ymax，label】
        if self.transform is not None:  # 对测试集的图片默认的transform是None
            target = np.array(target)  # 下面这个transform理解不了啊！！！！！
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])  # 输入的图像，位置（4维度），标签（1维度）
            # to rgb
            img = img[:, :, (2, 1, 0)]  # opencv读入图像的顺序是BGR，该操作将图像转为RGB
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))  # 首先将narry的向量（x，）转化为（x,1）,然后又重新组织成了一样的格式
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width  # permute将通道数提前，符合pytorch的格式
        # 将通道数提前，为了统一torch的后续训练操作。
# ###############################################################################################################################################

    def pull_image(self, index):
        """Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        """
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)  # 从图片的路径直接读取图片

    def pull_anno(self, index):
        """Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        """
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)  # ##后面的1,1,就是上面实例化VOCAnno的后两个参数width和height，用于对标签进行归一化
        return img_id[1], gt  # 返回的是图片的名字（去后缀名），还有groundtruth [[xmin, ymin, xmax, ymax, label_ind], ... ]

    def pull_tensor(self, index):
        """Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        """
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
