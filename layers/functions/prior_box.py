from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    """cfg= voc = {
    'num_classes': 2, #【改成自己训练的类别数】
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000, #【改成自己训练的迭代次数】
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',}
     """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios']) # 6
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):#每个特征层的尺寸大小  'feature_maps': [38, 19, 10, 5, 3, 1],
            for i, j in product(range(f), repeat=2):#生成平面的网格位置坐标
                f_k = self.image_size / self.steps[k]  #300/[8, 16, 32, 64, 100, 300] f_k=[37.5, 18.75, 9.375, 4.6875, 3, 1]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size#'min_sizes': [30, 60, 111, 162, 213, 264]/300=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88]=s_k
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))#'max_sizes': [60, 111, 162, 213, 264, 315]/300=[0.2 0.37 0.54 0.71 0.88 1.05]
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:#'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)  #对超出范围的点坐标位置和prior的宽高限制在0-1之间
        return output #返回的output中是所有的生成框anchor的位置和尺寸
