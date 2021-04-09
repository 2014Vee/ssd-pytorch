from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch
# 参考https://blog.csdn.net/goodxin_ie/article/details/89577922

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
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
        for k, f in enumerate(self.feature_maps): # 遍历6个特征图，每个特征图分别生成默认狂
            for i, j in product(range(f), repeat=2): # 针对每个特征图生成所有的坐标
                """
                    将特征图的坐标对应回原图坐标，然后缩放成0-1的相对距离
                    原始公式应该为cx = (j+0.5) * step /min_dim，这里拆分成两步计算
                """
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                """第一种ratio为1的框"""
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                """产生第二种ratio为1的默认框"""
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        """将产生的默认框转化为n行4列的标准形式，每行代表一个默认框[x,y,w,h]"""
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0) # 如果超出范围的坐标位置和prior的宽高限制在0-1之间
        return output # 最后返回的output就是默认anchor的位置和尺寸
