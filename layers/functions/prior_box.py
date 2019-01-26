from __future__ import division
from math import sqrt as sqrt
from itertools import product as product

import torch

import pdb


class PriorBox(object):
    '''
    Compute prior box coordinates in center-offset form for each source feature map.
    '''
    def __init__(self, cfg):
        super(PriorBox, self).__init__()

        self.image_size = cfg['min_dim']
        # Number of priors for feature map location (either 4 or 6)
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
        for k, f in enumerate(self.feature_maps):
            # Cartesian product; e.g., if f = 2, product(range(f), repeat = 2) = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for i, j in product(range(f), repeat = 2):  # (i, j) is each location of feature maps
                f_k = self.image_size / self.steps[k]
                # Unit center x,y
                cx = (j + 0.5) / f_k    # Scale to almost to (0 ~ 1)
                cy = (i + 0.5) / f_k

                # Aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size   # Scale to almost to (0 ~ 1)
                mean += [cx, cy, s_k, s_k]

                # Aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # Rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # Back to torch land
        output = torch.Tensor(mean).view(-1, 4)

        if self.clip:
            output.clamp_(max = 1, min = 0)

        return output