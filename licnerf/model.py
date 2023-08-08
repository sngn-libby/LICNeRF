# Modified from Mip-NeRF (https://github.com/google/mipnerf)
# Copyright (c) 2021 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import random

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import src.model.mipnerf.helper as helper
import utils.store_image as store_image
from src.model.interface import LitModel
from src.model.mipnerf.model import MipNeRF

from src.compressai.models.sensetime import JointCheckerboardHierarchicalPriors

__all__ = [
    "MipNeRFTransformedLIC",
]


class MipNeRFTransformedLIC(JointCheckerboardHierarchicalPriors):
    def __init__(self,
                 # NeRF configs
                 num_samples: int = 128,
                 num_levels: int = 2,
                 resample_padding: float = 0.01,
                 stop_level_grad: bool = True,
                 use_viewdirs: bool = True,
                 lindisp: bool = False,
                 ray_shape: str = "cone",
                 min_deg_point: int = 0,
                 max_deg_point: int = 16,
                 deg_view: int = 4,
                 density_noise: float = 0,
                 density_bias: float = -1,
                 rgb_padding: float = 0.001,
                 disable_integration: bool = False,

                 # LIC configs
                 N=192,
                 M=192,
                 **kwargs
                 ):
        super().__init__(N=N, M=M, **kwargs)
        self.nerf = MipNeRF(num_samples,
                            num_levels,
                            resample_padding,
                            stop_level_grad,
                            use_viewdirs,
                            lindisp,
                            ray_shape,
                            min_deg_point,
                            max_deg_point,
                            deg_view,
                            density_noise,
                            density_bias,
                            rgb_padding,
                            disable_integration,)

        self.density_mixer = nn.Sequential( # input: concat(x_a, d_a) / output: y
            nn.Conv2d(2 * M, M * 5 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 5 // 3, M * 4 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 4 // 3, M, 1),
        )


