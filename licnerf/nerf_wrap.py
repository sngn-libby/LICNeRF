import os
import random

from typing import (
    Union, Optional, Callable, Any
)

import gin
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from pytorch_lightning.core.optimizer import (
    Optimizer, LightningOptimizer
)

from src.model.interface import LitModel
from src.model.mipnerf import MipNeRF, LitMipNeRF


@gin.configurable()
class LitMipNeRF(LitMipNeRF):
    def __init__(
            self,
            lr_init: float = 5.0e-4,
            lr_final: float = 5.0e-6,
            lr_delay_steps: int = 2500,
            lr_delay_mult: float = 0.01,
            coarse_loss_mult: float = 0.1,
            randomized: bool = True,
            use_multiscale: bool = False,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super().__init__()

    def generate_density(self, batch, batch_idx):

        
    def training_step(self, batch, batch_idx):
        rendered_results = self.model
