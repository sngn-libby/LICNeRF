import os
import random

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from src.model.interface import LitModel
from src.compressproj.models.google import JointAutoregressiveHierarchicalPriors


@gin.configurable()
class LitJointAutoregressiveHierarchicalPriors(LitModel):

    def __init__(self,
                 lr_init: float=5.0e-4,
                 lr_final: float=5.0e-6,
                 lr_delay_steps: int=2500,
                 lr_delay_mult: float=0.01,
                 N=192,
                 M=192,
                 **kwargs,
                 ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super().__init__(**kwargs)
        self.model = JointAutoregressiveHierarchicalPriors(N, M)

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.model.update(force=True)

        x = batch['x']
        out = self.model(x)
        x_hat = out['x_hat']
        likelihoods = out['likelihoods']
        loss = self.criterion({
            'x_hat': x_hat,
            'likelihodds': likelihoods,
        })
        return loss
