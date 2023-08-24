import os
import random

import gin
import numpy as np
from typing import (
    Union, Optional, Callable, Any
)

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from pytorch_lightning.core.optimizer import (
    Optimizer, LightningOptimizer
)

from src.model.interface import LitModel
from src.compressproj.utils.eval_model.__main__ import *
from src.compressproj.models.google import JointAutoregressiveHierarchicalPriors


@gin.configurable()
class LitJointAutoregressiveHierarchicalPriors(LitModel):

    def __init__(self,
                 lr_init: float=5.0e-4,
                 lr_final: float=5.0e-6,
                 lr_scheduler_gamma: float=1/3,
                 lr_milestones=[150, 180, 210, 240],
                 N=192,
                 M=192,
                 **kwargs,
                 ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super().__init__(**kwargs)
        self.model = JointAutoregressiveHierarchicalPriors(N, M)

        # self.automatic_optimization = False

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

        metrics = compute_metrics(x, x_hat)
        _psnr = metrics["psnr-rgb"]
        _msssim = metrics["ms-ssim-rgb"]
        self.log("train/psnr",      _psnr, on_step=True, prog_bar=True, logger=True)
        self.log("train/ms-ssim",   _msssim, on_step=True, prog_bar=True, logger=True)
        self.log("train/loss",      loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.model.update(force=True)

        x = batch['x']
        out = inference(self.model, x)

        return out

    def validation_epoch_end(self, out):
        self.log("val/psnr",        out['psnr-rgb'], on_epoch=True, sync_dist=True)
        self.log("val/ms-ssim",     out['ms-ssim-rgb'], on_epoch=True, sync_dist=True)
        self.log("val/bpp",         out['bpp'], on_epoch=True, sync_dist=True)
        self.log("val/enc_time",    out['encoding_time'], on_epoch=True, sync_dist=True)
        self.log("val/dec_time",    out['decoding_time'], on_epoch=True, sync_dist=True)

    def test_epoch_end(self, out):
        self.log("test/psnr",       out['psnr-rgb'], on_epoch=True, sync_dist=True)
        self.log("test/ms-ssim",    out['ms-ssim-rgb'], on_epoch=True, sync_dist=True)
        self.log("test/bpp",        out['bpp'], on_epoch=True, sync_dist=True)
        self.log("test/enc_time",   out['encoding_time'], on_epoch=True, sync_dist=True)
        self.log("test/dec_time",   out['decoding_time'], on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        parameters = {
            n
            for n, p in self.model.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        aux_parameters = {
            n
            for n, p in self.model.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }

        # Make sure we don't have an intersection of parameters
        params_dict = dict(self.model.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0

        optimizer = optim.Adam(
            (params_dict[n] for n in sorted(parameters)),
            lr=gin.query_parameter('run.learning_rate'),
        )
        aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=gin.query_parameter('run.aux_learning_rate'),
        )
        return optimizer, aux_optimizer

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_idx: int = 0,
        optimizer_closure: Optional[Callable[[], Any]] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        max_epochs = gin.query_parameter('run.max_epochs')
        idx = [i if epoch < milestone else max_epochs for i, milestone in enumerate(self.milestones)]
        idx = min(idx)
        lr = self.lr_init * (self.lr_scheduler_gamma ** idx)
        lr = lr if lr > self.lr_final else self.lr_final
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step(closure=optimizer_closure)

    def load_state_dict(
            self,
            state_dict,
            strict: bool = True,
    ):
        N = state_dict['model.g_a.0.weight'].size(0)
        M = state_dict['model.g_a.6.weight'].size(0)
        model = self(N, M)
        model.load_state_dict(state_dict)
        self.model = model



