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
import torch.optim as optim
from typing import Union, Optional, Callable, Any
from pytorch_lightning.core.optimizer import Optimizer, LightningOptimizer
from pytorch_msssim import ms_ssim

import src.model.mipnerf.helper as helper
import utils.store_image as store_image
from src.model.interface import LitModel
from src.model.mipnerf.model import MipNeRF

from src.compressproj.layers import GDN, MaskedConv2d
from src.compressproj.models.utils import conv, deconv
from src.compressproj.losses import RateDistortionLoss
from src.compressproj.models.sensetime import JointCheckerboardHierarchicalPriors

__all__ = [
    "TransformNeRFLIC",
    "MipNeRFTransformedLIC",
]

@gin.configurable()
class TransformNeRFLIC(nn.Module):
    def __init__(self,
                 lic_model,
                 nerf_model,
                 gamma=0.8,
                 learning_rate=1e-4,
                 aux_learning_rate=1e-3,
                 lmbda=0.0130,

                 lr_init: float = 5.0e-4,
                 lr_final: float = 5.0e-6,
                 lr_delay_steps: int = 2500,
                 lr_delay_mult: float = 0.01,
                 coarse_loss_mult: float = 0.1,
                 randomized: bool = True,
                 use_multiscale: bool = False,
                 train_nerf: bool = True,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        self.criterion = RateDistortionLoss(self.lmbda)

        N = self.lic_model.N
        M = self.lic_model.M

        self.lic_model.g_a = nn.Sequential(
            # conv(11, N, kernel_size=5, stride=2),
            conv(19, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.lic_model.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 4, kernel_size=5, stride=2),
        )

        self.lic_model.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=2, kernel_size=3),
        )


@gin.configurable()
class MipNeRFTransformedLIC(LitModel):
    r"""
    .. transformation code-block:: none
            x ──────────────────┐    ┌───┐    y
                    ┌────┐      ├──►─┤g_a├──►─┐
            x ─────►┤nerf├──►───┘    └───┘    │
                    └────┘ density            :
                                        y_hat ▼
                      ┌────┐ density ┌───┐    │
            x_hat_1 ──┤nerf├──────┬◄─┤g_s├────┘
                      └────┘      │  └───┘
            x_hat ────────────────┘

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """
    def __init__(self,
                 lic_model,
                 nerf_model,
                 gamma=0.8,
                 learning_rate=1e-4,
                 aux_learning_rate=1e-3,
                 lmbda=0.0130,

                 lr_init: float = 5.0e-4,
                 lr_final: float = 5.0e-6,
                 lr_delay_steps: int = 2500,
                 lr_delay_mult: float = 0.01,
                 coarse_loss_mult: float = 0.1,
                 randomized: bool = True,
                 use_multiscale: bool = False,
                 train_nerf: bool = True,
                 **kwargs):

        super().__init__(**kwargs)

        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        self.criterion = RateDistortionLoss(self.lmbda)

        N = self.lic_model.N
        M = self.lic_model.M

        self.lic_model.g_a = nn.Sequential(
            # conv(11, N, kernel_size=5, stride=2),
            conv(19, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.lic_model.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, 4, kernel_size=5, stride=2),
        )

        self.lic_model.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=2, kernel_size=3),
        )

    def setup(self, stage):
        self.near = self.trainer.datamodule.near
        self.far = self.trainer.datamodule.far
        self.white_bkgd = self.trainer.datamodule.white_bkgd

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.lic_model.train()
        self.lic_model.update(force=True)
        if self.train_nerf:
            self.nerf_model.train()
        else:
            self.nerf_model.eval()

        x = batch["target"]

        # nerf
        ret = self(
            x, batch, self.randomized, self.white_bkgd, self.near, self.far
        )
        rendered_results = ret["render"]
        rgb_coarse = rendered_results[0][0]
        rgb_fine = rendered_results[1][0]
        mask = batch["multloss"] if self.use_multiscale else None
        if len(rgb_coarse.shape) < len(mask.shape):
            mask = torch.squeeze(mask, dim=-1)

        loss0 = helper.img2mse(rgb_coarse, x, mask)
        loss1 = helper.img2mse(rgb_fine, x, mask)
        nerf_loss = loss0 * self.coarse_loss_mult + loss1
        with torch.no_grad():
            if self.use_multiscale:
                loss0 = helper.img2mse(rgb_coarse, x, None)
                loss1 = helper.img2mse(rgb_fine, x, None)
            psnr0 = helper.mse2psnr(loss0)
            psnr1 = helper.mse2psnr(loss1)

        if self.train_nerf:
            self.log("train/nerf_psnr(c)", psnr0, on_step=True, prog_bar=True, logger=True)
            self.log("train/nerf_psnr(f)", psnr1, on_step=True, prog_bar=True, logger=True)
            self.log("train/nerf_loss", nerf_loss, on_step=True)

        # compression
        for k in ret.keys():
            if torch.is_tensor(ret[k]) and ret[k].dtype == torch.float64:
                ret[k] = ret[k].type(torch.float32)
        x_hat = ret['x_hat']
        lic_loss = self.criterion({
            "x_hat": ret["x_hat"], "likelihoods": ret["likelihoods"]
        }, ret["x"])
        for k in lic_loss.keys():
            lic_loss[k].requires_grad_(True)

        psnr = helper.mse2psnr(helper.img2mse(ret["x"], x_hat, None))
        # ssim = ms_ssim(ret["x"], x_hat) # image size should be larger than 160 due to the 4 downsamplings in ms-ssim
        self.log("train/lic_psnr", psnr, on_step=True, prog_bar=True, logger=True)
        # self.log("train/lic_ms_ssim", ssim, on_step=True, prog_bar=True, logger=True)
        self.log("train/lic_loss", lic_loss, on_step=True)

        return lic_loss
        # return lic_loss * self.gamma + nerf_loss * (1 - self.gamma)

    def validation_step(self, batch, batch_idx):

        self.lic_model.eval()
        self.lic_model.update(force=True)
        self.nerf_model.eval()

        nerf_ret = self.forward_nerf(batch, False, self.white_bkgd, self.near, self.far)
        K, B, N, C = nerf_ret.shape

        x = batch["target"]
        xd = torch.stack(nerf_ret["density"], dim=0)
        rgb = torch.stack(nerf_ret["rgb"], dim=0)
        t_vals = torch.stack(nerf_ret["t_vals"], dim=0)

        nerf_x = torch.cat([rgb, xd], dim=-1)
        K, B, N, C = nerf_x.shape
        nerf_x = self.nerf2img_shape(nerf_x)

        if len(x.shape) == 2:
            H = W = int(B ** (1/2))
            C = x.shape[-1]
            x = x.permute(1, 0)
            x = x[None, :, :]
            x = x.reshape(-1, C, H, W)
            x = x.repeat(N, *([1] * (len(x.shape) - 1)))
        else:
            print(f":: ERROR :: len(x.shape) should be 2 - {x.shape}")
            exit()

        x_cat = torch.cat([x, nerf_x], dim=1)
        x_cat = x_cat.type(nerf_x.dtype).to(self.device)

        print(f":: DEBUG (val) :: x_cat.shape ({x_cat.shape})")

        out_enc = self.lic_model.compress(x_cat)
        out_dec = self.lic_model.decompress(out_enc["strings"], out_enc["shape"])

        # compression model output contains x_rec and xd_rec
        x_hat = out_dec["x_hat"]
        x_rec = x_hat[:, :3, ...]
        xd_rec = x_hat[:, 3:, ...]

        xd_rec = self.img2nerf_shape(xd_rec, K)
        t_vals = self.img2nerf_shape(t_vals, K)

        ret = []
        for i in range(self.nerf_model.num_levels):
            rgb = self.nerf_model.rgb_activation(rgb)
            rgb = rgb * (1 + 2 * self.nerf_model.rgb_padding) - self.nerf_model.rgb_padding
            comp_rgb, distance, acc, _ = helper.volumetric_rendering(
                rgb[i], xd_rec, t_vals[i], batch["rays_d"], white_bkgd=self.white_bkgd
            )
            ret.append((comp_rgb, distance, acc))

        rgb_fine = ret[1][0]

        return {
            "x_hat": x_rec,
            "strings": out_enc["strings"],
            "x": x,
            "rgb": rgb_fine,
        }

    def validation_epoch_end(self, outputs):
        # nerf
        val_image_sizes = self.trainer.datamodule.val_image_sizes
        xs = self.alter_gather_cat(outputs, "x", val_image_sizes)
        rgbs = self.alter_gather_cat(outputs, "rgb", val_image_sizes)
        psnr_mean = self.psnr_each(rgbs, xs).mean()
        ssim_mean = self.ssim_each(rgbs, xs).mean()
        lpips_mean = self.lpips_each(rgbs, xs).mean()
        self.log("val/nerf_psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/nerf_ssim", ssim_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/nerf_lpips", lpips_mean.item(), on_epoch=True, sync_dist=True)

        # compression
        H, W = xs.shape[-2:]
        x_hats = self.alter_gather_cat(outputs, "x_hat", val_image_sizes)
        strings = self.alter_gather_cat(outputs, "strings", val_image_sizes)
        bpp_loss = (sum(len(s[0]) for s in strings) * 8.0 / (H * W)).mean()
        mse_loss = helper.img2mse(x_hats, xs, None).mean()
        lic_loss = (self.lmbda * 255 ** 2 * mse_loss + bpp_loss).mean()
        psnr_mean = self.psnr_each(x_hats, xs).mean()
        # ms_ssim_mean = self.ms_ssim_each(x_hats, xs).mean()
        self.log("val/lic_psnr", psnr_mean.item(), on_epoch=True, sync_dist=True)
        # self.log("val/lic_ms_ssim", ms_ssim_mean.item(), on_epoch=True, sync_dist=True)
        self.log("val/lic_bpp", bpp_loss.item(), on_epoch=True, sync_dist=True)
        self.log("val/lic_loss", lic_loss.item(), on_epoch=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        all_image_sizes = (
            dmodule.all_image_sizes
            if not dmodule.eval_test_only
            else dmodule.test_image_sizes
        )
        rgbs = self.alter_gather_cat(outputs, "rgb", all_image_sizes)
        xs = self.alter_gather_cat(outputs, "x", all_image_sizes)
        psnr = self.psnr(rgbs, xs, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        ssim = self.ssim(rgbs, xs, dmodule.i_train, dmodule.i_val, dmodule.i_test)
        lpips = self.lpips(
            rgbs, xs, dmodule.i_train, dmodule.i_val, dmodule.i_test
        )

        self.log("test/nerf_psnr", psnr["test"], on_epoch=True)
        self.log("test/nerf_ssim", ssim["test"], on_epoch=True)
        self.log("test/nerf_lpips", lpips["test"], on_epoch=True)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(self.logdir, f"render_model_{random.randint(10, 99)}")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs)

            result_path = os.path.join(self.logdir, "results.json")
            self.write_stats(result_path, psnr, ssim, lpips)

        # compression
        H, W = xs.shape[-2:]
        x_hats = self.alter_gather_cat(outputs, "x_hat", all_image_sizes)
        strings = self.alter_gather_cat(outputs, "strings", all_image_sizes)
        bpp_loss = (sum(len(s[0]) for s in strings) * 8.0 / (H * W)).mean()
        mse_loss = helper.img2mse(x_hats, xs, None).mean()
        lic_loss = (self.lmbda * 255 ** 2 * mse_loss + bpp_loss).mean()
        psnr_mean = self.psnr_each(x_hats, xs).mean()
        # ms_ssim_mean = self.ms_ssim_each(x_hats, xs).mean()
        self.log("test/lic_psnr", psnr_mean.item(), on_epoch=True)
        # self.log("test/lic_ms_ssim", ms_ssim_mean.item(), on_epoch=True)
        self.log("test/lic_bpp", bpp_loss.item(), on_epoch=True)
        self.log("test/lic_loss", lic_loss.item(), on_epoch=True)

        return psnr, ssim, lpips

    def optimizer_step(self,
                       epoch: int,
                       batch_idx: int,
                       optimizers,
                       optimizer_idx: int = 0,
                       optimizer_closure: Optional[Callable[[], Any]] = None,
                       on_tpu: bool = False,
                       using_native_amp: bool = False,
                       using_lbfgs: bool = False,
                       ):

        step = self.trainer.global_step
        max_steps = gin.query_parameter("run.max_steps")

        if self.lr_delay_steps > 0:
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        t = np.clip(step / max_steps, 0, 1)
        scaled_lr = np.exp(np.log(self.lr_init) * (1 - t) + np.log(self.lr_final) * t)
        new_lr = delay_rate * scaled_lr

        for pg in optimizers.param_groups:
            pg["lr"] = new_lr

        optimizers.step(closure=optimizer_closure)

    def configure_optimizers(self):

        parameters = {
            n for n, p in self.lic_model.named_parameters()
            if not n.endswith(".quantiles") and p.requires_grad
        }
        aux_parameters = {
            n for n, p in self.lic_model.named_parameters()
            if n.endswith(".quantiles") and p.requires_grad
        }

        # Make sure we don't have an intersection of parameters
        params_dict = dict(self.lic_model.named_parameters())
        inter_params = parameters & aux_parameters
        union_params = parameters | aux_parameters

        assert len(inter_params) == 0
        assert len(union_params) - len(params_dict.keys()) == 0

        nerf_optimizer = optim.Adam(
            params=self.nerf_model.parameters(), lr=self.lr_init, betas=(0.9, 0.999)
        )
        lic_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(parameters)),
            lr=self.learning_rate,
        )
        lic_aux_optimizer = optim.Adam(
            (params_dict[n] for n in sorted(aux_parameters)),
            lr=self.aux_learning_rate,
        )

        if self.train_nerf:
            return nerf_optimizer, lic_optimizer, lic_aux_optimizer
        return lic_optimizer, lic_aux_optimizer

    def nerf2img_shape(self, x: torch.Tensor):
        K, B, N, C = x.shape
        H = W = int(B ** (1 / 2))
        x = x.permute(2, 0, 3, 1)
        return x.reshape(N, C * K, H, W)

    def img2nerf_shape(self, x, K):
        if len(x.shape) == 4:
            N, M, H, W = x.shape
            if M // K > 0:
                x = x.reshape(N, M, M // K, H * W)
                return x.permute(2, 3, 0, 1)
            else:
                x = x.reshape(N, M, H * W)
                return x.permute(2, 0, 1)
        return x

    def forward(self, x, rays, randomized, white_bkgd, near, far):
        # B, C, H, W = x.shape

        nerf_ret = self.forward_nerf(rays, randomized, white_bkgd, near, far) # density
        xd = torch.stack(nerf_ret["density"], dim=0)
        rgb = torch.stack(nerf_ret["rgb"], dim=0)
        t_vals = torch.stack(nerf_ret["t_vals"], dim=0)

        nerf_x = torch.cat([rgb, xd], dim=-1)
        # KxBxNxC -> Nx(C*K)xHxW (HxW = B)
        K, B, N, C = nerf_x.shape
        nerf_x = self.nerf2img_shape(nerf_x)

        H = W = int(B ** (1/2))
        C = x.shape[-1]
        x = x.permute(1, 0)
        x = x[None, :, :]
        x = x.reshape(-1, C, H, W)
        x = x.repeat(N, *([1] * (len(x.shape) - 1)))

        x_cat = torch.cat([x, nerf_x], dim=1)
        x_cat = x_cat.type(torch.FloatTensor).to(self.device)

        out = self.lic_model(x_cat)
        x_hat = out["x_hat"]
        likelihoods = out["likelihoods"]
        x_rec = x_hat[:, :3, ...]
        xd_rec = x_hat[:, 3:, ...]
        # x_rec, xd_rec = x_hat.chunk(2, dim=1)

        # Nx(C*K)xHxW (HxW = B) -> KxBxNxC
        xd_rec = self.img2nerf_shape(xd_rec, K)
        t_vals = self.img2nerf_shape(t_vals, K)

        ret = []
        for i in range(self.nerf_model.num_levels):
            rgb = self.nerf_model.rgb_activation(rgb)
            rgb = rgb * (1 + 2 * self.nerf_model.rgb_padding) - self.nerf_model.rgb_padding
            comp_rgb, distance, acc, _ = helper.volumetric_rendering(
                rgb[i], xd_rec, t_vals[i], rays["rays_d"], white_bkgd=white_bkgd
            )
            ret.append((comp_rgb, distance, acc))

        return {
            "x": x,
            "x_hat": x_rec,
            "likelihoods": likelihoods,
            "render": ret,
        }

    def forward_nerf(self, rays, randomized, white_bkgd, near, far):
        ret = {
            "density": [],
            "rgb": [],
            "t_vals": [],
        }
        for i_level in range(self.nerf_model.num_levels):
            kwargs = {
                "rays_o": rays["rays_o"],
                "rays_d": rays["rays_d"],
                "radii": rays["radii"],
                "randomized": randomized,
                "ray_shape": self.nerf_model.ray_shape,
            }
            if i_level == 0:
                kwargs.update({
                    "num_samples": self.nerf_model.num_samples,
                    "near": near,
                    "far": far,
                    "lindisp": self.nerf_model.lindisp,
                })
                t_vals, samples = helper.sample_along_rays(**kwargs)
            else:
                kwargs.update({
                    "t_vals": t_vals,
                    "weights": weights,
                    "stop_level_grad": self.nerf_model.stop_level_grad,
                    "resample_padding": self.nerf_model.resample_padding,
                })
                t_vals, samples = helper.resample_along_rays(**kwargs)

            samples_enc = helper.integrated_pos_enc(
                samples=samples, min_deg=self.nerf_model.min_deg_point, max_deg=self.nerf_model.max_deg_point
            )
            viewdirs_enc = helper.pos_enc(
                rays["viewdirs"], min_deg=0, max_deg=self.nerf_model.deg_view, append_identity=True
            )

            raw_rgb, raw_density = self.nerf_model.mlp(samples_enc, viewdirs_enc)
            if randomized and (self.nerf_model.density_noise > 0):
                raw_density += self.nerf_model.density_noise * torch.rand_like(raw_density)
            density = self.nerf_model.density_activation(raw_density + self.nerf_model.density_bias)

            ret["density"].append(density)
            ret["rgb"].append(raw_rgb)
            ret["t_vals"].append(t_vals)

            rgb = self.nerf_model.rgb_activation(raw_rgb)
            rgb = rgb * (1 + 2 * self.nerf_model.rgb_padding) - self.nerf_model.rgb_padding
            comp_rgb, distance, acc, weights = helper.volumetric_rendering(
                rgb, density, t_vals, rays["rays_d"], white_bkgd=white_bkgd
            )
            """
            ret.append((comp_rgb, distance, acc))
            """
        return ret

    def render_rays(self, batch, batch_idx):
        ret = {}
        rendered_results = self.model(
            batch, False, self.white_bkgd, self.near, self.far
        )
        rgb_fine = rendered_results[1][0]
        target = batch["target"]
        ret["target"] = target
        ret["rgb"] = rgb_fine
        return ret

    def load_state_dict(self, state_dict, strict=True):
        nerf_state_dict = dict()
        for k, v in state_dict.items():
            if 'model' in k:
                k = k.replace('model.', '')
                nerf_state_dict[k] = v
        #
        # for name, module in self.named_modules():
        return self.nerf_model.load_state_dict(nerf_state_dict, strict)



