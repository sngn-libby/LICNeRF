import os
import argparse
import math
import random
import json
import shutil
import sys
import time
from typing import *
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_lightning import LightningModule
from pytorch_msssim import ms_ssim

from src.compressproj.models.utils import (
    parse_args,
    save_checkpoint,
    CustomDataParallel,
    AverageMeter,
    RateDistortionLoss,
)
from src.compressproj.datasets import ImageFolder
from src.compressproj.zoo import image_models
from utils.train_utils import *

# nerfusion model
from src.nerfusion.opt import get_opts
from src.nerfusion.models.nerfusion import NeRFusion2
from src.nerfusion.models.rendering import render_depth_map, MAX_SAMPLES

# data
from torch.utils.data import Dataset, DataLoader
from src.data.ray_utils import batchified_get_rays
from torchvision import transforms
from src.data.data_util.nerf_360_v2 import *
from src.nerfusion.datasets import dataset_dict
from src.nerfusion.datasets.ray_utils import axisangle_to_R, get_rays
from src.data.sampler import DDPSequnetialSampler, MultipleImageDDPSampler


torch.set_printoptions(sci_mode=False)


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        print(f":: Log :: Seed is set to {args.seed}")
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader = set_nerf_dataloader(args, device)

    lic_model = image_models[args.model](quality=args.quality)
    lic_model = lic_model.to(device)

    # NeRFusion (pretrained)
    nerfusion = NeRFusion2(args.scale)
    nerf_checkpoint = torch.load(args.nerf_checkpoint)
    nerf_state_dict = nerf_checkpoint["state_dict"]
    nerfusion.load_state_dict(nerf_state_dict)

    if args.cuda and torch.cuda.device_count() > 1:
        lic_model = CustomDataParallel(lic_model)
        nerfusion = CustomDataParallel(nerfusion)

    optimizer, aux_optimizer = configure_optimizers(lic_model, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210, 240], gamma=1 / 3)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print(":: Model :: Loading ckpt", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        lic_model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print(":: Log :: last epoch:", last_epoch)
    # test_model(last_epoch, lic_model, de_model, test_dataloader, args.lmbda)

    best_loss = np.inf
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            lic_model,
            nerfusion,
            train_dataloader,
            test_dataloader,
            criterion,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = validate(epoch, lic_model, nerfusion, test_dataloader, criterion)
        lr_scheduler.step(loss)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                args,
                {
                    "epoch": epoch,
                    "state_dict": lic_model.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                False,
            )

    test_model(last_epoch, lic_model, nerfusion, test_dataloader, args.lmbda)



def train_one_epoch(
        lic_model,
        nerf,
        train_dataloader,
        test_dataloader,
        criterion,
        optimizer,
        aux_optimizer,
        epoch,
        clip_max_norm,
):
    lic_model.train()
    lic_model.update(force=True)
    device = next(lic_model.parameters()).device

    for i, batch in enumerate(train_dataloader):
        d = batch["image"]
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        with torch.no_grad(): # with lic dataset --> no context
            rays_o, rays_d = batch["rays_o"], batch["rays_d"]
            kwargs = {"test_time": True, "random_bg": True}
            depth = render_depth_map(nerf, rays_o, rays_d, **kwargs)  # 중간 결과로 받아오기
            print(f":: DEBUG :: depth.shape {depth.shape}")

        d_cat = torch.concat([d, depth], dim=1)

        out = lic_model(d_cat)
        out_criterion = criterion(out, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(lic_model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = lic_model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 50 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def validate(
        epoch,
        lic_model: nn.Module,
        nerf: nn.Module,
        test_dataloader,
        criterion,
):
    lic_model.eval()
    device = next(lic_model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()


    with torch.no_grad():
        for batch in test_dataloader:
            d = batch["rgb"]
            d = d.to(device)

            poses = batch["pose"]
            directions = batch["direction"]
            rays_o, rays_d = get_rays(directions, poses)
            kwargs = {"test_time": True, "random_bg": True}
            nerf_res = render(nerf, rays_o, rays_d, **kwargs) # 중간 결과로 받아오기

            d_cat = torch.concat([d, nerf_res['rgb']], dim=1)
            out = lic_model(d_cat)
            out_criterion = criterion(out, d)

            aux_loss.update(lic_model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.3f} |"
        f"\tBpp loss: {bpp_loss.avg:.2f} |"
        f"\tAux loss: {aux_loss.avg:.2f}\n"
    )

    return loss.avg


def test_model(
        epoch,
        lic_model: nn.Module,
        nerf: nn.Module,
        test_dataloader,
        lmbda
):
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    avg_psnr = AverageMeter()
    avg_msssim = AverageMeter()

    lic_model.eval()
    device = next(lic_model.parameters()).device

    with torch.no_grad():
        for batch in test_dataloader:
            d = batch["rgb"]
            d = d.to(device)

            poses = batch["pose"]
            directions = batch["direction"]
            rays_o, rays_d = get_rays(directions, poses)
            kwargs = {"test_time": True, "random_bg": True}
            nerf_res = render(nerf, rays_o, rays_d, **kwargs) # 중간 결과로 받아오기

            d_cat = torch.concat([d, nerf_res['rgb']], dim=1)

            out_enc = lic_model.compress(d_cat)
            out_dec = lic_model.decompress(strings=out_enc["strings"], shape=out_enc["shape"])

            bpp = update_meter(bpp_loss, d, out_enc, out_dec, "bpp")
            mse = update_meter(mse_loss, d, out_enc, out_dec, "mse")
            update_meter(avg_psnr, d, out_enc, out_dec, "psnr")
            update_meter(avg_msssim, d, out_enc, out_dec, "ms_ssim")
            loss.update(bpp + lmbda * 255 ** 2 * mse)

    print(
        f"Test Average losses (epoch {epoch}):"
        f"\tLoss: {loss.avg:.6f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tBpp loss: {bpp_loss.avg:.6f} |"
        f"\tPSNR: {avg_psnr.avg:.2f} |"
        f"\tMS-SSIM: {avg_msssim.avg:.6f}\n"
    )

    return loss.avg


def load_nerf360v2_dataset(args, transform):
    return NeRF360v2_Dataset(**load_nerf360v2_data(args), transform=transform)


def load_nerf360v2_data(args):
    datadir = args.dataset
    scene_name = args.scene
    factor = 4
    cam_scale_factor = 0.95
    train_skip = 1
    val_skip = 1
    test_skip = 1
    near = None
    far = None
    strict_scaling = False

    # res = (
    (
        images,
        intrinsics,
        extrinsics,
        image_sizes,
        near,
        far,
        ndc_coeffs,
        i_split, #(i_train, i_val, i_test, i_all),
        render_poses
    ) = load_nerf_360_v2_data(
        datadir=datadir,
        scene_name=scene_name,
        factor=factor,
        cam_scale_factor=cam_scale_factor,
        train_skip=train_skip,
        val_skip=val_skip,
        test_skip=test_skip,
        near=near,
        far=far,
        strict_scaling=strict_scaling,
    )

    return {
        "images":images,
        "intrinsics":intrinsics,
        "extrinsics":extrinsics,
        "image_sizes":image_sizes,
        "near":near,
        "far":far,
        "ndc_coeffs":ndc_coeffs,
        "i_split":i_split,
        "render_poses":render_poses,
    }
    #return res


class NeRF360v2_Dataset(Dataset):
    def __init__(
            self,
            images,
            intrinsics,
            extrinsics,
            image_sizes,
            near,
            far,
            ndc_coeffs,
            i_split,
            render_poses,
            use_pixel_centers=True,
            load_radii=False,
            ndc_coord=False,
            multlosses=None,
            num_devices=torch.cuda.device_count() * 4,
            transform=None,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        normals = None
        ray_info = self.parse_info(images, normals, render_poses, i_split[0])
        for name, value in ray_info.items():
            setattr(self, name, value)

    def len(self):
        return len(self.rays_d)

    def __getitem__(self, idx):
        ret = {}

        ret["rays_o"] = self.rays_o[idx]
        ret["rays_d"] = self.rays_d[idx]
        ret["viewdirs"] = self.viewdirs[idx]
        ret["image"] = np.zeros_like(ret["rays_o"])
        ret["radii"] = np.zeros((ret["rays_o"].shape[0], 1))
        ret["multloss"] = np.zeros((ret["rays_o"].shape[0], 1))
        ret["normals"] = np.zeros_like(ret["rays_o"])

        if self.images is not None:
            image = torch.from_numpy(self.images[idx])
            if self.transform is not None:
                image = self.transform(image)
            ret["image"] = image

        if self.radii is not None:
            ret["radii"] = self.radii[idx]

        if self.multloss is not None:
            ret["multloss"] = self.multloss[idx]

        if self.normals is not None:
            ret["normals"] = torch.from_numpy(self.normals[idx])

        return ret


    def parse_info(
            self,
            _images,
            _normals,
            render_poses,
            idx,
            dummy=True,
    ):
        images = None
        normals = None
        radii = None
        multloss = None

        if _images is not None:
            extrinsics_idx = self.extrinsics[idx]
            intrinsics_idx = self.intrinsics[idx]
            image_sizes_idx = self.image_sizes[idx]
        else:
            extrinsics_idx = render_poses
            N_render = len(render_poses)
            intrinsics_idx = np.stack([self.intrinsics[0] for _ in range(N_render)])
            image_sizes_idx = np.stack([self.image_sizes[0] for _ in range(N_render)])

        _rays_o, _rays_d, _viewdirs, _radii, _multloss = batchified_get_rays(
            intrinsics_idx,
            extrinsics_idx,
            image_sizes_idx,
            self.use_pixel_centers,
            self.load_radii,
            self.ndc_coord,
            self.ndc_coeffs,
            self.multlosses[idx] if self.multlosses is not None else None,
        )

        device_count = self.num_devices
        n_dset = len(_rays_o)
        dummy_num = (
            (device_count - n_dset % device_count) % device_count if dummy else 0
        )

        rays_o = np.zeros((n_dset + dummy_num, 3), dtype=np.float32)
        rays_d = np.zeros((n_dset + dummy_num, 3), dtype=np.float32)
        viewdirs = np.zeros((n_dset + dummy_num, 3), dtype=np.float32)

        rays_o[:n_dset], rays_o[n_dset:] = _rays_o, _rays_o[:dummy_num]
        rays_d[:n_dset], rays_d[n_dset:] = _rays_d, _rays_d[:dummy_num]
        viewdirs[:n_dset], viewdirs[n_dset:] = _viewdirs, _viewdirs[:dummy_num]

        viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=1, keepdims=True)

        if _images is not None:
            images_idx = np.concatenate([_images[i].reshape(-1, 3) for i in idx])
            images = np.zeros((n_dset + dummy_num, 3))
            images[:n_dset] = images_idx
            images[n_dset:] = images[:dummy_num]

        if _normals is not None:
            normals_idx = np.concatenate([_normals[i].reshape(-1, 4) for i in idx])
            normals = np.zeros((n_dset + dummy_num, 4))
            normals[:n_dset] = normals_idx
            normals[n_dset:] = normals[:dummy_num]

        if _radii is not None:
            radii = np.zeros((n_dset + dummy_num, 1), dtype=np.float32)
            radii[:n_dset], radii[n_dset:] = _radii, _radii[:dummy_num]

        if _multloss is not None:
            multloss = np.zeros((n_dset + dummy_num, 1), dtype=np.float32)
            multloss[:n_dset], multloss[n_dset:] = _multloss, _multloss[:dummy_num]

        rays_info = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "viewdirs": viewdirs,
            "images": images,
            "radii": radii,
            "multloss": multloss,
            "normals": normals,
        }
        return rays_info


def set_nerf_dataloader(args, device):

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = load_nerf360v2_dataset(args, transform=train_transforms)
    test_dataset = load_nerf360v2_dataset(args, transform=test_transforms)

    train_sampler = MultipleImageDDPSampler(
        batch_size=args.batch_size,
        num_replicas=None,
        rank=None,
        total_len=len(train_dataset),
        epoch_size=args.epochs,
        tpu=False,
    )
    test_sampler = DDPSequnetialSampler(
        batch_size=args.batch_size,
        num_replicas=None,
        rank=None,
        N_total=len(test_dataset),
        tpu=False,
    )

    train_dataloader = DataLoader(train_dataset,
                                  batch_sampler=train_sampler,
                                  num_workers=torch.cuda.device_count() * 4,
                                  persistent_workers=True,
                                  batch_size=None,
                                  pin_memory=True if device == "cuda" else False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_sampler=test_sampler,
                                 num_workers=torch.cuda.device_count() * 4,
                                 batch_size=None,
                                 pin_memory=True if device == "cuda" else False)

    return train_dataloader, test_dataloader


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=list(image_models.keys()).append(['checker2021', 'mbt-de']),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset",
        #choices=["nerf", "nsvf", "colmap", "nerfpp", "scannet", "google_scanned"]
    )
    parser.add_argument(
        "--test-dataset", type=str, help="Test dataset",
        #choices=["nerf", "nsvf", "colmap", "nerfpp", "scannet", "google_scanned"]
    )
    parser.add_argument(
        "--scene", type=str, required=True,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=200,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4 * torch.cuda.device_count(),
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=random.randrange(100, 1000), help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")

    # NeRFusion
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--nerf_checkpoint", type=str, help="Path to a depth estimation (iDisc) checkpoint")
    parser.add_argument("--downsample", type=float, default=1.0, help="scene scale")
    parser.add_argument("--split", required=True, type=str)
    parser.add_argument("--ray_sampling_strategy", type=str,
                        default="all_images",
                        choices=["all_images", "same_images"])

    args = parser.parse_args(argv)
    return args



if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)
