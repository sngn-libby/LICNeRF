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

from torchvision import transforms

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
from src.nerfusion.models.rendering import render, MAX_SAMPLES

# data
from torch.utils.data import DataLoader
from src.nerfusion.datasets import dataset_dict
from src.nerfusion.datasets.ray_utils import axisangle_to_R, get_rays

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
        d = batch["rgb"]
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        with torch.no_grad(): # with lic dataset --> no context
            poses = batch["pose"]
            directions = batch["direction"]
            rays_o, rays_d = get_rays(directions, poses)
            kwargs = {"test_time": True, "random_bg": True}
            nerf_res = render(nerf, rays_o, rays_d, **kwargs)  # 중간 결과로 받아오기

        d_cat = torch.concat([d, nerf_res['rgb']], dim=1)

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


def set_nerf_dataloader(args, device):
    dataset = dataset_dict[args.dataset]
    kwargs = {'root_dir': os.path.join(os.path.expanduser("~"), "datasets"),
              'downsample': args.downsample}
    train_dataset = dataset(split=args.split, **kwargs)
    train_dataset.batch_size = args.batch_size
    train_dataset.ray_sampling_strategy = args.ray_sampling_strategy
    test_dataset = dataset(split='test', **kwargs)

    train_dataloader = DataLoader(train_dataset,
                                  num_workers=torch.cuda.device_count() * 4,
                                  persistent_workers=True,
                                  batch_size=None,
                                  pin_memory=True if device == "cuda" else False)
    test_dataloader = DataLoader(test_dataset,
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
        choices=["nerf", "nsvf", "colmap", "nerfpp", "scannet", "google_scanned"]
    )
    parser.add_argument(
        "--test-dataset", type=str, help="Test dataset",
        choices=["nerf", "nsvf", "colmap", "nerfpp", "scannet", "google_scanned"]
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
