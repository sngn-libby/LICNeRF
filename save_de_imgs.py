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

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

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
from src.newcrfs.NewCRFDepth import NewCRFDepth
from src.newcrfs.utils import *

torch.set_printoptions(sci_mode=False)


def main(argv):
    args = parse_args(argv)

    args.seed = 777
    if args.seed is not None:
        print(f":: Log :: Seed is set to {args.seed}")
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
    kodak_dataset = ImageFolder(args.test_dataset, split="kodak", transform=test_transforms)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    kodak_dataloader = DataLoader(
        kodak_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    # Depth Estimation Network
    de_model = NewCRFDepth(version="large07", inv_depth=False, max_depth=80)
    if args.de_ckpt:
        print(f":: Model :: DE Loading ckpt {args.de_ckpt}")
        de_checkpoints = torch.load(args.de_ckpt)
        de_model.parse_and_load_state_dict(de_checkpoints["model"])
    de_model = de_model.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        de_model = CustomDataParallel(de_model)

    train_one_epoch(
        de_model,
        train_dataloader,
        test_dataloader,
        args.clip_max_norm,
        os.path.join(args.dataset, "train"),
    )
    print(f":: FINISH ::")
    exit()
    test_model(de_model, kodak_dataloader, os.path.join(args.test_dataset, "kodak"))
    validate(de_model, test_dataloader, os.path.join(args.dataset, "test"))


def train_one_epoch(
        de_model,
        train_dataloader,
        test_dataloader,
        clip_max_norm,
        dataset_path,
):
    device = next(de_model.parameters()).device
    files = list(map(str, train_dataloader.dataset.samples))
    tar_dataset_path = dataset_path
    tar_dataset_path = tar_dataset_path.replace("train", "train_depth")
    print(f"\n\n:: SESSION :: [Train] {tar_dataset_path}\n\n\n")
    if not os.path.exists(tar_dataset_path):
        os.makedirs(tar_dataset_path)

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
        size = len(d)

        with torch.no_grad(): # with lic dataset --> no context
            depth_est = de_model(d)
            d_flip = flip_lr(d)
            depth_est_flip = de_model(d_flip)
            depth = post_process_depth(depth_est, depth_est_flip)

        for j in range(size):
            file_path = files[i * size + j].replace("train", "train_depth")
            file_path = file_path.replace("png", "npy").replace("jpg", "npy")
            print(f"Save {files[i * size + j]} -> {file_path}")
            depth_np = depth[j].cpu().numpy()
            np.save(file_path, depth_np)
	
    print(":: Log :: Finished to save train_dataloader")


def validate(
        de_model: nn.Module,
        test_dataloader,
        dataset_path,
):
    device = next(de_model.parameters()).device

    files = list(map(str, test_dataloader.dataset.samples))
    tar_dataset_path = dataset_path
    key = "test" if "test" in dataset_path else "kodak"
    tar_dataset_path = tar_dataset_path.replace(key, key + "_depth")
    print(f"\n\n:: SESSION :: [Validate] {tar_dataset_path}\n\n\n")
    if not os.path.exists(tar_dataset_path):
        os.makedirs(tar_dataset_path)

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            size = len(d)
            d = d.to(device)
            depth_est = de_model(d)
            d_flip = flip_lr(d)
            depth_est_flip = de_model(d_flip)
            depth = post_process_depth(depth_est, depth_est_flip)

            for j in range(size):
                file_path = files[i * size + j].replace("test", "test_depth").replace("kodak", "kodak_depth")
                file_path = file_path.replace("png", "npy").replace("jpg", "npy")
                print(f"Save {files[i * size + j]} -> {file_path}")
                depth_np = depth[j].cpu().numpy()
                np.save(file_path, depth_np)
                #file_path = os.path.join(dataset_path, "test_depth", files[i * size + j])
                #save_image(depth[j], file_path)


def test_model(
        de_model: nn.Module,
        test_dataloader,
        dataset_path,
):
    device = next(de_model.parameters()).device
    files = list(map(str, test_dataloader.dataset.samples))
    #files, _ = map(list, zip(*test_dataloader.dataset.samples))
    tar_dataset_path = dataset_path
    key = "test" if "test" in dataset_path else "kodak"
    tar_dataset_path = tar_dataset_path.replace(key, key + "_depth")
    print(f"\n\n:: SESSION :: [Test] {tar_dataset_path} ({os.path.exists(tar_dataset_path)})\n\n\n")
    if not os.path.exists(tar_dataset_path):
        print(f":: Log :: {tar_dataset_path} is created")
        os.makedirs(tar_dataset_path)

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            d = d.to(device)
            size = len(d)

            depth_est = de_model(d)
            d_flip = flip_lr(d)
            depth_est_flip = de_model(d_flip)
            depth = post_process_depth(depth_est, depth_est_flip)

            for j in range(size):
                file_path = files[i * size + j].replace("test", "test_depth").replace("kodak", "kodak_depth")
                file_path = file_path.replace("png", "npy").replace("jpg", "npy")
                print(f"Save {files[i * size + j]} -> {file_path}")
                depth_np = depth[j].cpu().numpy()
                np.save(file_path, depth_np)
                #save_image(depth[j], file_path)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return -10 * math.log10(F.mse_loss(a, b))


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)

