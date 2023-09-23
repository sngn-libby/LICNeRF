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

from pytorch_msssim import ms_ssim

from src.compressproj.models.utils import (
    parse_args,
    save_checkpoint,
    CustomDataParallel,
    AverageMeter,
    RateDistortionLoss,
)
from src.compressproj.datasets import ImageFolderDE as ImageFolder
from src.compressproj.zoo import image_models
# from src.idisc.models.idisc import IDisc
from src.newcrfs.NewCRFDepth import NewCRFDepth
from src.newcrfs.utils import *

torch.set_printoptions(sci_mode=False)


def main(argv):
    args = parse_args(argv)

    args.seed = 777 # for synchronize randomcropped depth map
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
    print(f":: Log :: {device.upper()}, #{torch.cuda.device_count()}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
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

    for i, d in enumerate(train_dataloader):
        d = d.to(device)
    for i, d in enumerate(test_dataloader):
        d = d.to(device)
    for i, d in enumerate(kodak_dataloader):
        d = d.to(device)


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)

