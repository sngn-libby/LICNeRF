import argparse
import math
import random
import shutil
import sys
import time

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchsummary

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
from src.compressproj.datasets import ImageFolder
from src.compressproj.zoo import image_models


def main(argv):
    args = parse_args(argv)

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

    lic_model = image_models[args.model](quality=args.quality)
    lic_model = lic_model.to(device)

    torchsummary.summary(lic_model, (3, *args.patch_size), device=device)

    de_model = IDisc.build(args.de_config)

    if args.cuda and torch.cuda.device_count() > 1:
        lic_model = CustomDataParallel(lic_model)

    optimizer, aux_optimizer = configure_optimizers(lic_model, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210, 240], gamma=1 / 3)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        lic_model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print("last epoch:", last_epoch)
    test_model(args, kodak_dataloader, lic_model)

    best_loss = float("inf")
    # for epoch in range(last_epoch, last_epoch + 1):
    for epoch in range(last_epoch, args.epochs):
        # print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            lic_model,
            criterion,
            train_dataloader,
            kodak_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = validate(epoch, test_dataloader, lic_model, de_model, criterion)
        lr_scheduler.step(loss)
        #
        # is_best = loss < best_loss
        # best_loss = min(loss, best_loss)

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
                # is_best,
                False,
            )

    test_model(args, kodak_dataloader, lic_model)


def main(config: Dict[str, Any], args: argparse.Namespace):
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = IDisc.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()

    f16 = config["training"].get("f16", False)
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)

    save_dir = os.path.join(args.base_path, config["data"]["data_root"])
    assert hasattr(
        custom_dataset, config["data"]["train_dataset"]
    ), f"{config['data']['train_dataset']} not a custom dataset"
    valid_dataset = getattr(custom_dataset, config["data"]["val_dataset"])(
        test_mode=True, base_path=save_dir, crop=config["data"]["crop"]
    )
    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=4,
        sampler=valid_sampler,
        pin_memory=True,
        drop_last=False,
    )

    is_normals = config["model"]["output_dim"] > 1
    if is_normals:
        metrics_tracker = RunningMetric(list(DICT_METRICS_NORMALS.keys()))
    else:
        metrics_tracker = RunningMetric(list(DICT_METRICS_DEPTH.keys()))

    print("Start validation...")
    with torch.no_grad():
        validate.best_loss = np.inf
        validate(
            model,
            test_loader=valid_loader,
            config=config,
            metrics_tracker=metrics_tracker,
            context=context,
        )


def train_one_epoch(
        lic_model,
        de_model,
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

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        with torch.no_grad(): # with lic dataset --> no context
            depth, _, _ = de_model(d)

        print(f":: DEBUG :: depth.shape: {depth.shape}")
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
        de_model: nn.Module,
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
        for d in test_dataloader:
            d = d.to(device)
            depth, _, _ = de_model(d)
            d_cat = torch.concat([d, depth], dim=1)
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