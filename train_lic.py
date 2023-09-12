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
from src.compressproj.models.sensetime import JointCheckerboardHierarchicalPriors

torch.set_printoptions(sci_mode=False)

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


def train_one_epoch(
    model, criterion, train_dataloader, test_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    model.update(force=True)
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device) # [ 16, 3, 256, 256 ]

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d) # [ 16, 3, 256, 256 ]

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )
            # test_model(None,test_dataloader=test_dataloader, model=model)
            # test_epoch(epoch, test_dataloader, model, criterion)
        # if i in [3000]:
        # # if i in [300, 500, 1000, 1500, 3000, 10000, 15000, 20000]:
        #     print(f":: Log :: Save {i}-th values")
        #     with open("D:/Research_LUT/z_hat_list.txt", 'w') as f:
        #         for i, z_hat_list in enumerate(out_net["z_hat_list"]):
        #             f.write("channel " + str(i) + "\n")
        #             for j, val in enumerate(z_hat_list):
        #                 if val == 0:
        #                     continue
        #                 f.write(str(j) + ":" + str(int(val)) + "\n")
        #     with open("D:/Research_LUT/mean_list.txt", 'w') as f:
        #         for i, mu_list in enumerate(out_net["mu_list"]):
        #             if mu_list is None:
        #                 continue
        #             f.write("channel " + str(i) + "\n")
        #             for key, val in mu_list.items():
        #                 f.write(str(key) + ":" + str(int(val)) + "\n")
        #     with open("D:/Research_LUT/scale_list.txt", 'w') as f:
        #         for i, sigma_list in enumerate(out_net["sigma_list"]):
        #             if sigma_list is None:
        #                 continue
        #             f.write("channel " + str(i) + "\n")
        #             for key, val in sigma_list.items():
        #                 f.write(str(key) + ":" + str(int(val)) + "\n")
        # if i == 300:
        #     break



def test_model(args, test_dataloader, model):
    model.eval()
    model.update(force=True)
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    avg_enc_proc_time = AverageMeter()
    avg_dec_proc_time = AverageMeter()
    avg_enc_time = AverageMeter()
    avg_dec_time = AverageMeter()
    avg_psnr = AverageMeter()
    avg_msssim = AverageMeter()

    def mse(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.mse_loss(a, b).item()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            H, W = d.shape[-2:]

            # Encoding
            # start = time.time()
            out_enc = model.compress(d)
            # enc_time = time.time() - start

            # start = time.time()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            # dec_time = time.time() - start

            num_pixels = H * W
            out = dict()
            out["bpp_loss"] = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            out["mse_loss"] = mse(d, out_dec["x_hat"])
            out["loss"] = (args.lmbda if args is not None else 0.0130) * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]
            out["psnr"] = psnr(d, out_dec["x_hat"])
            out["ms_ssim"] = ms_ssim(d, out_dec["x_hat"], data_range=1.0).item()

            bpp_loss.update(out["bpp_loss"])
            loss.update(out["loss"])
            mse_loss.update(out["mse_loss"])
            avg_psnr.update(out["psnr"])
            avg_msssim.update(out["ms_ssim"])

            avg_enc_time.update(out_enc["cost_time"])
            avg_dec_time.update(out_dec["cost_time"])
            avg_enc_proc_time.update(out_enc["tar_cost_time"])
            avg_dec_proc_time.update(out_dec["tar_cost_time"])

    print(
        f"Test Average losses:"
        f"\tLoss: {loss.avg:.6f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tBpp loss: {bpp_loss.avg:.6f} |"
        f"\tPSNR: {avg_psnr.avg:.2f} |"
        f"\tMS-SSIM: {avg_msssim.avg:.6f} |"
        f"\tEnc Time: {avg_enc_time.avg:.6f} |"
        f"\tDec Time: {avg_dec_time.avg:.6f} |"
        f"\tTar Enc Time: {avg_enc_proc_time.avg:.6f} |"
        f"\tTar Dec Time: {avg_dec_proc_time.avg:.6f}\n"
    )

    return loss.avg

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
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


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        print('Seed:', args.seed)
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

    net = image_models[args.model](quality=args.quality)
    net = net.to(device)

    torchsummary.summary(net, (3, *args.patch_size), device=device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 180, 210, 240], gamma=1 / 3)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print("last epoch:", last_epoch)
    test_model(args, kodak_dataloader, net)

    best_loss = float("inf")
    # for epoch in range(last_epoch, last_epoch + 1):
    for epoch in range(last_epoch, args.epochs):
        # print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            kodak_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)
        #
        # is_best = loss < best_loss
        # best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                args,
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                # is_best,
                False,
            )

    test_model(args, kodak_dataloader, net)


if __name__ == "__main__":
    print(sys.argv[1:], '\n')
    main(sys.argv[1:])