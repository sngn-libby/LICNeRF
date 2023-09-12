import os
import random

from src.compressproj.models.utils import (
    parse_args,
    save_checkpoint,
    CustomDataParallel,
    AverageMeter,
    RateDistortionLoss,
)
from src.compressproj.zoo import image_models
from src.newcrfs.NewCRFDepth import NewCRFDepth
from src.newcrfs.utils import *
from utils.train_utils import *

torch.set_printoptions(sci_mode=False)


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        print(f":: Log :: Seed is set to {args.seed}")
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader, test_dataloader, kodak_dataloader = set_dataloader(args, device)

    lic_model = image_models[args.model](quality=args.quality)
    lic_model = lic_model.to(device)

    # Depth Estimation Network
    print(f":: Log :: DE Config file at {args.de_config}")
    de_model = NewCRFDepth(version="large07", inv_depth=False, max_depth=80)
    if args.de_ckpt:
        print(f":: Model :: DE Loading ckpt {args.de_ckpt}")
        de_checkpoints = torch.load(args.de_ckpt)
        de_model.parse_and_load_state_dict(de_checkpoints["model"])
    de_model = de_model.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        lic_model = CustomDataParallel(lic_model)
        de_model = CustomDataParallel(de_model)

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
            de_model,
            train_dataloader,
            test_dataloader,
            criterion,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = validate(epoch, lic_model, de_model, test_dataloader, criterion)
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

    test_model(last_epoch, lic_model, de_model, kodak_dataloader, args.lmbda)


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
            # depth, _, _ = de_model(d)
            depth_est = de_model(d)

            d_flip = flip_lr(d)
            depth_est_flip = de_model(d_flip)
            depth = post_process_depth(depth_est, depth_est_flip)

        d_cat = torch.concat([d, depth], dim=1) # n_channel - 1

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
            # depth, _, _ = de_model(d)
            depth_est = de_model(d)
            d_flip = flip_lr(d)
            depth_est_flip = de_model(d_flip)
            depth = post_process_depth(depth_est, depth_est_flip)

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


def test_model(
        epoch,
        lic_model: nn.Module,
        de_model: nn.Module,
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
        for d in test_dataloader:
            d = d.to(device)
            depth, _, _ = de_model(d)
            d_cat = torch.concat([d, depth], dim=1)
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


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(argv)

