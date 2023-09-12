import math
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms
from src.compressproj.datasets import ImageFolder

from pytorch_msssim import ms_ssim

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


def update_meter(meter, d, out_enc, out_dec, key):
    def mse(a: torch.Tensor, b: torch.Tensor) -> float:
        return F.mse_loss(a, b).item()

    if "bpp" in key:
        num_pixels = d.shape[-2] * d.shape[-1]
        val = sum(len(s[0]) for s in out_enc["strings"]) * 8. / num_pixels
    elif "mse" in key:
        val = mse(d, out_dec["x_hat"])
    elif "pnsr" == key:
        val = psnr(d, out_dec["x_hat"])
    elif "ms_ssim" == key:
        val = ms_ssim(d, out_dec["x_hat"], data_range=1.).item()
    else:
        return None

    meter.update(val)
    return val

def set_dataloader(args, device="cuda"):
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )
    test_transforms = transforms.Compose(
        [transforms.CenterCrop(args.patch_size), transforms.ToTensor()]
    )

    train_dataset = ImageFolder(args.dataset, split="train", transform=train_transforms)
    test_dataset = ImageFolder(args.dataset, split="test", transform=test_transforms)
    kodak_dataset = ImageFolder(args.test_dataset, split="kodak", transform=test_transforms)

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

    return train_dataloader, test_dataloader, kodak_dataloader

