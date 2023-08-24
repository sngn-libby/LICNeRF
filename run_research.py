# ------------------------------------------------------------------------------------
# LIC-NeRF
# Copyright (c) 2023 MCML, Yonsei Univ. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import argparse
import logging
import os
import random
import shutil
from typing import *
from datetime import datetime

from pytorch_lightning.plugins import DDPPlugin

import gin
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)

from utils.select_option import select_callback, select_dataset, select_research_model


def print_session_log(log_str: str, f="=", n=150):
    log_str = " " + log_str + " "
    print()
    print("{s:{f}^{n}}".format(s=log_str, f=f, n=n))
    print()


def set_logger(logbase, exp_name):
    logging.getLogger("lightning").setLevel(logging.ERROR)
    if logbase is None:
        logbase = "logs"
    os.makedirs(logbase, exist_ok=True)
    logdir = os.path.join(logbase, exp_name)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, exp_name), exist_ok=True)
    logger = pl_loggers.TensorBoardLogger(
        save_dir=logdir,
        name=exp_name,
    )
    return logger, logdir


@gin.configurable()
def run(
        ginc: str,
        ginb,
        scene_name: Optional[str],
        ckpt_path: Optional[str] = None,
        datadir: Optional[str] = None,
        logbase: Optional[str] = None,
        research_model_name: Optional[str] = None,
        lic_model_name: Optional[str] = None,
        nerf_model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        postfix: Optional[str] = None,
        entity: Optional[str] = None,
        # Optimization
        max_steps: int = -1,
        max_epochs: int = -1,
        precision: int = 32,
        # Logging
        log_every_n_steps: int = 1000,
        progressbar_refresh_rate: int = 5,
        # Run Mode
        run_train: bool = True,
        run_eval: bool = True,
        run_render: bool = False,
        num_devices: Optional[int] = None,
        num_sanity_val_steps: int = 0,
        seed: int = 777,
        debug: bool = False,
        save_last: bool = True,
        grad_max_norm=0.0,
        grad_clip_algorithm="norm",
        add_noise: float=0.0,

        # train configs
        lmbda=0.0130,
        learning_rate=1e-4,
        aux_learning_rate=1e-3,
        gamma=0.8,

        lr_init: float = 5.0e-4,
        lr_final: float = 5.0e-6,
        lr_delay_steps: int = 2500,
        lr_delay_mult: float = 0.01,
        coarse_loss_mult: float = 0.1,
        randomized: bool = True,
        use_multiscale: bool = False,

        train_nerf: bool = True,

        # LIC configs
        N=192,
        M=192,

        # nerf configs
        num_samples: int = 128,
        num_levels: int = 2,
        resample_padding: float = 0.01,
        stop_level_grad: bool = True,
        use_viewdirs: bool = True,
        lindisp: bool = False,
        ray_shape: str = "cone",
        min_deg_point: int = 0,
        max_deg_point: int = 16,
        deg_view: int = 4,
        density_noise: float = 0,
        density_bias: float = -1,
        rgb_padding: float = 0.001,
        disable_integration: bool = False,
):
    print(f":: Log :: run parameters")
    for n, v in vars().items():
        if isinstance(v, str) and '~' in v:
            setattr(vars(), n, v)

    device = "gpu" if torch.cuda.is_available() else "cpu"

    date = datetime.now()
    date = date.strftime("%Y%m%d")
    datadir = datadir.rstrip("/")
    homedir = os.path.expanduser('~')
    datadir = datadir.replace('~', homedir)
    logbase = logbase.replace('~', homedir)

    exp_name = (
            lic_model_name + "_" + nerf_model_name + "_" + dataset_name + "_" + scene_name + "_" + str(seed).zfill(3)
    )
    print(f":: Log :: [ Model ] {exp_name}")
    if postfix is not None:
        exp_name += "_" + postfix
    if debug:
        exp_name += "_debug"
    if num_devices is None:
        num_devices = torch.cuda.device_count()
    if nerf_model_name in ["plenoxel"]:
        num_devices = 1

    logger, logdir = set_logger(logbase, exp_name)

    txt_path = os.path.join(logdir, "config.gin")
    with open(txt_path, "w") as fp_txt:
        for config_path in ginc:
            fp_txt.write(f"Config from {config_path}\n\n")
            with open(config_path, "r") as fp_config:
                readlines = fp_config.readlines()
            for line in readlines:
                fp_txt.write(line)
            fp_txt.write("\n")

        fp_txt.write("\n### Binded options\n")
        for line in ginb:
            fp_txt.write(line + "\n")

    seed_everything(seed, workers=True)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        monitor="val/lic_loss",
        dirpath=logdir,
        filename="best",
        save_top_k=1,
        mode="max",
        save_last=save_last,
    )
    tqdm_progrss = TQDMProgressBar(refresh_rate=progressbar_refresh_rate)

    callbacks = []
    if not nerf_model_name in ["plenoxel"]:
        callbacks.append(lr_monitor)
    callbacks += [model_checkpoint, tqdm_progrss]
    callbacks += select_callback(nerf_model_name)

    # multi gpu
    ddp_plugin = DDPPlugin(find_unused_parameters=False) if num_devices > 1 else None

    trainer = Trainer(
        logger=logger if run_train else None,
        log_every_n_steps=log_every_n_steps,
        devices=num_devices,
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator=device,
        replace_sampler_ddp=False,
        strategy=ddp_plugin,
        check_val_every_n_epoch=1,
        precision=precision,
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks=callbacks,
        gradient_clip_algorithm=grad_clip_algorithm,
        gradient_clip_val=grad_max_norm,
    )

    if ckpt_path is not None:
        if ".ckpt" not in ckpt_path:
            ckpt_path = f"{ckpt_path}/last.ckpt"
    print(f":: Log :: Provided checkpoints: {ckpt_path}")

    data_module = select_dataset(
        dataset_name = dataset_name,
        scene_name = scene_name,
        datadir = datadir,
    )


    model = select_research_model(
        research_model_name=research_model_name,
        lic_model_name=lic_model_name,
        nerf_model_name=nerf_model_name,
        train_kwargs=train_kwargs,
        lic_kwargs=lic_kwargs,
        nerf_kwargs=nerf_kwargs,
    )
    model.logdir = logdir

    print_session_log("Train session starts")
    best_ckpt = os.path.join(logdir, f"{date}_best.ckpt")
    version0 = os.path.join(logdir, exp_name, "version_0")
    if os.path.exists(version0):
        shutil.rmtree(version0, True)

    trainer.fit(model, data_module, ckpt_path=ckpt_path)

    # eval
    print_session_log("Eval session starts")
    trainer.test(model, data_module, ckpt_path=best_ckpt)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc", action="append", help="gin configuration file",
    )
    parser.add_argument(
        "--ginb", action="append", help="gin bindings",
    )
    parser.add_argument(
        "--add_noise", type=float, default=0, help="proportion of gaussian noise"
    )
    parser.add_argument(
        "--scene_name", type=str, default=None, help="scene name to render"
    )
    parser.add_argument(
        "--seed", type=int, default=random.randint(10, 99)
    )

    args = parser.parse_args()
    for ginc_path in args.ginc:
        config_path_expand_user(ginc_path)

    ginbs =[]
    if args.ginb:
        ginbs.extend(args.ginb)

    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {args.ginb}")

    gincs = gin.parse_config_files_and_bindings(args.ginc, ginbs)

    print(f"\n:: Log :: Provided Arguments '{args}' (gincs: {gincs}, ginbs:{ginbs})")

    return args, ginbs


def config_path_expand_user(ginc):
    tmppath = os.path.abspath('~/tmp.txt')
    homedir = os.path.expanduser('~')
    with open(tmppath, 'w+') as fw:
        with open(ginc, 'r') as fr:
            lines = fr.readlines()
            for l in lines:
                if '~' in l:
                    l = l.replace('~', homedir)
                fw.write(l)
    shutil.move(tmppath, ginc)



if __name__ == "__main__":
    args, ginbs = parse_args()

    logging.info(f"Gin configuration files: {args.ginc}")
    logging.info(f"Gin bindings: {args.ginb}")

    run(
        ginc=args.ginc,
        ginb=ginbs,
        scene_name=args.scene_name,
        seed=args.seed,
        add_noise=args.add_noise,
    )



