import argparse
import logging
import os
import random
import shutil
from typing import *
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils.select_option import select_dataset


if __name__ == '__main__':
    dataset_name = "nerf_360_v2"
    datadir = "/Users/Libby/nerf_360_v2"
    scene_name = "garden"

    dataset = select_dataset(
        dataset_name,
        datadir,
        scene_name,
    )

    for d in dataset:
        print(d.shape)
        x = d['target']
        plt.imshow(x.permute(1, 2, 0))
        plt.show()
