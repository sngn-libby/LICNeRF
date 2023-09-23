# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import argparse
import shutil
import random
import sys
from datetime import datetime, timedelta
import math
import time
import os

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import plotly.express as px

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils import data

import numpy as np

from collections import defaultdict, deque

from compressai.zoo import image_models
from src.compressproj.layers.layers import CoordConv2d, CoordDeconv2d, MaskedCoordConv2d


def distribution_counter(counter: defaultdict, vals, precision=4):
    if counter is None:
        counter = defaultdict(int)
    counter['precision'] = precision
    for val in np.array(vals.tolist()):
        counter[np.round(val, decimals=precision)] += 1
    return counter


def save_distribution(counter: defaultdict = None, vals=None, precision=4, filename="", mode='w+') -> defaultdict:
    # with open(os.path.expanduser("~/LUT/results/"+filename), mode) as f:
    with open(os.getcwd() + "/results/" + filename, mode) as f:
        if vals is not None:
            counter = distribution_counter(counter, vals, precision)
        for k, v in counter.items():
            f.write(':'.join([str(k), str(v)]))
            f.write("\n")
    return counter


def save_values(vals, precision=4, filename="", mode="a+"):
    with open(os.getcwd() + "/results/" + filename, mode) as f:
        if vals is not None:
            f.write(",".join(map(str, torch.round(vals, decimals=precision).detach().to("cpu").numpy())))
        f.write('\n')


def sampled_points(x: np.ndarray, n_sample=3, precision=-1):
    """
    Args:
        x: values
        precision:

    Returns:
    """
    x = x.reshape(-1)
    vals, counts = np.unique(np.round(x, decimals=precision) if precision != -1 else x, return_counts=True)
    if n_sample > len(vals):
        n_sample = len(vals)
    probs = counts / np.sum(counts)
    sampled_idx = np.random.choice(np.arange(len(vals)), n_sample, replace=False, p=probs)
    sampled_x = vals[sampled_idx]
    sampled_cnt = counts[sampled_idx]

    return sampled_x, sampled_cnt


def build_fifo_dict_distribution(M, img, scales, means, s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr, size_diff=1, rf=1):
    def update_dicts(dicts, k, v):
        if not hasattr(dicts, k):
            dicts[k] = v

    img = img.transpose((1, 0, 2, 3))
    scales = scales.transpose((1, 0, 2, 3))
    means = means.transpose((1, 0, 2, 3))
    img = np.round(img)

    s = img.min(axis=1).min(axis=1).min(axis=1)
    e = img.max(axis=1).max(axis=1).max(axis=1)
    B, C, H, W = img.shape

    if size_diff == 0:
        img = np.pad(img, ((0, 0), (0, 0), (0, 1), (0, 1)))

    for c in range(M):
        keys = img_to_keys_2d(img[c], s_arr[c], e[c] - s[c] + 1, H-size_diff, W-size_diff)
        uniq_keys = np.unique(keys)
        if upper_bound_arr[c] < len(uniq_keys):
            upper_bound_arr[c] = len(uniq_keys)
            s_arr[c] = s[c]
            mean_dist_arr[c] = dict()
            scale_dist_arr[c] = dict()

        for k in uniq_keys:
            locs = np.where(keys == k)
            if not locs[0].size:
                continue
            tar_scale = (scales[c].reshape(-1)[locs])[0]
            tar_mean = (means[c].reshape(-1)[locs])[0]
            update_dicts(scale_dist_arr[c], k, tar_scale)
            update_dicts(mean_dist_arr[c], k, tar_mean)

    return s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr


def calc_mean_for_indexes(M,
                          img:np.ndarray,
                          scales:np.ndarray,
                          means:np.ndarray,
                          scale_sum_arr,
                          mean_sum_arr,
                          idx_scale_size_arr,
                          idx_mean_size_arr,
                          size_diff=0,):
    img = np.round(img)
    img = img.transpose((1, 0, 2, 3))
    scales = scales.transpose((1, 0, 2, 3))
    means = means.transpose((1, 0, 2, 3))
    s = img.min(axis=1).min(axis=1).min(axis=1)
    e = img.max(axis=1).max(axis=1).max(axis=1)
    B, C, H, W = img.shape

    if size_diff == 0:
        img = np.pad(img, ((0, 0), (0, 0), (0, 1), (0, 1)))

    for c in range(M):
        # if mean_sum_arr[c] is None:
        # if scale_sum_arr[c] is None:
        if idx_scale_size_arr[c] is None:
            mean_sum_arr[c] = dict()
            scale_sum_arr[c] = dict()
            idx_scale_size_arr[c] = dict()
            idx_mean_size_arr[c] = dict()
        # if idx_mean_size_arr[c] is None:
        indexes = img_to_indexes_2d(img[c], s[c], e[c] - s[c] + 1, H-size_diff, W-size_diff)
        uniq_indexes = np.unique(indexes)

        for idx in uniq_indexes:
            locs = np.where(indexes == idx)
            tar_scale = scales[c].reshape(-1)[locs]
            tar_mean = means[c].reshape(-1)[locs]
            with open(f"D:/Research_LUT/parameters/idx/{c}_{idx}_scale.txt", "a+") as f:
                f.write(",".join(list(map(str, tar_scale))))
                f.write("\n")
            with open(f"D:/Research_LUT/parameters/idx/{c}_{idx}_mean.txt", "a+") as f:
                f.write(",".join(list(map(str, tar_mean))))
                f.write("\n")


            if idx in scale_sum_arr[c].keys():
                scale_sum_arr[c][idx] += sum(tar_scale)
                idx_scale_size_arr[c][idx] += len(tar_scale)
            else:
                # print(f"error: c({c}), idx({idx}), tar_scale({tar_scale}), {len(tar_scale)}, {idx_scale_size_arr[c]}")
                scale_sum_arr[c][idx] = sum(tar_scale)
                idx_scale_size_arr[c][idx] = len(tar_scale)

            if idx in mean_sum_arr[c].keys():
                mean_sum_arr[c][idx] += sum(tar_mean)
                idx_mean_size_arr[c][idx] += len(tar_mean)
            else:
                mean_sum_arr[c][idx] = sum(tar_mean)
                idx_mean_size_arr[c][idx] = len(tar_mean)

    return scale_sum_arr, mean_sum_arr, idx_scale_size_arr, idx_mean_size_arr


def build_scaled_freq_distribution(M, img, scales, means,
                                   s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr,
                                   scale_factor=1.,
                                   sampling_gap: int=None, size_diff=0, max_lut_size=-1):
    if sampling_gap is not None:
        img = sampling_values(img, sampling_gap)
    scaler = img
    scaler[scaler == 0] = 1
    img = img * (1 / np.sqrt(np.abs(scaler))) * scale_factor
    return build_freq_distribution(M, img, scales, means, s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr, size_diff, max_lut_size=max_lut_size)


def build_scaled_freq_dict_distribution(M, img, scales, means,
                                        s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr,
                                        scale_factor=1.,
                                        sampling_gap: int=None, size_diff=0, max_lut_size=-1):
    if sampling_gap is not None:
        img = sampling_values(img, sampling_gap)
    scaler = img
    scaler[scaler == 0] = 1
    img = img * (1 / np.sqrt(np.abs(scaler))) * scale_factor
    return build_freq_dict_distribution(M, img, scales, means, s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr, size_diff, max_lut_size=max_lut_size)

def build_mean_distribution(M,
                            img:np.ndarray,
                            scales:np.ndarray,
                            means:np.ndarray,
                            s_arr,
                            upper_bound_arr,
                            scale_dist_arr,
                            mean_dist_arr,
                            n_arr,
                            size_diff=0,
                            max_lut_size=-1):

    def update_arr(arr, n_arr, key, val, update=False):
        if key not in arr.keys():
            arr[key] = val
            if update == False:
                n_arr[key] = 0
            else:
                n_arr[key] = 1
        else:
            m = arr[key]
            n = n_arr[key]

            if update:
                n_arr[key] += 1
            arr[key] = (m / (n+1)) * n + val / (n+1)

    img = img.transpose((1, 0, 2, 3))
    scales = scales.transpose((1, 0, 2, 3))
    means = means.transpose((1, 0, 2, 3))

    # print("z_hat[0]:", np.unique(img[0]), "\n")
    img = np.round(img)

    s = img.min(axis=1).min(axis=1).min(axis=1)
    e = img.max(axis=1).max(axis=1).max(axis=1)
    B, C, H, W = img.shape

    if size_diff == 0:
        img = np.pad(img, ((0, 0), (0, 0), (0, 1), (0, 1)))

    for c in range(M):
        upper_bound = e[c] - s[c] + 1
        if upper_bound_arr[c] < upper_bound:
            upper_bound_arr[c] = min(np.int(upper_bound), max_lut_size)
            s_arr[c] = s[c]
            mean_dist_arr[c] = None # Initialize!
            scale_dist_arr[c] = None
            n_arr[c] = None

        if mean_dist_arr[c] is None:
            scale_dist_arr[c] = dict()
            mean_dist_arr[c] = dict()
            n_arr[c] = dict()

        indexes = img_to_indexes_2d(img[c], s_arr[c], upper_bound_arr[c], H - size_diff, W - size_diff, size=max_lut_size)
        uniq_indexes = np.unique(indexes)

        for idx in uniq_indexes:
            locs = np.where(indexes == idx)
            tar_scale = (scales[c].reshape(-1)[locs])[0]
            tar_mean = (means[c].reshape(-1)[locs])[0]

            update_arr(scale_dist_arr[c], n_arr[c], idx, tar_scale)
            update_arr(mean_dist_arr[c], n_arr[c], idx, tar_mean, update=True)

    return s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr, n_arr


def build_fifo_channel_distribution(M,
                                    img: np.ndarray,
                                    scales: np.ndarray,
                                    means: np.ndarray,
                                    s_arr,
                                    upper_bound_arr,
                                    scale_dist_arr,
                                    mean_dist_arr,
                                    size_diff=1,):
    def update_arr(arr, key, val):
        if key not in arr.keys():
            arr[key] = val

    img = np.round(img)
    img = img.transpose((1, 0, 2, 3))
    scales = scales.transpose((1, 0, 2, 3))
    means = means.transpose((1, 0, 2, 3))

    s = img.min(axis=1).min(axis=1).min(axis=1)

    if size_diff == 0:
        img = np.pad(img, ((0, 0), (0, 0), (0, 1), (0, 1)))

    img = img.sum(axis=0).reshape(-1)

    for c in range(M):
        s_arr[c] = s[c]
        indexes = (img - s[c]).astype(np.int)
        uniq_indexes = np.unique(indexes)
        upper_bound = uniq_indexes.max()
        if upper_bound_arr[c] < upper_bound:
            upper_bound_arr[c] = upper_bound
            scale_dist_arr[c] = dict()
            mean_dist_arr[c] = dict()

        if mean_dist_arr[c] is None:
            scale_dist_arr[c] = dict()
            mean_dist_arr[c] = dict()

        for idx in uniq_indexes:
            if idx in scale_dist_arr[c].keys():
                continue
            locs = np.where(indexes == idx)

            tar_scale = (scales[c].reshape(-1)[locs])[0]
            tar_mean = (means[c].reshape(-1)[locs])[0]

            update_arr(scale_dist_arr[c], idx, tar_scale)
            update_arr(mean_dist_arr[c], idx, tar_mean)

    return s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr

def build_fifo_distribution(M,
                            output:np.ndarray,
                            scales:np.ndarray,
                            means:np.ndarray,
                            s_arr,
                            upper_bound_arr,
                            scale_dist_arr,
                            mean_dist_arr,
                            size_diff=0,
                            rf=2):

    def update_arr(arr, key, val):
        if key not in arr.keys():
            arr[key] = val

    img = output.transpose((1, 0, 2, 3))
    scales = scales.transpose((1, 0, 2, 3))
    means = means.transpose((1, 0, 2, 3))

    img = np.round(img)
    C, B, H, W = img.shape
    # if H == 16:
    #     size_diff = 0

    s = img.min(axis=1).min(axis=1).min(axis=1)
    e = img.max(axis=1).max(axis=1).max(axis=1)

    add = rf - 1 - size_diff
    nH = H + add
    nW = W + add

    # if size_diff == 0:
    #     img = np.pad(img, ((0, 0), (0, 0), (0, 1), (0, 1)))
    if rf > 2:
        img = np.pad(img, ((0, 0), (0, 0), (0, add), (0, add)))
    # if size_diff == 0:
    #     img = np.pad(img, ((0, 0), (0, 0), (0, rf - 1), (0, rf - 1)))
    #     nH += rf - 1
    #     nW += rf - 1

    n_times = M // C
    if n_times > 1:
        img = img.repeat(n_times).reshape([C * n_times, B, nH, nW])
        s = s.repeat(n_times)
        e = e.repeat(n_times)

    for c in range(M):
        upper_bound = e[c] - s[c] + 1
        if upper_bound_arr[c] < upper_bound and (upper_bound ** 4) > 0:
            upper_bound_arr[c] = np.int(upper_bound)
            s_arr[c] = s[c]
            mean_dist_arr[c] = None # Initialize!
            scale_dist_arr[c] = None

        if mean_dist_arr[c] is None:
            scale_dist_arr[c] = dict()
            mean_dist_arr[c] = dict()

        clip_img = np.clip(img[c], s_arr[c], upper_bound_arr[c])
        # indexes = _img_to_indexes_2d(clip_img, s_arr[c], upper_bound_arr[c], H - size_diff, W - size_diff)
        indexes = _img_to_indexes_2d_rf(clip_img, s_arr[c], upper_bound_arr[c], H - size_diff, W - size_diff, rf=rf)
        uniq_indexes = np.unique(indexes)

        f_scale = scales[c].reshape(-1)
        f_mean = means[c].reshape(-1)
        for idx in uniq_indexes:
            locs = np.where(indexes == idx)
            tar_scale = (f_scale[locs])[0]
            tar_mean = (f_mean[locs])[0]

            update_arr(scale_dist_arr[c], idx, tar_scale)
            update_arr(mean_dist_arr[c], idx, tar_mean)

    return s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr


def build_fifo_distribution_3d(M,
                            img:np.ndarray,
                            scales:np.ndarray,
                            means:np.ndarray,
                            s_arr,
                            upper_bound_arr,
                            scale_dist_arr,
                            mean_dist_arr,
                            size_diff=1,
                            rf=2):

    def update_arr(arr, key, val):
        if key not in arr.keys():
            arr[key] = val

    img = img.transpose((1, 0, 2, 3))
    scales = scales.transpose((1, 0, 2, 3))
    means = means.transpose((1, 0, 2, 3))

    img = np.round(img)
    C, B, H, W = img.shape

    n_times = M // C
    if n_times > 1:
        img = img.repeat(n_times).reshape([C * n_times, B, H, W])
        # s = s.repeat(n_times)
        # e = e.repeat(n_times)

    if size_diff == 0:
        img = np.pad(img, ((0, rf - 1), (0, 0), (0, 1), (0, 1))) # channel-wise padding
    else:
        img = np.pad(img, ((0, rf - 1), (0, 0), (0, 0), (0, 0)))

    s = img.min(axis=1).min(axis=1).min(axis=1)
    e = img.max(axis=1).max(axis=1).max(axis=1)

    for c in range(M):
        upper_bound = e[c] - s[c] + 1
        if upper_bound_arr[c] < upper_bound and (upper_bound ** 4) > 0:
            upper_bound_arr[c] = np.int(upper_bound)
            s_arr[c] = s[c]
            mean_dist_arr[c] = None # Initialize!
            scale_dist_arr[c] = None

        if mean_dist_arr[c] is None:
            scale_dist_arr[c] = dict()
            mean_dist_arr[c] = dict()

    for c in range(M):
        clip_imgs = img[c: c + rf]
        s = s_arr[c: c + rf]
        k = upper_bound_arr[c: c + rf]
        if k.size == 1:
            s = s.repeat(rf)
            k = k.repeat(rf)
        # print(c, clip_imgs.shape, k[0])
        for i in range(rf):
            clip_imgs[i] = np.clip(clip_imgs[i], s[i], k[i] - 1)
        indexes = _img_to_indexes_3d(clip_imgs, s, k, H - size_diff, W - size_diff, rf).astype(np.int)

        f_scale = scales[c].reshape(-1)
        f_mean = means[c].reshape(-1)
        indexes = np.clip(indexes, 0, len(f_scale) - 1)

        uniq_indexes = np.unique(indexes)

        for idx in uniq_indexes:
            locs = np.where(indexes == idx)
            tar_scale = (f_scale[locs])[0]
            tar_mean = (f_mean[locs])[0]

            update_arr(scale_dist_arr[c], idx, tar_scale)
            update_arr(mean_dist_arr[c], idx, tar_mean)

    return s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr


def build_freq_distribution(M,
                            img: np.ndarray,
                            scales: np.ndarray,
                            means: np.ndarray,
                            s_arr,
                            upper_bound_arr,
                            scale_dist_arr,
                            mean_dist_arr,
                            size_diff=0,
                            max_lut_size=-1,
                            rf=2,
                            ):

    img = img.transpose((1, 0, 2, 3))
    scales = scales.transpose((1, 0, 2, 3))
    means = means.transpose((1, 0, 2, 3))

    img = np.round(img)

    s = img.min(axis=1).min(axis=1).min(axis=1)
    e = img.max(axis=1).max(axis=1).max(axis=1)
    B, C, H, W = img.shape

    if size_diff == 0:
        img = np.pad(img, ((0, 0), (0, 0), (0, 1), (0, 1)))

    def update_arr(arr, keys, values):
        for (k, v) in zip(keys, values):
            if k not in arr.keys():
                arr[k] = 0
            arr[k] += v

    def normalize_arr(arr):
        k, v = sorted(arr.items(), key=lambda item: item[1], reverse=True)[0]
        arr = dict()
        arr[k] = min(v, 1e2)
        # for k, v in arr.items():
        #     arr[k] -= max_cnt

    max_lut_size = max_lut_size if max_lut_size != -1 else 99999
    for c in range(M):
        if mean_dist_arr[c] is None:
            scale_dist_arr[c] = dict()
            mean_dist_arr[c] = dict()
            s_arr[c] = s[c]
            upper_bound_arr[c] = min(max_lut_size, np.int(e[c] - s[c] + 1))
            # print(s[c], upper_bound_arr[c])

        # indexes = img_to_indexes_2d(img[c], s_arr[c], upper_bound_arr[c], H - size_diff, W - size_diff, size=max_lut_size)
        indexes = _img_to_indexes_2d_rf(img[c], s_arr[c], upper_bound_arr[c], H - size_diff, W - size_diff, rf=rf)
        uniq_indexes = np.unique(indexes)

        for idx in uniq_indexes:
            if idx not in scale_dist_arr[c].keys():
                scale_dist_arr[c][idx] = dict()
                mean_dist_arr[c][idx] = dict()

            locs = np.where(indexes == idx)
            tar_scale = scales[c].reshape(-1)[locs]
            tar_mean = means[c].reshape(-1)[locs]

            update_arr(scale_dist_arr[c][idx], *sampled_points(tar_scale))
            update_arr(mean_dist_arr[c][idx], *sampled_points(tar_mean))

            normalize_arr(scale_dist_arr[c][idx])
            normalize_arr(mean_dist_arr[c][idx])

    return s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr


def build_freq_dict_distribution(M,
                                 img: np.ndarray,
                                 scales: np.ndarray,
                                 means: np.ndarray,
                                 s_arr,
                                 upper_bound_arr,
                                 scale_dist_arr,
                                 mean_dist_arr,
                                 size_diff=0,
                                 max_lut_size=-1,
                                 ):

    img = img.transpose((1, 0, 2, 3))
    scales = scales.transpose((1, 0, 2, 3))
    means = means.transpose((1, 0, 2, 3))

    img = np.round(img)

    s = img.min(axis=1).min(axis=1).min(axis=1)
    e = img.max(axis=1).max(axis=1).max(axis=1)
    B, C, H, W = img.shape

    if size_diff == 0:
        img = np.pad(img, ((0, 0), (0, 0), (0, 1), (0, 1)))

    def update_arr(arr, keys, values):
        for (k, v) in zip(keys, values):
            if k not in arr.keys():
                arr[k] = 0
            arr[k] += v

    def normalize_arr(arr):
        k, v = sorted(arr.items(), key=lambda item: item[1], reverse=True)[0]
        arr = dict()
        arr[k] = min(v, 1e2)

    max_lut_size = max_lut_size if max_lut_size != -1 else 99999
    for c in range(M):
        if mean_dist_arr[c] is None:
            scale_dist_arr[c] = dict()
            mean_dist_arr[c] = dict()
            s_arr[c] = s[c]
            upper_bound_arr[c] = min(max_lut_size, np.int(e[c] - s[c] + 1))

        keys = img_to_keys_2d(img[c], s_arr[c], e[c] - s[c] + 1, H-size_diff, W-size_diff)
        uniq_keys = np.unique(keys)

        for k in uniq_keys:
            if not hasattr(scale_dist_arr[c], k):
                scale_dist_arr[c][k] = dict()
                mean_dist_arr[c][k] = dict()

            locs = np.where(keys == k)
            tar_scale = scales[c].reshape(-1)[locs]
            tar_mean = means[c].reshape(-1)[locs]

            update_arr(scale_dist_arr[c][k], *sampled_points(tar_scale))
            update_arr(mean_dist_arr[c][k], *sampled_points(tar_mean))
            normalize_arr(scale_dist_arr[c][k])
            normalize_arr(mean_dist_arr[c][k])

    return s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr


def build_fifo_distribution_pair(M,
                                 img,
                                 scales: np.ndarray,
                                 means: np.ndarray,
                                 s_arr,
                                 upper_bound_arr,
                                 scale_dist_arr,
                                 mean_dist_arr,
                                 size_diff=0):

    def update_dict(arr_dict, key, val):
        if key not in arr_dict.keys():
            arr_dict[key] = val

    def update_arr(src_scale_arr, src_mean_arr, scale_arr, mean_arr, indexes):
        uniq_indexes = np.unique(indexes)
        for idx in uniq_indexes:
            locs = np.where(indexes == idx)
            tar_scale = (src_scale_arr.reshape(-1)[locs])[0]
            tar_mean = (src_mean_arr.reshape(-1)[locs])[0]
            update_dict(scale_arr, idx, tar_scale)
            update_dict(mean_arr, idx, tar_mean)

    img = np.round(img)

    img = img.transpose((1, 0, 2, 3))
    scales = scales.transpose((1, 0, 2, 3))
    means = means.transpose((1, 0, 2, 3))

    s = img.min(axis=1).min(axis=1).min(axis=1)
    e = img.max(axis=1).max(axis=1).max(axis=1)
    B, C, H, W = img.shape

    if size_diff == 0:
        img = np.pad(img, ((0, 0), (0, 0), (0, 1), (0, 1)))

    for c in range(M):
        upper_bound = e[c] - s[c] + 1
        if upper_bound_arr[c] < upper_bound:
            upper_bound_arr[c] = np.int(upper_bound)
            s_arr[c] = s[c]
            mean_dist_arr[c][0] = mean_dist_arr[c][1] = None # Initialize!
            scale_dist_arr[c][0] = scale_dist_arr[c][1] = None

        if mean_dist_arr[c][0] is None:
            scale_dist_arr[c][0] = dict()
            scale_dist_arr[c][1] = dict()
            mean_dist_arr[c][0] = dict()
            mean_dist_arr[c][1] = dict()

        indexes1, indexes2 = img_to_indexes_pair(img[c], s_arr[c], upper_bound_arr[c], H - size_diff, W - size_diff)
        update_arr(indexes2, indexes2, scale_dist_arr[c][0], mean_dist_arr[c][0], indexes1)
        update_arr(scales[c], means[c], scale_dist_arr[c][1], mean_dist_arr[c][1], indexes2)

    return s_arr, upper_bound_arr, scale_dist_arr, mean_dist_arr


def select_fifo_value_from_distribution_pair(M, scale_dist_arr, mean_dist_arr, filename="values.txt"):
    with open("D:/Research_LUT/parameters/" + filename, "w+") as f:
        f.write("")

    def iterate_arr(arr):
        res_arr = dict()
        for idx, val in arr.items():
            if idx in res_arr.keys():
                res_arr[idx] += [val]
            else:
                res_arr[idx] = [val]
        return res_arr

    for c in range(M):
        res_arr = iterate_arr(scale_dist_arr[c][0])

        # for idx, scale_dict in scale_dist_arr[c][0]


def select_frequently_appeared_value_from_distribution(M, scale_dist_arr, mean_dist_arr, filename="values.txt"):
    with open("D:/Research_LUT/parameters/"+filename, "w+") as f:
        f.write("")

    for c in range(M):
        res_arr = dict()
        for idx, scale_dict in scale_dist_arr[c].items():
            sorted_scale = sorted(scale_dict.items(), key=lambda item: item[1], reverse=True)
            res_arr[idx] = [sorted_scale[0][0]]
            # res_arr[idx] = [scale_dict]
        for idx, mean_dict in mean_dist_arr[c].items():
            sorted_mean = sorted(mean_dict.items(), key=lambda item: item[1], reverse=True)
            res_arr[idx] += [sorted_mean[0][0]]
            # res_arr[idx] += [mean_dict]

        with open("D:/Research_LUT/parameters/"+filename, "a+") as f:
            f.write(f"channel:{c}\n")
            f.write("\n".join([f"{k}:{v}" for k, v in res_arr.items()]))
            f.write("\n")


fig_idx = 0
def draw_distribution(counter: defaultdict, precision=4, title=""):
    # Tiding up the counter precision
    nc = defaultdict(int)
    if precision != counter['precision']:
        for k, v in counter.items():
            nc[np.round(k, decimals=precision)] += v
    else:
        nc = counter

    global fig_idx

    # Plot Distiribution
    # print(list(nc.keys())[0], list(nc.values())[0])
    plt.scatter(list(nc.keys())[1:], list(nc.values())[1:], alpha=0.6)
    plt.xlabel("values")
    plt.ylabel("counts")
    plt.title(title)
    # plt.legend()
    plt.savefig(f"D:/Research_LUT/distribution_results/distribution_{int(fig_idx // 2):03d}_{int(fig_idx % 2)}.png")

    fig_idx += 1
    if fig_idx // 2 == 192:
        fig_idx = 0


def sampling_indexes(indexes: np.ndarray, gap=2):
    uniq_indexes = np.unique(indexes)
    sorted_uniq_indexes = np.sort(uniq_indexes)
    sampled_uniq_indexes = sorted_uniq_indexes[::2]
    for i, idx in enumerate(sampled_uniq_indexes[1::2]):
        indexes[indexes == idx] = sampled_uniq_indexes[i * 2]
    return indexes


def sampling_values(vals: np.ndarray, gap=2):
    shape = vals.shape
    vals = sampling_indexes(vals.reshape(-1), gap)
    return vals.reshape(shape)


# ==================== LUT ===================== #

def clip_range(img: np.ndarray,
               under_threshold: int = None, under_val: int = None,
               upper_threshold: int = None, upper_val: int = None):
    if under_threshold is None and upper_threshold is None:
        return img

    res = img.copy()
    if under_threshold is not None:
        under_val = under_val if under_val is not None else under_threshold
        res[img < under_threshold] = under_val
    if upper_threshold is not None:
        upper_val = upper_val if upper_val is not None else upper_threshold
        res[img > upper_threshold] = upper_val
    return res


def get_sampling_val(s, k, size):
    # if size is None or (k <= size).all():
    #     return s
    # return size // 2 + int(abs(s) > (size // 2))
    return k // 2


def img_to_indexes_pair(img: np.ndarray, s: int, k: int, H: int, W: int, size=None):
    clip_size = k if (size is None or k <= size) else size

    img_tl = clip_range(img[:, 0:H, 0:W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1)
    img_tr = clip_range(img[:, 0:H, 1:1 + W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1)
    img_bl = clip_range(img[:, 1:1 + H, 0:W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1)
    img_br = clip_range(img[:, 1:1 + H, 1:1 + W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1)

    idx1 = img_tl * k + img_tr
    idx2 = img_bl * k + img_br
    # indexes = img_tl * (k ** 3) + img_tr * (k ** 2) + img_bl * k + img_br
    return idx1, idx2


def img_to_indexes_2d_size(img: np.ndarray, s: int, k: int, H: int, W: int, size = None):
    clip_size = k if (size is None or k <= size) else size
    upper_bound = (clip_size ** 4.) - 1

    # Resolution
    # img *= (10 ** 2)

    img_tl = clip_range(img[:, 0:H, 0:W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1)
    img_tr = clip_range(img[:, 0:H, 1:1 + W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1)
    img_bl = clip_range(img[:, 1:1 + H, 0:W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1)
    img_br = clip_range(img[:, 1:1 + H, 1:1 + W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1)

    indexes = img_tl * (k ** 3) + img_tr * (k ** 2) + img_bl * k + img_br
    indexes = clip_range(indexes, under_threshold=0, upper_threshold=upper_bound).astype(np.int)
    return indexes, clip_size


def img_to_indexes_2d(img: np.ndarray, s: int, k: int, H: int, W: int, size: int = None):
    indexes, _ = img_to_indexes_2d_size(img, s, k, H, W, size)
    return indexes


def img_to_keys_2d(img: np.ndarray, s: int, k: int, H: int, W: int, size: int=None):
    clip_size = k if (size is None or k <= size) else size

    img_tl = clip_range(img[:, 0:H, 0:W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1).astype(int).astype(str)
    img_tr = clip_range(img[:, 0:H, 1:1 + W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1).astype(int).astype(str)
    img_bl = clip_range(img[:, 1:1 + H, 0:W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1).astype(int).astype(str)
    img_br = clip_range(img[:, 1:1 + H, 1:1 + W].flatten() - s, under_threshold=0, upper_threshold=clip_size-1).astype(int).astype(str)

    keys = []
    for tl, tr, bl, br in zip(img_tl, img_tr, img_bl, img_br):
        keys += [tl + tr + bl + br]

    return np.array(keys)


def _img_to_indexes_2d_rf(img: np.ndarray, s, k, H, W, rf=2):
    indexes = 0
    for i in range(rf):
        for j in range(rf):
            _img = img[:, i:H + i, j: W + j].flatten() - s
            _img = np.clip(_img, 0, (rf ** 2 - 1) * k)
            indexes = indexes * k + _img
    return indexes



def _img_to_indexes_2d(img: np.ndarray, s, k, H, W):
    img_tl = img[:, 0:H, 0:W].flatten() - s
    img_tr = img[:, 0:H, 1:1 + W].flatten() - s
    img_bl = img[:, 1:1 + H, 0:W].flatten() - s
    img_br = img[:, 1:1 + H, 1:1 + W].flatten() - s
    indexes = img_tl * (k ** 3) + img_tr * (k ** 2) + img_bl * k + img_br
    return indexes


def _img_to_indexes_3d(img: np.ndarray, s, k, H, W, rf):
    assert len(img.shape) == 4, ":: Error :: Invalid input shape"
    indexes = 0
    prev_k = 0
    for i in range(rf):
        tl = img[i, :, 0:H, 0:W].flatten() - s[i]
        tr = img[i, :, 0:H, 1:1+W].flatten() - s[i]
        bl = img[i, :, 1:1+H, 0:W].flatten() - s[i]
        br = img[i, :, 1:1+H, 1:1+W].flatten() - s[i]
        indexes = indexes * (prev_k ** 4) + (tl * k[i] ** 3) + (tr * k[i] ** 2) + (bl * k[i]) + br
        prev_k = k[i]
    return indexes


def img_to_indexes_3d(img: np.ndarray, s: np.ndarray, k: np.ndarray, H: int, W: int, rf: int, size: int = None):
    indexes = 0
    if size is None:
        size = max(k)
    for i in range(rf):
        half = get_sampling_val(s[i], k[i], size)
        tl = clip_range((img[i, :, 0:H, 0:W] - half).flatten(), under_threshold=0, under_val=0, upper_threshold=size,
                        upper_val=size)
        tr = clip_range((img[i, :, 0:H, 1:1 + W] - half).flatten(), under_threshold=0, under_val=0,
                        upper_threshold=size, upper_val=size)
        bl = clip_range((img[i, :, 1:1 + H, 0:W] - half).flatten(), under_threshold=0, under_val=0,
                        upper_threshold=size, upper_val=size)
        br = clip_range((img[i, :, 1:1 + H, 1:1 + W] - half).flatten(), under_threshold=0, under_val=0,
                        upper_threshold=size, upper_val=size)
        indexes = indexes * (k[i] ** 4) + tl * (k[i] ** 3) + tr * (k[i] ** 2) + bl * k[i] + br
    return indexes


# ===================================================== #


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
                sys.stdout.flush()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.2f} ({global_avg:.2f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class CustomDataset(data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super(CustomDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(args, state, is_best=False, filename="checkpoint.pth.tar"):
    now = str(datetime.now().date()).replace('-', '')[2:]
    seed = str(args.seed)
    filename = "./" + now + '_' + seed + '_' + filename
    print(f":: Save :: {os.getcwd() + filename}")
    torch.save(state, filename)
    bestloss_filename = None
    if is_best:
        bestloss_filename = now + '_' + seed + '_' + "checkpoint_best_loss.pth.tar"
        shutil.copyfile(filename, bestloss_filename)
    return filename, bestloss_filename


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=list(image_models.keys()).append(['checker2022', 'direct2022']),
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "--test-dataset", type=str, help="Test dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
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
        default=4,
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

    # LUT
    parser.add_argument(
        "--k", default=11, type=int, help="LUT's number of division (default: %(default))",
    )

    # Cherrypick
    parser.add_argument("--cherrypick", action="store_true", default=False, help="Save cherrypick model to disk")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--lut_checkpoint", type=str, help="Path to a lut model (numpy) checkpoint")
    parser.add_argument("--cherrypick_checkpoint", type=str, help="Path to a cherrypick network checkpoint")

    # DE
    parser.add_argument("--de_config", type=str, help="Path to a depth estimation (iDisc) config")
    parser.add_argument("--de_ckpt", type=str, help="Path to a depth estimation (iDisc) checkpoint")

    args = parser.parse_args(argv)
    return args


def find_named_module(module, query):
    """Helper function to find a named module. Returns a `nn.Module` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the module name to find

    Returns:
        nn.Module or None
    """

    return next((m for n, m in module.named_modules() if n == query), None)


def find_named_buffer(module, query):
    """Helper function to find a named buffer. Returns a `torch.Tensor` or `None`

    Args:
        module (nn.Module): the root module
        query (str): the buffer name to find

    Returns:
        torch.Tensor or None
    """
    return next((b for n, b in module.named_buffers() if n == query), None)


def _update_registered_buffer(
        module,
        buffer_name,
        state_dict_key,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
):
    new_size = state_dict[state_dict_key].size()
    registered_buf = find_named_buffer(module, buffer_name)

    if policy in ("resize_if_empty", "resize"):
        if registered_buf is None:
            raise RuntimeError(f'buffer "{buffer_name}" was not registered')

        if policy == "resize" or registered_buf.numel() == 0:
            registered_buf.resize_(new_size)

    elif policy == "register":
        if registered_buf is not None:
            raise RuntimeError(f'buffer "{buffer_name}" was already registered')

        module.register_buffer(buffer_name, torch.empty(new_size, dtype=dtype).fill_(0))

    else:
        raise ValueError(f'Invalid policy "{policy}"')


def update_registered_buffers(
        module,
        module_name,
        buffer_names,
        state_dict,
        policy="resize_if_empty",
        dtype=torch.int,
):
    """Update the registered buffers in a module according to the tensors sized
    in a state_dict.

    (There's no way in torch to directly load a buffer with a dynamic size)

    Args:
        module (nn.Module): the module
        module_name (str): module name in the state dict
        buffer_names (list(str)): list of the buffer names to resize in the module
        state_dict (dict): the state dict
        policy (str): Update policy, choose from
            ('resize_if_empty', 'resize', 'register')
        dtype (dtype): Type of buffer to be registered (when policy is 'register')
    """
    valid_buffer_names = [n for n, _ in module.named_buffers()]
    for buffer_name in buffer_names:
        if buffer_name not in valid_buffer_names:
            raise ValueError(f'Invalid buffer name "{buffer_name}"')

    for buffer_name in buffer_names:
        _update_registered_buffer(
            module,
            buffer_name,
            f"{module_name}.{buffer_name}",
            state_dict,
            policy,
            dtype,
        )


def depthwise_conv(in_channels, out_channels, kernel_size=1, stride=1, padding=-1, dilation=1, groups=1):
    padding = kernel_size // 2 if padding == -1 else padding
    groups = in_channels if groups == 1 else groups
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=out_channels if in_channels > out_channels else in_channels,
        # groups=groups,
    )


def conv(in_channels, out_channels, kernel_size=5, stride=2, padding=-1, dilation=1):
    padding = kernel_size // 2 if padding == -1 else padding
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )


def coord_conv(in_channles, out_channels, kernel_size=5, stride=2, padding=-1, dilation=1):
    padding = kernel_size // 2 if padding == -1 else padding
    return CoordConv2d(
        in_channles + 2,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )


def depthwise_deconv(in_channels, out_channels, kernel_size=1, stride=2, output_padding=-1, padding=-1, dilation=1, groups=-1):
    output_padding = stride - 1 if output_padding == -1 else output_padding
    padding = kernel_size // 2 if padding == -1 else padding
    groups = out_channels if in_channels > out_channels else in_channels if groups == -1 else groups
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=output_padding,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2, output_padding=-1, padding=-1, dilation=1):
    output_padding = stride - 1 if output_padding == -1 else output_padding
    padding = kernel_size // 2 if padding == -1 else padding
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=output_padding,
        padding=padding,
        dilation=dilation
    )


def coord_deconv(in_channles, out_channels, kernel_size=5, stride=2, output_padding=-1, padding=-1, dilation=1):
    output_padding = stride - 1 if output_padding == -1 else output_padding
    padding = kernel_size // 2 if padding == -1 else output_padding
    return CoordDeconv2d(
        in_channles + 2,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=output_padding,
        padding=padding,
        dilation=dilation
    )


def quantize_ste(x):
    """Differentiable quantization via the Straight-Through-Estimator."""
    # STE (straight-through estimator) trick: x_hard - x_soft.detach() + x_soft
    return (torch.round(x) - x).detach() + x


def gaussian_kernel1d(
        kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """1D Gaussian kernel."""
    khalf = (kernel_size - 1) / 2.0
    x = torch.linspace(-khalf, khalf, steps=kernel_size, dtype=dtype, device=device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def gaussian_kernel2d(
        kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype
):
    """2D Gaussian kernel."""
    kernel = gaussian_kernel1d(kernel_size, sigma, device, dtype)
    return torch.mm(kernel[:, None], kernel[None, :])


def gaussian_blur(x, kernel=None, kernel_size=None, sigma=None):
    """Apply a 2D gaussian blur on a given image tensor."""
    if kernel is None:
        if kernel_size is None or sigma is None:
            raise RuntimeError("Missing kernel_size or sigma parameters")
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        device = x.device
        kernel = gaussian_kernel2d(kernel_size, sigma, device, dtype)

    padding = kernel.size(0) // 2
    x = F.pad(x, (padding, padding, padding, padding), mode="replicate")
    x = torch.nn.functional.conv2d(
        x,
        kernel.expand(x.size(1), 1, kernel.size(0), kernel.size(1)),
        groups=x.size(1),
    )
    return x


def meshgrid2d(N: int, C: int, H: int, W: int, device: torch.device):
    """Create a 2D meshgrid for interpolation."""
    theta = torch.eye(2, 3, device=device).unsqueeze(0).expand(N, 2, 3)
    return F.affine_grid(theta, (N, C, H, W), align_corners=False)
