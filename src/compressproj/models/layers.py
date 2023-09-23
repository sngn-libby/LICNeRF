import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.compressproj.layers import GDN, MaskedConv2d
from src.compressproj.models.utils import *



def build_conv_layers(rf_string: str,
                      size_list,
                      stride_list=None,
                      padding_list=None,
                      dilation_list=None,
                      synthesis=False,
                      activation="GDN"):

    assert activation in ["GDN", "ReLU", "LeakyReLU"]

    # rf_string = "3d-1d-1"
    # size_list = [(N, M), (M, M * 2), (M * 2, M * 2)]
    layer_rf = rf_string.split('-')
    if stride_list is None:
        stride_list = [1] * len(layer_rf)
    if padding_list is None:
        padding_list = np.array([int(i[0]) for i in layer_rf]) // 2
    if dilation_list is None:
        dilation_list = [1] * len(layer_rf)

    conv_layer = nn.Sequential()
    for i, rf in enumerate(layer_rf[:-1]):
        kernel_size = int(rf[0])
        dilation = dilation_list[i]
        conv_layer.append(
            conv(*size_list[i], stride=stride_list[i], kernel_size=kernel_size, padding=padding_list[i], dilation=dilation) if rf[-1] != "d" else \
            depthwise_conv(*size_list[i], stride=stride_list[i], kernel_size=kernel_size, padding=padding_list[i], dilation=dilation)
        )
        if synthesis:
            conv_layer.append(
                GDN(size_list[i][-1]) if activation=="GDN"
                else nn.ReLU(inplace=True) if activation=="ReLU"
                else nn.LeakyReLU(inplace=True))
        else:
            conv_layer.append(nn.LeakyReLU(inplace=True))
    kernel_size = int(layer_rf[-1][0])
    conv_layer.append(
        conv(*size_list[-1], stride=stride_list[-1], kernel_size=kernel_size, padding=padding_list[-1], dilation=dilation_list[-1]) if layer_rf[-1][-1] != "d" else\
        depthwise_conv(*size_list[-1], stride=stride_list[-1], kernel_size=kernel_size, padding=padding_list[-1], dilation=dilation_list[-1])
    )

    return conv_layer


def build_deconv_layers(rf_string: str,
                        size_list,
                        stride_list=None,
                        padding_list=None,
                        dilation_list=None,
                        synthesis=False,
                        activation="GDN"):
    assert activation in ["GDN", "ReLU", "LeakyReLU"]

    layer_rf = rf_string.split('-')
    if stride_list is None:
        stride_list = [1] * len(layer_rf)
    if padding_list is None:
        padding_list = np.array([int(i[0]) for i in layer_rf]) // 2
    if dilation_list is None:
        dilation_list = [1] * len(layer_rf)

    deconv_layer = nn.Sequential()
    for i, rf in enumerate(layer_rf[:-1]):
        kernel_size = int(rf[0])
        stride = stride_list[i]
        padding = padding_list[i]
        dilation = dilation_list[i]
        deconv_layer.append(
            deconv(*size_list[i], stride=stride, kernel_size=kernel_size, padding=padding, dilation=dilation) if rf[-1] != "d" else\
            depthwise_deconv(*size_list[i], stride=stride, kernel_size=kernel_size, padding=padding, dilation=dilation)
        )
        if synthesis:
            deconv_layer.append(
                GDN(size_list[i][-1], inverse=True) if activation=="GDN"
                else nn.ReLU(inplace=True) if activation=="ReLU"
                else nn.LeakyReLU(inplace=True))
        else:
            deconv_layer.append(nn.LeakyReLU(inplace=True))
    kernel_size = int(layer_rf[-1][0])
    if synthesis:
        deconv_layer.append(
            deconv(*size_list[-1], stride=stride_list[-1], kernel_size=kernel_size, padding=padding_list[-1], dilation=dilation_list[-1]) if rf[-1][-1] != "d" else \
            depthwise_deconv(*size_list[-1], stride=stride_list[-1], kernel_size=kernel_size, padding=padding_list[-1], dilation=dilation_list[-1])
        )
    else:
        deconv_layer.append(
            conv(*size_list[-1], stride=stride_list[-1], kernel_size=kernel_size, padding=padding_list[-1], dilation=dilation_list[-1]) if layer_rf[-1][-1] != "d" else\
            depthwise_conv(*size_list[-1], stride=stride_list[-1], kernel_size=kernel_size, padding=padding_list[-1], dilation=dilation_list[-1])
        )

    return deconv_layer


def vanilla_g_a(N, M):
    return nn.Sequential(
        conv(3, N, kernel_size=5, stride=2),
        GDN(N),
        conv(N, N, kernel_size=5, stride=2),
        GDN(N),
        conv(N, N, kernel_size=5, stride=2),
        GDN(N),
        conv(N, M, kernel_size=5, stride=2),
    )


def vanilla_g_s(N, M):
    return nn.Sequential(
        deconv(M, N, kernel_size=5, stride=2),
        GDN(N, inverse=True),
        deconv(N, N, kernel_size=5, stride=2),
        GDN(N, inverse=True),
        deconv(N, N, kernel_size=5, stride=2),
        GDN(N, inverse=True),
        deconv(N, 3, kernel_size=5, stride=2),
    )


def vanilla_coord_g_a(N, M):
    return nn.Sequential(
        coord_conv(3, N, kernel_size=5, stride=2),
        GDN(N),
        coord_conv(N, N, kernel_size=5, stride=2),
        GDN(N),
        coord_conv(N, N, kernel_size=5, stride=2),
        GDN(N),
        coord_conv(N, M, kernel_size=5, stride=2),
    )


def vanilla_coord_g_s(N, M):
    return nn.Sequential(
        coord_deconv(M, N, kernel_size=5, stride=2),
        GDN(N, inverse=True),
        coord_deconv(N, N, kernel_size=5, stride=2),
        GDN(N, inverse=True),
        coord_deconv(N, N, kernel_size=5, stride=2),
        GDN(N, inverse=True),
        coord_deconv(N, 3, kernel_size=5, stride=2),
    )


def vanilla_coord_h_a(N, M):
    return nn.Sequential(
        coord_conv(M, N, stride=1, kernel_size=3),
        nn.LeakyReLU(inplace=True),
        coord_conv(N, N, stride=2, kernel_size=5),
        nn.LeakyReLU(inplace=True),
        coord_conv(N, N, stride=2, kernel_size=5),
    )


def vanilla_coord_h_s(N, M):
    return nn.Sequential(
        coord_deconv(N, M, stride=2, kernel_size=5),
        nn.LeakyReLU(inplace=True),
        coord_deconv(M, M * 3 // 2, stride=2, kernel_size=5),
        nn.LeakyReLU(inplace=True),
        coord_conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
    )


def vanilla_h_a(N, M):
    return nn.Sequential(
        conv(M, N, stride=1, kernel_size=3),
        nn.LeakyReLU(inplace=True),
        conv(N, N, stride=2, kernel_size=5),
        nn.LeakyReLU(inplace=True),
        conv(N, N, stride=2, kernel_size=5),
    )


def vanilla_h_s(N, M):
    return nn.Sequential(
        deconv(N, M, stride=2, kernel_size=5),
        nn.LeakyReLU(inplace=True),
        deconv(M, M * 3 // 2, stride=2, kernel_size=5),
        nn.LeakyReLU(inplace=True),
        conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
    )


def rf27_h_a(N, M, stride=1):
    return nn.Sequential(
        conv(M, N, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(N, N, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(N, N, stride=stride, kernel_size=3),
    )


def depthwise_rf9_h_s(N, M, stride=1):
    return nn.Sequential(
        depthwise_deconv(N, M, stride=stride, kernel_size=3),
        nn.LeakyReLU(inplace=True),
        depthwise_deconv(M, M * 2, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        depthwise_conv(M * 2, M * 2, stride=1, kernel_size=1),
    )


def depthwise_rf27_h_s(N, M, stride=1):
    return nn.Sequential(
        deconv(N, M, stride=stride, kernel_size=3),
        nn.LeakyReLU(inplace=True),
        depthwise_deconv(M, M * 2, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        depthwise_conv(M * 2, M * 2, stride=1, kernel_size=1),
    )


def rf8_h_a(N, M):
    return nn.Sequential(
        conv(M, N, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(N, N, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(N, N, stride=1, kernel_size=2),
    )


def depthwise_rf4_h_a(N, M):
    return nn.Sequential(
        depthwise_conv(M, N, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        depthwise_conv(N, N, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        depthwise_conv(N, N, stride=1, kernel_size=2),
    )


def rf8_h_s(N, M):
    return nn.Sequential(
        deconv(N, M, stride=1, kernel_size=2),
        nn.LeakyReLU(inplace=True),
        deconv(M, M * 2, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(M * 2, M * 2, stride=1, kernel_size=1),
    )


def partial_rf4_h_s(N, M):
    return nn.Sequential(
        depthwise_deconv(N, M, stride=1, kernel_size=2),
        nn.LeakyReLU(inplace=True),
        deconv(M, M * 2, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(M * 2, M * 2, stride=1, kernel_size=1),
    )


def depthwise_rf4_h_s(N, M):
    return nn.Sequential(
        depthwise_deconv(N, M, stride=1, kernel_size=2),
        nn.LeakyReLU(inplace=True),
        depthwise_deconv(M, M * 2, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        depthwise_conv(M * 2, M * 2, stride=1, kernel_size=1),
    )


def depthwise_rf8_h_a(N, M):
    return nn.Sequential(
        depthwise_conv(M, N, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        depthwise_conv(N, N, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        conv(N, N, stride=1, kernel_size=2),
    )


def depthwise_rf8_h_s(N, M):
    return nn.Sequential(
        deconv(N, M, stride=1, kernel_size=2),
        nn.LeakyReLU(inplace=True),
        depthwise_deconv(M, M * 2, stride=1, kernel_size=1),
        nn.LeakyReLU(inplace=True),
        depthwise_conv(M * 2, M * 2, stride=1, kernel_size=1),
    )


def entropy_parameters(N, M):
    return nn.Sequential(
        nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
    )


def coord_entropy_parameters(N, M):
    return nn.Sequential(
        coord_conv(M * 12 // 3, M * 10 // 3, kernel_size=1, stride=1),
        nn.LeakyReLU(inplace=True),
        coord_conv(M * 10 // 3, M * 8 // 3, kernel_size=1, stride=1),
        nn.LeakyReLU(inplace=True),
        coord_conv(M * 8 // 3, M * 6 // 3, kernel_size=1, stride=1),
    )


def depthwise_entropy_parameters(N, M):
    return nn.Sequential(
        depthwise_conv(M * 4, M * 4, 1),
        nn.LeakyReLU(inplace=True),
        depthwise_conv(M * 4, M * 4, 1),
        nn.LeakyReLU(inplace=True),
        depthwise_conv(M * 4, M * 2, 1),
    )
