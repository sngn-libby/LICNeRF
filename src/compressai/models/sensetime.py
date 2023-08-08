import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from src.compressai.entropy_models import EntropyBottleneck, GaussianConditional
from src.compressai.layers import GDN, MaskedConv2d, CheckerboardConv
from src.compressai.registry import register_model
from src.compressai.models.google import JointAutoregressiveHierarchicalPriors

from .base import (
    SCALES_LEVELS,
    SCALES_MAX,
    SCALES_MIN,
    CompressionModel,
    get_scale_table,
)
from .utils import conv, deconv

__all__ = [
    "JointCheckerboardHierarchicalPriors",
]


@register_model("checkerboard2021")
class JointCheckerboardHierarchicalPriors(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.context_prediction = CheckerboardConv(M, 2 * M)

    def forward(self, x):
        device = x.device

        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        anchor = torch.zeros_like(y_hat).to(device)
        non_anchor = torch.zeros_like(y_hat).to(device)

        anchor[:, :, 0::2, 1::2] = y_hat[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y_hat[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, 0::2] = y_hat[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, 1::2] = y_hat[:, :, 1::2, 1::2]

        # Anchor
        ctx_params_anchor = torch.zeros_like(params)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scale_anchor, mean_anchor = gaussian_params_anchor.chunk(2, 1)

        # Non-Anchor
        ctx_params_non_anchor = self.context_prediction(anchor)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scale_non_anchor, mean_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales = torch.zeros_like(scale_anchor)
        means = torch.zeros_like(mean_anchor)

        # Anchor + Non-Anchor
        scales[:, :, 0::2, 1::2] = scale_anchor[:, :, 0::2, 1::2]
        scales[:, :, 1::2, 0::2] = scale_anchor[:, :, 1::2, 0::2]
        scales[:, :, 0::2, 0::2] = scale_non_anchor[:, :, 0::2, 0::2]
        scales[:, :, 1::2, 1::2] = scale_non_anchor[:, :, 1::2, 1::2]
        means[:, :, 0::2, 1::2] = mean_anchor[:, :, 0::2, 1::2]
        means[:, :, 1::2, 0::2] = mean_anchor[:, :, 1::2, 0::2]
        means[:, :, 0::2, 0::2] = mean_non_anchor[:, :, 0::2, 0::2]
        means[:, :, 1::2, 1::2] = mean_non_anchor[:, :, 1::2, 1::2]

        _, y_likelihoods = self.gaussian_conditional(y, scales, means=means)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        start_time = time.process_time()
        device = x.device

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        start_log_time = time.process_time()

        B, C, H, W = y.shape
        anchor = torch.zeros((B, C, H, W//2 + W%2)).to(device)
        non_anchor = torch.zeros((B, C, H, W//2 + W%2)).to(device)
        half_scale = torch.zeros((B, C, H, W//2 + W%2)).to(device)
        half_mean = torch.zeros((B, C, H, W//2 + W%2)).to(device)

        anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]

        # Anchor
        params = self.h_s(z_hat)
        ctx_params_anchor = torch.zeros_like(params)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        half_scale[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 1::2]
        half_scale[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 0::2]
        half_mean[:, :, 0::2, :] = means_anchor[:, :, 0::2, 1::2]
        half_mean[:, :, 1::2, :] = means_anchor[:, :, 1::2, 0::2]
        indexes_anchor = self.gaussian_conditional.build_indexes(half_scale)
        anchor_strings = self.gaussian_conditional.compress(anchor, indexes_anchor, means=half_mean)
        anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=half_mean)

        # Non-Anchor
        y_hat = torch.zeros_like(y)
        y_hat[:, :, 0::2, 1::2] = anchor_quantized[:, :, 0::2, :]
        y_hat[:, :, 1::2, 0::2] = anchor_quantized[:, :, 1::2, :]
        ctx_params_non_anchor = self.context_prediction(y_hat)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        half_scale = torch.zeros((B, C, H, W//2 + W%2)).to(device)
        half_mean = torch.zeros((B, C, H, W//2 + W%2)).to(device)
        half_scale[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 0::2]
        half_scale[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 1::2]
        half_mean[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 0::2]
        half_mean[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 1::2]
        end_log_time = time.process_time()

        indexes_non_anchor = self.gaussian_conditional.build_indexes(half_scale)
        non_anchor_strings = self.gaussian_conditional.compress(non_anchor, indexes_non_anchor, means=half_mean)

        end_time = time.process_time()
        cost_time = end_time - start_time
        tar_cost_time = end_log_time - start_log_time

        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape": z.size()[-2:],
            "cost_time": cost_time,
            "tar_cost_time": tar_cost_time,
        }

    def decompress(self, strings, shape):
        start_time = time.process_time()

        z_hat = self.entropy_bottleneck.decompress(strings[-1], shape)
        device = z_hat.device

        start_log_time = time.process_time()

        params = self.h_s(z_hat)

        B, C, H, W = params.shape
        y_hat = torch.zeros((B, self.M, H, W)).to(device)
        half_scale = torch.zeros((B, self.M, H, W//2 + W%2)).to(device)
        half_mean = torch.zeros((B, self.M, H, W//2 + W%2)).to(device)

        # Anchor
        ctx_params_anchor = torch.zeros_like(params)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        half_scale[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 1::2]
        half_scale[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 0::2]
        half_mean[:, :, 0::2, :] = means_anchor[:, :, 0::2, 1::2]
        half_mean[:, :, 1::2, :] = means_anchor[:, :, 1::2, 0::2]
        indexes_anchor = self.gaussian_conditional.build_indexes(half_scale)
        anchor_quantized = self.gaussian_conditional.decompress(strings[0], indexes_anchor, means=half_mean)

        # Non-Anchor
        y_hat[:, :, 0::2, 1::2] = anchor_quantized[:, :, 0::2, :]
        y_hat[:, :, 1::2, 0::2] = anchor_quantized[:, :, 1::2, :]
        ctx_params_non_anchor = self.context_prediction(y_hat)
        ctx_params_non_anchor[:, :, 0::2, 1::2] = 0
        ctx_params_non_anchor[:, :, 1::2, 0::2] = 0
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        # print("\n", y_hat[0][0], "\n", ctx_params_non_anchor[0][0], "\n", scales_non_anchor[0][0])
        half_scale[:, :, 0::2, :] = scales_non_anchor[:, :, 0::2, 0::2]
        half_scale[:, :, 1::2, :] = scales_non_anchor[:, :, 1::2, 1::2]
        half_mean[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 0::2]
        half_mean[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 1::2]

        end_log_time = time.process_time()

        indexes_non_anchor = self.gaussian_conditional.build_indexes(half_scale)
        non_anchor_quantized = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor, means=half_mean)
        y_hat[:, :, 0::2, 0::2] = non_anchor_quantized[:, :, 0::2, :]
        y_hat[:, :, 1::2, 1::2] = non_anchor_quantized[:, :, 1::2, :]

        x_hat = self.g_s(y_hat)

        end_time = time.process_time()
        cost_time = end_time - start_time
        tar_cost_time = end_log_time - start_log_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time,
            "tar_cost_time": tar_cost_time,
        }