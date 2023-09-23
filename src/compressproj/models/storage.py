
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

from compressai.ans import BufferedRansEncoder, RansDecoder
from src.compressproj import env
from src.compressproj.models.utils import *
from src.compressproj.datasets import ImageFolder
from src.compressproj.models.google import JointAutoregressiveHierarchicalPriors
from src.compressproj.layers import CheckerboardConv, GDN, MaskedConv2d
from compressai.zoo import image_models





class CheckerboardAutoregressiveHierarchicalPriors(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        '''
        {s=1,p=1,d=2} --> z_hat:16x16
        Args:
            N:
            M:
            **kwargs:
        '''
        super(CheckerboardAutoregressiveHierarchicalPriors, self).__init__(N, M, **kwargs)
        print("CheckerboardAutoregressiveHierarchicalPriors Model ****************")
        # Vanilla with RF=5
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )
        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        # Vanilla with RF=2
        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=1, kernel_size=2),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=1, kernel_size=2),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )
        if False:
            self.h_a = nn.Sequential(
                conv(M, N, stride=1, kernel_size=3),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=1, kernel_size=2),  # RF = 3
            )

            self.h_s = nn.Sequential(
                deconv(N, M, stride=1, kernel_size=2),  # RF = 3
                nn.LeakyReLU(inplace=True),
                deconv(M, M * 3 // 2, stride=2, kernel_size=5),
                nn.LeakyReLU(inplace=True),
                conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
            )
        if True:
            # 1x1 conv forward
            print('Exp 2-1 Version')
            # z_hat: [16, 192, 17, 17]
            # params: [16, 384, 16, 16]
            self.h_a = nn.Sequential(
                conv(M, N, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                # conv(N, N, stride=1, kernel_size=2, padding=1, dilation=2),  # RF = 3
                conv(N, N, stride=1, kernel_size=2),  # RF = 3
            )

            self.h_s = nn.Sequential(
                # deconv(N, M, stride=1, kernel_size=2, output_padding=0, padding=1, dilation=2),  # RF = 3
                deconv(N, M, stride=1, kernel_size=2),  # RF = 3
                nn.LeakyReLU(inplace=True),
                deconv(M, M * 3 // 2, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(M * 3 // 2, M * 2, stride=1, kernel_size=1),
            )
            self.context_prediction = CheckerboardConv(M, 2 * M, kernel_size=3, padding=1, stride=1)
        if True:
            print('Exp 4-3 Checkerboard Version')
            self.h_a = nn.Sequential(
                conv(M, N, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                # conv(N, N, stride=1, kernel_size=2, padding=1, dilation=2),  # RF = 3
                conv(N, N, stride=1, kernel_size=2),  # RF = 3
            )
            self.h_s = nn.Sequential(
                # deconv(N, M, stride=1, kernel_size=2, output_padding=0, padding=1, dilation=2),  # RF = 3
                deconv(N, M, stride=1, kernel_size=2),  # RF = 3
                nn.LeakyReLU(inplace=True),
                deconv(M, M * 3 // 2, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(M * 3 // 2, M * 2, stride=1, kernel_size=1),
            )
            self.context_prediction = CheckerboardConv(M, 2 * M, kernel_size=1, stride=1)
        elif False:
            # 1x1 conv forward
            print('Exp 2-2 Version')
            self.h_a = nn.Sequential(
                conv(M, N, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=1, kernel_size=5),  # RF = 3
            )

            self.h_s = nn.Sequential(
                deconv(N, M, stride=1, kernel_size=5),  # RF = 3
                nn.LeakyReLU(inplace=True),
                deconv(M, M * 3 // 2, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(M * 3 // 2, M * 2, stride=1, kernel_size=1),
            )

        self.context_prediction = CheckerboardConv(M, 2 * M, kernel_size=3, padding=1, stride=1)
        self.context_prediction = CheckerboardConv(M, 2 * M, kernel_size=5, stride=1, padding=2) # Vanilla
        if exp_depthwise:
            print('Exp 4-4 Version (Depthwise Network)')
            self.h_a = nn.Sequential(
                conv(M, N, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(N, N, stride=1, kernel_size=2),  # RF = 3
            )
            self.h_s = nn.Sequential(
                depthwise_deconv(N, N, stride=1, kernel_size=2),  # RF = 3
                nn.LeakyReLU(inplace=True),
                deconv(N, M * 3 // 2, stride=1, kernel_size=1),
                nn.LeakyReLU(inplace=True),
                conv(M * 3 // 2, M * 2, stride=1, kernel_size=1),
            )
            self.context_prediction = CheckerboardConv(M, 2 * M, kernel_size=1, stride=1)

    def get_entropy_bottleneck(self):
        return self.entropy_bottleneck

    def forward(self, x):
        """
        anchor :
            0 1 0 1 0
            1 0 1 0 1
            0 1 0 1 0
            1 0 1 0 1
            0 1 0 1 0
        non-anchor (use anchor as context):
            1 0 1 0 1
            0 1 0 1 0
            1 0 1 0 1
            0 1 0 1 0
            1 0 1 0 1
        """
        B, C, H, W = x.shape

        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat) # (8, 384, 16, 16)
        # print(f'{y.shape}, {z_hat.shape}, {params.shape}')
        # print("forward z_hat[0]", z_hat[0])

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )

        anchor = torch.zeros_like(y_hat).to(x.device)
        non_anchor = torch.zeros_like(y_hat).to(x.device)

        anchor[:, :, 0::2, 1::2] = y_hat[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y_hat[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, 0::2] = y_hat[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, 1::2] = y_hat[:, :, 1::2, 1::2]

        # Compress Anchor
        ctx_params_anchor = torch.zeros([B, 2 * self.M, H // 16, W // 16]).to(x.device) # Vanilla
        # ctx_params_anchor = torch.zeros([B, 2 * self.M, H // 32, W // 32]).to(x.device)
        # print(ctx_params_anchor.shape, params.shape)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scale_anchor, mean_anchor = gaussian_params_anchor.chunk(2, 1)

        # Compress Non-Anchor
        ctx_params_non_anchor = self.context_prediction(anchor)
        if env.debug and env.debug_print:
            print(f'compare: {anchor.shape}\n{ctx_params_non_anchor.shape}, \n{params.shape}')

        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scale_non_anchor, mean_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_hat = torch.zeros([B, self.M, H // 16, W // 16]).to(x.device)
        means_hat = torch.zeros([B, self.M, H // 16, W // 16]).to(x.device)

        scales_hat[:, :, 0::2, 1::2] = scale_anchor[:, :, 0::2, 1::2]
        scales_hat[:, :, 1::2, 0::2] = scale_anchor[:, :, 1::2, 0::2]
        scales_hat[:, :, 0::2, 0::2] = scale_non_anchor[:, :, 0::2, 0::2]
        scales_hat[:, :, 1::2, 1::2] = scale_non_anchor[:, :, 1::2, 1::2]
        means_hat[:, :, 0::2, 1::2] = mean_anchor[:, :, 0::2, 1::2]
        means_hat[:, :, 1::2, 0::2] = mean_anchor[:, :, 1::2, 0::2]
        means_hat[:, :, 0::2, 0::2] = mean_non_anchor[:, :, 0::2, 0::2]
        means_hat[:, :, 1::2, 1::2] = mean_non_anchor[:, :, 1::2, 1::2]

        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # def load_state_dict(self, state_dict):
    #     update_registered_buffers(
    #         self.entropy_bottleneck,
    #         "entropy_bottleneck",
    #         ["_quantized_cdf", "_offset", "_cdf_length"],
    #         state_dict,
    #     )
    #     update_registered_buffers(
    #         self.global_entropy_bottleneck,
    #         "global_entropy_bottleneck",
    #         ["_quantized_cdf", "_offset", "_cdf_length"],
    #         state_dict,
    #     )
    #     super().load_state_dict(state_dict)

    # @classmethod
    # def from_state_dict(cls, state_dict):
    #     N = state_dict["g_a.0.weight"].size(0)
    #     M = state_dict["g_a.6.weight"].size(0)
    #     net = cls(N, M)
    #     net.load_state_dict(state_dict)
    #     return net

    def compress(self, x):
        """
        if y[i, :, j, k] == 0
        then bpp = 0
        Not recommend
        """

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        start_time = time.process_time()

        B, C, H, W = x.shape

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        start_log_time = time.process_time()
        params = self.h_s(z_hat)

        anchor = torch.zeros_like(y).to(x.device)
        non_anchor = torch.zeros_like(y).to(x.device)

        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        non_anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        non_anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]

        # Compress Anchor
        ctx_params_anchor = torch.zeros([B, self.M * 2, H // 16, W // 16]).to(x.device)
        # ctx_params_anchor = torch.zeros([B, self.M * 2, 34, 50]).to(x.device)
        # print(f'ctx_params:{ctx_params_anchor.shape}, params:{params.shape}, {H}, {W}')
        # ctx_params_anchor = torch.zeros_like(params, device=x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        anchor_strings = self.gaussian_conditional.compress(anchor, indexes_anchor, means_anchor)
        anchor_quantized = self.gaussian_conditional.decompress(anchor_strings, indexes_anchor, means=means_anchor)

        # Compress Non-Anchor
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        end_log_time = time.process_time()
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor)
        non_anchor_strings = self.gaussian_conditional.compress(non_anchor, indexes_non_anchor, means=means_non_anchor)

        end_time = time.process_time()
        cost_time = end_time - start_time
        tar_cost_time = end_log_time - start_log_time

        return {
            "strings": [anchor_strings, non_anchor_strings, z_strings],
            "shape": z.size()[-2:],
            "cost_time": cost_time,
            "tar_cost_time": tar_cost_time,
        }

    def decompress_to_lut(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[2], shape) # [64, 192, 9, 9]
        params = self.h_s(z_hat) # [64, 384, 16, 16]

        B, C, H, W = params.shape

        # decompress anchor
        ctx_params_anchor = torch.zeros([B, 2 * self.M, H, W]).to(z_hat.device) # [64, 384, 16, 16]
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        anchor_quantized = self.gaussian_conditional.decompress(strings[0], indexes_anchor, means=means_anchor)

        # decompress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor_quantized) # [64, 384, 16, 16]
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        ) # [64, 384, 16, 16]
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1) # [64, 192, 16, 16] * 2
        if env.debug and env.debug_print:
            print(f'ctx_params_non_anchor.shape: {ctx_params_non_anchor.shape}\n'
                  f'gaussian_params_non_anchor.shape: {gaussian_params_non_anchor.shape}\n'
                  f'scales_non_anchor.shape: {scales_non_anchor.shape}')

        return {
            'z_hat': z_hat,
            'scales': scales_non_anchor,
            'means': means_non_anchor,
        }

    def decompress(self, strings, shape):
        """
        if y[i, :, j, k] == 0
        then bpp = 0
        Not recommend
        """
        start_time = time.process_time()

        z_hat = self.entropy_bottleneck.decompress(strings[2], shape)

        # z_hat_tmp = self.entropy_bottleneck.decompress_lut(strings[2], shape)["outputs"]
        # B, C, H, W = z_hat_tmp.shape
        # z_hat_tmp = z_hat_tmp.reshape(-1).to(torch.device('cpu'))
        # L = B * C * H * W
        # print(B, C, H, W)
        # for i, tmp in enumerate(z_hat_tmp):
        #     channel = ( i // (H * W) ) % C
        #     if self.z_hat_dict[channel].get(tmp.item()) != None:
        #         self.z_hat_dict[channel][tmp.item()] += 1
        #     else:
        #         self.z_hat_dict[channel][tmp.item()] = 0

        start_log_time = time.process_time()
        params = self.h_s(z_hat)

        B, C, H, W = z_hat.shape

        # decompress anchor
        # ctx_params_anchor = torch.zeros([B, 2 * self.M, H * 4, W * 4]).to(z_hat.device)
        # ctx_params_anchor = torch.zeros([B, 2 * self.M, 16, 16]).to(z_hat.device) # Vanilla Cropped
        ctx_params_anchor = torch.zeros_like(params, device=z_hat.device) # Vanilla Full
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor)
        anchor_quantized = self.gaussian_conditional.decompress(strings[0], indexes_anchor, means=means_anchor)

        # decompress non-anchor
        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(scales_non_anchor)
        non_anchor_quantized = self.gaussian_conditional.decompress(strings[1], indexes_non_anchor,
                                                                    means=means_non_anchor)

        y_hat = anchor_quantized + non_anchor_quantized
        end_log_time = time.process_time()
        x_hat = self.g_s(y_hat)

        end_time = time.process_time()

        cost_time = end_time - start_time
        tar_cost_time = end_log_time - start_log_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time,
            "tar_cost_time": tar_cost_time,
        }

    def compress_slice_concatenate(self, x):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        B, C, H, W = x.shape

        y = self.g_a(x)

        y_a = y[:, :, 0::2, 0::2]
        y_d = y[:, :, 1::2, 1::2]
        y_b = y[:, :, 0::2, 1::2]
        y_c = y[:, :, 1::2, 0::2]

        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        params = self.h_s(z_hat)

        anchor = torch.zeros_like(y).to(x.device)
        anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]
        ctx_params_anchor = torch.zeros([B, self.M * 2, H // 16, W // 16]).to(x.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_strings = self.gaussian_conditional.compress(y_b, indexes_b, means_b)
        y_b_quantized = self.gaussian_conditional.decompress(y_b_strings, indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_strings = self.gaussian_conditional.compress(y_c, indexes_c, means_c)
        y_c_quantized = self.gaussian_conditional.decompress(y_c_strings, indexes_c, means=means_c)

        anchor_quantized = torch.zeros_like(y).to(x.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_strings = self.gaussian_conditional.compress(y_a, indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_strings = self.gaussian_conditional.compress(y_d, indexes_d, means=means_d)

        return {
            "strings": [y_a_strings, y_b_strings, y_c_strings, y_d_strings, z_strings],
            "shape": z.size()[-2:]
        }

    def decompress_slice_concatenate(self, strings, shape):
        """
        Slice y into four parts A, B, C and D. Encode and Decode them separately.
        Workflow is described in Figure 1 and Figure 2 in Supplementary Material.
        NA  A   ->  A B
        A  NA       C D
        After padding zero:
            anchor : 0 B     non-anchor: A 0
                     c 0                 0 D
        """
        start_time = time.process_time()

        z_hat = self.entropy_bottleneck.decompress(strings[4], shape)
        params = self.h_s(z_hat)

        B, C, H, W = z_hat.shape
        ctx_params_anchor = torch.zeros([B, self.M * 2, H * 4, W * 4]).to(z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(
            torch.cat([ctx_params_anchor, params], dim=1)
        )
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)

        scales_b = scales_anchor[:, :, 0::2, 1::2]
        means_b = means_anchor[:, :, 0::2, 1::2]
        indexes_b = self.gaussian_conditional.build_indexes(scales_b)
        y_b_quantized = self.gaussian_conditional.decompress(strings[1], indexes_b, means=means_b)

        scales_c = scales_anchor[:, :, 1::2, 0::2]
        means_c = means_anchor[:, :, 1::2, 0::2]
        indexes_c = self.gaussian_conditional.build_indexes(scales_c)
        y_c_quantized = self.gaussian_conditional.decompress(strings[2], indexes_c, means=means_c)

        anchor_quantized = torch.zeros([B, self.M, H * 4, W * 4]).to(z_hat.device)
        anchor_quantized[:, :, 0::2, 1::2] = y_b_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 0::2] = y_c_quantized[:, :, :, :]

        ctx_params_non_anchor = self.context_prediction(anchor_quantized)
        gaussian_params_non_anchor = self.entropy_parameters(
            torch.cat([ctx_params_non_anchor, params], dim=1)
        )
        scales_non_anchor, means_non_anchor = gaussian_params_non_anchor.chunk(2, 1)

        scales_a = scales_non_anchor[:, :, 0::2, 0::2]
        means_a = means_non_anchor[:, :, 0::2, 0::2]
        indexes_a = self.gaussian_conditional.build_indexes(scales_a)
        y_a_quantized = self.gaussian_conditional.decompress(strings[0], indexes_a, means=means_a)

        scales_d = scales_non_anchor[:, :, 1::2, 1::2]
        means_d = means_non_anchor[:, :, 1::2, 1::2]
        indexes_d = self.gaussian_conditional.build_indexes(scales_d)
        y_d_quantized = self.gaussian_conditional.decompress(strings[3], indexes_d, means=means_d)

        # Add non_anchor_quantized
        anchor_quantized[:, :, 0::2, 0::2] = y_a_quantized[:, :, :, :]
        anchor_quantized[:, :, 1::2, 1::2] = y_d_quantized[:, :, :, :]

        x_hat = self.g_s(anchor_quantized)

        end_time = time.process_time()

        cost_time = end_time - start_time

        return {
            "x_hat": x_hat,
            "cost_time": cost_time
        }

