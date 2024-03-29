from torch import nn

from src.compressproj.registry import register_model
from src.compressproj.layers import GDN
from .google import JointAutoregressiveHierarchicalPriors
from .utils import conv, deconv


__all__ = [
    "DE_Minnen2018",
    "DE_PE_Minnen2018",
    "NeRF_Minnen2018",
]


@register_model("mbt-de")
class DE_Minnen2018(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        print(f":: Model :: {self.__class__.__name__} (mbt-de)")

        self.g_a = nn.Sequential(
            conv(4, N, kernel_size=5, stride=2),
            #conv(3 + 1 + 2 * 10, N, kernel_size=5, stride=2), # pe augmented version
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )


@register_model("mbt-de-pe")
class DE_PE_Minnen2018(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        print(f":: Model :: {self.__class__.__name__} (mbt-de-pe)")

        self.g_a = nn.Sequential(
            conv(3 + 1 + 2 * 10, N, kernel_size=5, stride=2), # pe augmented version
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )


@register_model("mbt-nerf")
class NeRF_Minnen2018(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        print(f":: Model :: {self.__class__.__name__} (mbt-nerf)")

        self.g_a = nn.Sequential(
            conv(3 + 1, N, kernel_size=5, stride=2),  # pe augmented version
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )
