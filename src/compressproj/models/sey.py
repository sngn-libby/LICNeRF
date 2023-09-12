from torch import nn

from src.compressproj.registry import register_model
from src.compressproj.layers import GDN
from .google import JointAutoregressiveHierarchicalPriors
from .utils import conv, deconv


__all__ = [
    "DE_Minnen2018",
]


@register_model("mbt_de")
class DE_Minnen2018(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(6, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )


    # @property
    # def in_channel(self):
    #     if "in_ch" in self:
    #         return self.in_ch
    #     return 3
    #
    # def _set_g_in_channel(self, ch):
    #     self.g_a[0] = conv(ch, self.N, kernel_size=5, stride=2)
    #     self.g_s[-1] = deconv(self.N, ch, kernel_size=5, stride=2)
    #     self.in_ch = ch
