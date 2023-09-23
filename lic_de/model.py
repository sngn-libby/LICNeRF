from src.compressproj.models.google import JointAutoregressiveHierarchicalPriors
from src.compressproj.models.utils import conv, deconv
from src.compressproj.registry import register_model


@register_model("mbt_de")
class DE_Minnen2018(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N, M, **kwargs)

    def _check_g_in_channel(self):
        if "in_ch" in self:
            return self.in_ch
        return 3

    def _set_g_in_channel(self, ch):
        self.g_a[0] = conv(ch, self.N, kernel_size=5, stride=2)
        self.g_s[-1] = deconv(self.N, ch, kernel_size=5, stride=2)
        self.in_ch = ch
