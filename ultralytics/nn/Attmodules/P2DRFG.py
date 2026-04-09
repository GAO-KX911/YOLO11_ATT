import torch
import torch.nn as nn


def _conv_bn_act(c1, c2, k=1, s=1, p=None, g=1):
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True),
    )


class DWConvBlock(nn.Module):
    """Depthwise conv + pointwise conv."""

    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.block = nn.Sequential(
            _conv_bn_act(c1, c1, k=k, g=c1),
            _conv_bn_act(c1, c2, k=1, p=0),
        )

    def forward(self, x):
        return self.block(x)


class P2DRFG(nn.Module):
    """
    P2 Detail Residual Fusion Guidance.

    Inputs:
        low:  backbone P2 feature
        high: upsampled higher-level feature (e.g. CARAFE output from P3->P2)
        main: main fused trunk output after Concat + C3k2
    """

    def __init__(self, c_low, c_high, c_main, c_out, gate_channels=16, alpha_init=0.5):
        super().__init__()

        gate_channels = max(8, int(gate_channels))

        self.low_proj = _conv_bn_act(c_low, c_out, k=1, p=0)
        self.high_proj = _conv_bn_act(c_high, c_out, k=1, p=0)
        self.main_proj = _conv_bn_act(c_main, c_out, k=1, p=0) if c_main != c_out else nn.Identity()

        self.detail_3 = DWConvBlock(c_out, c_out, k=3)
        self.detail_5 = DWConvBlock(c_out, c_out, k=5)
        self.detail_fuse = _conv_bn_act(c_out * 2, c_out, k=1, p=0)

        self.inter_proj = _conv_bn_act(c_out, c_out, k=1, p=0)

        self.gate_net = nn.Sequential(
            _conv_bn_act(c_out * 4, gate_channels, k=3),
            nn.Conv2d(gate_channels, c_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

        self.detail_out = _conv_bn_act(c_out, c_out, k=3)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        low, high, main = x

        low = self.low_proj(low)
        high = self.high_proj(high)
        main = self.main_proj(main)

        d3 = self.detail_3(low)
        d5 = self.detail_5(low)
        detail = self.detail_fuse(torch.cat((d3, d5), dim=1))

        diff = torch.abs(low - high)
        inter = self.inter_proj(low * high)
        gate = self.gate_net(torch.cat((low, high, diff, inter), dim=1))

        detail_guided = self.detail_out(gate * detail)
        alpha = torch.sigmoid(self.alpha)
        return main + alpha * detail_guided
