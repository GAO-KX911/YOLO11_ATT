import torch
import torch.nn as nn


def _conv_bn_act(c1, c2, k=1, s=1, p=None):
    if p is None:
        p = k // 2
    return nn.Sequential(
        nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True),
    )


class AAGF(nn.Module):
    """Adaptive agreement-guided fusion for two aligned feature maps."""

    def __init__(self, c_low, c_high, c_out, gate_channels=16, fuse_kernel=3):
        super().__init__()
        self.low_proj = _conv_bn_act(c_low, c_out, k=1, p=0)
        self.high_proj = _conv_bn_act(c_high, c_out, k=1, p=0)

        gate_channels = max(8, int(gate_channels))
        self.gate_net = nn.Sequential(
            _conv_bn_act(c_out * 4, gate_channels, k=3),
            nn.Conv2d(gate_channels, c_out * 2, kernel_size=1, stride=1, padding=0),
        )
        self.out_conv = _conv_bn_act(c_out * 3, c_out, k=fuse_kernel)

    def forward(self, x):
        low, high = x
        low = self.low_proj(low)
        high = self.high_proj(high)

        diff = torch.abs(low - high)
        inter = low * high

        gates = torch.sigmoid(self.gate_net(torch.cat((low, high, diff, inter), dim=1)))
        gate_low, gate_high = gates.chunk(2, dim=1)

        low_refined = gate_low * low
        high_refined = gate_high * high
        fused = torch.cat((low_refined, high_refined, inter), dim=1)
        return self.out_conv(fused)
