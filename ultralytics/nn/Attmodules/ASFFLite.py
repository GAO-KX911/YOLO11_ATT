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


class ASFFLite(nn.Module):
    """Lite two-branch adaptive spatial feature fusion for P2 enhancement."""

    def __init__(self, c_low, c_high, c_out, weight_channels=16, fuse_kernel=3):
        super().__init__()
        self.low_proj = _conv_bn_act(c_low, c_out, k=1, p=0)
        self.high_proj = _conv_bn_act(c_high, c_out, k=1, p=0)

        weight_channels = max(8, int(weight_channels))
        self.weight_net = nn.Sequential(
            _conv_bn_act(c_out * 2, weight_channels, k=3),
            nn.Conv2d(weight_channels, 2, kernel_size=1, stride=1, padding=0),
        )
        self.out_conv = _conv_bn_act(c_out, c_out, k=fuse_kernel)

    def forward(self, x):
        low, high = x
        low = self.low_proj(low)
        high = self.high_proj(high)

        weights = self.weight_net(torch.cat((low, high), dim=1)).softmax(dim=1)
        fused = weights[:, 0:1] * low + weights[:, 1:2] * high
        return self.out_conv(fused)
