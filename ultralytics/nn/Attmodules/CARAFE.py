import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):
    """Content-aware upsampling with channel-preserving input/output."""

    def __init__(self, channel, scale=2, kernel_size=5, encoder_kernel=3, compressed_channels=64):
        super().__init__()
        self.scale = scale
        self.kernel_size = kernel_size
        self.compressed_channels = min(channel, compressed_channels)

        self.comp = nn.Conv2d(channel, self.compressed_channels, kernel_size=1)
        self.enc = nn.Conv2d(
            self.compressed_channels,
            (self.scale * self.kernel_size) ** 2,
            kernel_size=encoder_kernel,
            padding=encoder_kernel // 2,
        )
        self.pix_shf = nn.PixelShuffle(self.scale)

    def forward(self, x):
        n, c, h, w = x.shape
        h_up, w_up = h * self.scale, w * self.scale

        kernel = self.comp(x)
        kernel = self.enc(kernel)
        kernel = self.pix_shf(kernel)
        kernel = F.softmax(kernel, dim=1)

        x_up = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        x_unfold = F.unfold(x_up, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        x_unfold = x_unfold.view(n, c, self.kernel_size * self.kernel_size, h_up, w_up)

        return torch.sum(x_unfold * kernel.unsqueeze(1), dim=2)
