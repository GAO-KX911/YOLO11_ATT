import torch
import torch.nn as nn

class CA(nn.Module):
    def __init__(self, channels: int, reduction: int = 32):
        """
        reduction: 压缩比，论文常用32， 也可用16
        """
        super().__init__()

        assert channels > 0
        mip = max(8, channels // reduction)

        self.Conv1 = nn.Conv2d(channels, mip, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        b, c, h, w = x.shape

        #分方向池化（保留坐标信息）
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True)

        # 拼接x_w -> (b, c, w, 1)
        x_w_t = x_w.permute(0,1,3,2)
        y = torch.cat([x_h, x_w_t], dim=2)

        y = self.act(self.bn1(self.Conv1(y)))

        y_h, y_w = torch.split(y, [h,w], dim=2)

        y_w = y_w.permute(0,1,3,2)

        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))

        out = x * a_h * a_w
        return out



