import math
import torch
import torch.nn as nn

def autopad(k, p = None, d = 1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k ,int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k ,int) else [x // 2 for x in k]

    return p

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k = 1, s = 1, p = None, g = 1, d = 1, act = True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k ,p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class RFEMBranch(nn.Module):
    def __init__(self, c1, c_, dilation = 1):
        super().__init__()
        self.cv1 = ConvBNAct(c1, c_, k = 1, s = 1)
        self.cv2 = ConvBNAct(c_, c_, k = 3, s = 1 , d = dilation)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        return x
    

class RFEM(nn.Module):
    def __init__(self, c1, c2, dilations=(3 , 5, 7), e = 0.5, shortcut = True, act = True):
        super().__init__()
        assert len(dilations) >= 1
        self.shortcut = shortcut and (c1 == c2)
        self.post_act = nn.SiLU(inplace=True) if act else nn.Identity()

        self.cv_short = ConvBNAct(c1, c2, k=1, s=1, act = False)

        hidden_total = max(1, int(c2*e))
        branch_num = len(dilations)
        branch_c = math.ceil(hidden_total / branch_num)

        self.branches = nn.ModuleList(
            [
                RFEMBranch(c1, branch_c, dilation=d) for d in dilations
            ]
        )

        self.cv_fuse = ConvBNAct(branch_c * branch_num, c2, k = 1, s = 1, act = False)
        self.cv_identity = ConvBNAct(c1, c2, k = 1, s = 1, act = False) \
            if(c1 != c2 and not self.shortcut) else nn.Identity()


    
    def forward(self, x):
        identity = self.cv_short(x) if self.shortcut else self.cv_identity(x)

        ys = [branch(x) for branch in self.branches]
        y = torch.cat(ys, dim=1)
        y = self.cv_fuse(y)
        y = y + identity
        y = self.post_act(y)
        return y
