import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import GroupNorm



class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False,gp_nums=4):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        # self.norm = BatchNorm2d(out_features, affine=True)

        self.norm = GroupNorm( gp_nums,out_features )
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.ac(out)
        return out

class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        # self.norm = BatchNorm2d(out_features, affine=True)

        self.norm = GroupNorm( num_groups=8,num_channels=out_features)

        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out
    
class UpBlock2d_GN(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d_GN, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        # self.norm = BatchNorm3d(out_features, affine=True)

        num_groups = min(8, out_features // 16 )
        self.norm = GroupNorm(num_groups=num_groups,num_channels=out_features)

    def forward(self, x,scale_factor=(2, 2 )):
        # out = F.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear')
        out = F.interpolate(x, scale_factor=scale_factor)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out
