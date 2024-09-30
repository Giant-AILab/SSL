from torch import nn
import torch.nn.functional as F
from torch.nn import GroupNorm
from models.adaptive_norm import AdaptiveNorm2d


class ResBlock2d_GN(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding,num_groups=None):
        super(ResBlock2d_GN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        if num_groups is None:
            num_groups = min(16, in_features// 32)
        self.norm1 = GroupNorm(num_groups=num_groups, num_channels=in_features)
        self.norm2 = GroupNorm(num_groups=num_groups, num_channels=in_features)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out
  
class ResBlock2d_GN_v2(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features,out_channels , kernel_size, padding , upsample_down ,nom_type ,gp_nums=4,scale_factor=1 ):
        super(ResBlock2d_GN_v2, self).__init__()

        if nom_type == 'AdaptiveNorm':
            
            self.norm1 = AdaptiveNorm2d( in_features)
            self.norm2 = AdaptiveNorm2d( out_channels )
        elif nom_type == 'GroupNorm':
            self.norm1 = GroupNorm(gp_nums, in_features)
            self.norm2 = GroupNorm(gp_nums, out_channels)
        else:
            print ('eeeeeeeeeeeeee' ,' nom_type ' ,nom_type)
        
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding)
        
        self.upsample_down = upsample_down

        self.scale_factor = scale_factor
        self.in_features = in_features
        self.out_channels = out_channels
        if self.upsample_down or ( self.in_features!=self.out_channels ):
            self.conv3 = nn.Conv2d(in_channels=in_features, out_channels=out_channels, kernel_size=1,
                                padding=0)
    def forward(self, x_inp ,w_b  ):

        scale_factor = self.scale_factor
        if w_b is None:
            out = self.norm1( x_inp )
        else:
            in_w_b = w_b[:,:self.in_features*2]
            w_b = w_b[:,self.in_features*2:]
            out = self.norm1( x_inp,in_w_b )

        out = F.relu(out)
        if self.upsample_down:
            out = F.interpolate(out, scale_factor=scale_factor)
        out = self.conv1(out)
        if w_b is None:
            out = self.norm2(out)
        else:
            out = self.norm2(out,w_b)
        out = F.relu(out)
        out = self.conv2(out)

        if self.upsample_down or ( self.in_features!=self.out_channels ) :
            if self.upsample_down:
                out = out + self.conv3( F.interpolate(x_inp, scale_factor=scale_factor) )
            else:
                out = out + self.conv3( x_inp )
        else:
            out += x_inp
        return out
  
