
import torch
from torch import nn
from models.resblock import ResBlock2d_GN_v2,ResBlock2d_GN
from utils import UpBlock2d_GN
import torch.nn.functional as F
  
class GM_2D(nn.Module):
    def __init__( self ):
        
        super().__init__()

        self.conv_in = nn.Conv2d(512 , 512,1)
        self.res_ms = nn.Sequential(
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
        )

        self.up0 = nn.Sequential(
            UpBlock2d_GN(512,512),
            ResBlock2d_GN(512,3,1,num_groups=8)
        )

        self.up1 = nn.Sequential(
            UpBlock2d_GN(512,256),
            ResBlock2d_GN(256,3,1,num_groups=8)
        )



        self.up2 = nn.Sequential(
            UpBlock2d_GN(256,128),
            ResBlock2d_GN(128,3,1,num_groups=4)
        )
 

        self.conv_img = nn.Conv2d(128, 3, 3, padding=1)

 
        self.mask_up0 = nn.Conv2d( 512 , 512 ,3,1,1  )
        self.re_up0 = ResBlock2d_GN_v2(256,512,3,1,False,nom_type='GroupNorm')
        self.re_ac0 = nn.PReLU(512 , 0.14159)


        self.mask_up1 = nn.Conv2d( 256 , 256 ,3,1,1  )
        self.re_up1 = ResBlock2d_GN_v2(128,256,3,1,False,nom_type='GroupNorm')
        self.re_ac1 = nn.PReLU(256 , 0.14159)


    def forward(self, x:torch.Tensor,encoder_map  ):
        """
        pose : B C D H W
        """
        _cat = encoder_map.pop(-1)

        x = self.conv_in(x)
        x = self.res_ms(x)
        x = self.up0(x)

        _cat = encoder_map.pop(-1)
        x_mm = torch.sigmoid( self.mask_up0(x) )
        x = x * x_mm + (1- x_mm)*self.re_up0(_cat,None)
        x = self.re_ac0( x )

        x = self.up1(x)

        _cat = encoder_map.pop(-1)
        x_mm = torch.sigmoid( self.mask_up1(x) )
        x = x * x_mm + (1- x_mm)*self.re_up1(_cat,None)
        x = self.re_ac1( x )

        x = self.up2(x)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # # mask_0 = torch.sigmoid(x[:,3:])
        # x = x[:,:3] * torch.sigmoid(x[:,3:])
        x = torch.sigmoid(x)

        return x
