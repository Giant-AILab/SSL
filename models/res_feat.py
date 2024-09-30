
import torch
from torch import nn
from models.resblock import ResBlock2d_GN_v2

  
class Res_Feat(nn.Module):
    def __init__( self ):
        
        super().__init__()
        self.down_0 = ResBlock2d_GN_v2(512,768,3,1,True,nom_type='GroupNorm',gp_nums=8,scale_factor=( 0.5,0.5))
        self.down_1 = ResBlock2d_GN_v2(768,1024,3,1,True,nom_type='GroupNorm',gp_nums=16,scale_factor=( 0.5,0.5))
        self.mid_0 = ResBlock2d_GN_v2(1024,1024,3,1,False,nom_type='GroupNorm',gp_nums=16,scale_factor=( 0.5,0.5))
        self.up_1 = ResBlock2d_GN_v2( 1024+1024 ,1024,3,1,True,nom_type='GroupNorm',gp_nums=16,scale_factor=( 2, 2))
        self.up_0 = ResBlock2d_GN_v2(1024+768,768,3,1,True,nom_type='GroupNorm',gp_nums=8,scale_factor=( 2,2))

 
        self.out_o = nn.Sequential(
            nn.GroupNorm(num_groups=12 ,num_channels=768 ),
            nn.ReLU(),
            nn.Conv2d( 768 ,512,3,1,1),
        )
  

    def forward(self, x:torch.Tensor  ):
        """
        pose : B C D H W
        """
        # x = self.conv_in(x)
        res_0 = self.down_0(x,None)
        res_1 = self.down_1( res_0,None)
        mid_0 = self.mid_0( res_1,None)
        # x = self.mid_1(mid_0,None)
        x = torch.cat([mid_0 , res_1 ],1)
        # x = torch.cat([x ,self.mid_1(x,None)],1)
        x = self.up_1.forward(x,None)
        x = torch.cat([x ,res_0],1)
        x = self.up_0.forward(x,None)

        # x = torch.cat([x ,res_0],1)
        # x = self.res_o(x)
        x = self.out_o(x)


        return x
  