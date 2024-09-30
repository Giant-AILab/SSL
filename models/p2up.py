
import torch
from torch import nn
from models.resblock import ResBlock2d_GN_v2


class P_UP2(nn.Module):
    def __init__( self ,add_mask=False ):
        
        super().__init__()
        self.add_mask = add_mask
        self.conv_0 = nn.Conv2d( 9  ,64,kernel_size=1 )

        # self.conv_0 = nn.Conv2d( 8  ,64,kernel_size=1 )
        # # in_features,out_channels , kernel_size, padding , upsample_down ,nom_type ,gp_nums=4
        self.down_0 = ResBlock2d_GN_v2(64,128,3,1,True,nom_type='GroupNorm',gp_nums=4,scale_factor=( 0.5,0.5))
        self.down_1 = ResBlock2d_GN_v2(128,256,3,1,True,nom_type='GroupNorm',gp_nums=8,scale_factor=( 0.5,0.5))
        self.mid_0 = ResBlock2d_GN_v2(256,512,3,1,False,nom_type='GroupNorm',gp_nums=8,scale_factor=( 0.5,0.5))

        self.res_mid = nn.ModuleList([ ResBlock2d_GN_v2(512,512,3,1,False,nom_type='GroupNorm',gp_nums=16,scale_factor=( 0.5,0.5)) 
                                      for _ in range(3)])
        
        self.up_1 = ResBlock2d_GN_v2( 512+256 ,384,3,1,True,nom_type='GroupNorm',gp_nums=16,scale_factor=( 2, 2))
        self.up_0 = ResBlock2d_GN_v2(384+128,256,3,1,True,nom_type='GroupNorm',gp_nums=8,scale_factor=( 2,2))
        self.up_out = ResBlock2d_GN_v2(256+64,128,3,1,True,nom_type='GroupNorm',gp_nums=8,scale_factor=( 2,2))
        self.out_0 = ResBlock2d_GN_v2(128,64,3,1,False,nom_type='GroupNorm',gp_nums=4,scale_factor=( 2,2))

        self.out_1 = nn.Sequential(
                    nn.Conv2d( 64 , 9 ,3,1,1),
                    nn.Tanh()
                )


    def forward(self,  pose:torch.Tensor   ):
        """
        cont_x: B C
        pose : B C D H W
        """
        cont_x = None
        b,h,w,p0,p1 = pose.shape
        pose = pose.permute(0,3,4,1,2) ##  
        pose = pose.reshape(b,p0*p1,h,w)
        p_x = self.conv_0(pose)
 
        res_0 = self.down_0.forward(p_x,cont_x)

        res_1 = self.down_1.forward(res_0,cont_x)

        mid_0 = self.mid_0.forward(res_1,cont_x)
        mid_r = mid_0.clone()
        for lay in self.res_mid:
            mid_r = lay.forward(mid_r,cont_x)

        x = torch.cat([mid_r ,res_1 ] , 1)
        x = self.up_1(x,cont_x)

        x = torch.cat([x ,res_0 ] , 1)
        x = self.up_0(x,cont_x)

        x = torch.cat([x ,p_x ] , 1)
        x = self.up_out(x,cont_x)

        x = self.out_0(x,cont_x)
        out = self.out_1(x)
        h,w = out.shape[-2:]
        out = out.reshape(b,p0,p1,h,w)
        out = out.permute(0,3,4,1,2) ##  

        return out

