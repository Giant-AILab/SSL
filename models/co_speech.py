import torch
from torch import nn
import torch.nn.functional as F
from utils import make_coordinate_grid
from utils import DownBlock2d  , SameBlock2d
from utils import pose_transformation_mat
from models import Res_Feat
from models import GM_2D
from models import ResBlock2d_GN
from models import P_UP2
    


class Co_Speech(nn.Module):

    def __init__(self, 
                image_channel = 3, 
                block_expansion =64 , 
                max_features = 512, 
                num_down_blocks = 2, 
                 ):
        super(Co_Speech, self).__init__()
        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3),gp_nums=4)
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            if i == 0:
                out_ccc = in_features
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.Res_Feat = Res_Feat()
        self.decoder = GM_2D()
        self.res_ms = nn.Sequential(
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
            ResBlock2d_GN(512,3,1),
        )
        self.pose_up = P_UP2()

    def _source_forward(self, source_image ):
        encoder_map = []
        feat = self.first(source_image)
        encoder_map.append(feat)
        for i in range(len(self.down_blocks)):
            feat = self.down_blocks[i](feat)
            encoder_map.append(feat)
        feature_2d = feat 
        feature_2d = self.res_ms(feature_2d)
        v_s_no = self.Res_Feat( feature_2d )

        return v_s_no , v_s_no , encoder_map
 
    
    def _driving_forward(self,  v_s_no:torch.Tensor , dev_z , encoder_map_init:list   ):

        encoder_map = encoder_map_init
        bs, c, h, w = v_s_no.shape
        z_pose_dev = dev_z['z_pose_dev']
        identity_grid = make_coordinate_grid(( 512 , h, w ),v_s_no.dtype)
        identity_grid = identity_grid.unsqueeze(0).repeat( bs,1,1,1,1)
        identity_grid = identity_grid.to(v_s_no.device)

        w_s_id = identity_grid.clone()
        _,d_m,_,_,_ = identity_grid.shape
        grid_driving = identity_grid.clone()
        grid_driving = grid_driving.reshape(bs,d_m * h * w,3)
        w_d_pose = z_pose_dev

        rot_mat_driving_down8 = pose_transformation_mat( w_d_pose )  
        rot_mat_driving_down8 = self.pose_up.forward( rot_mat_driving_down8 )

        rot_mat_driving_down8 = rot_mat_driving_down8.unsqueeze(1).repeat( 1, d_m ,1,1,1,1)
        t_down8_driving = w_d_pose['t_down8'].reshape(bs,-1,3) 

        rot_mat_driving_down8 = rot_mat_driving_down8.reshape(bs,-1,3,3)
        grid_driving = grid_driving.unsqueeze(-1)
        grid_driving = torch.matmul(rot_mat_driving_down8, grid_driving)
        grid_driving = grid_driving.squeeze(-1) + t_down8_driving
        grid_driving = grid_driving.reshape(bs,  d_m, h, w , 3)
        w_s_d = w_s_id + grid_driving
        v_s_d = F.grid_sample( v_s_no.unsqueeze(1) , w_s_d ,align_corners=False)
        out = self.decoder( v_s_d.squeeze(1) ,encoder_map)
 
        return out
     


    
    def _driving_gs(self,  v_s_no:torch.Tensor , z_p_dict , encoder_map_init:list  ):

        encoder_map = encoder_map_init
        bs, c, h, w = v_s_no.shape

        rot_mat_driving_down8 = z_p_dict['rot_mat_driving_down8']
        t_down8_driving = z_p_dict['t_down8_driving']

 

        identity_grid = make_coordinate_grid(( 512 , h, w ),v_s_no.dtype)
        identity_grid = identity_grid.unsqueeze(0).repeat( bs,1,1,1,1)
        identity_grid = identity_grid.to(v_s_no.device)

        w_s_id = identity_grid.clone()
        _,d_m,_,_,_ = identity_grid.shape
        grid_driving = identity_grid.clone()
        grid_driving = grid_driving.reshape(bs,d_m * h * w,3)

        rot_mat_driving_down8 = self.pose_up.forward( rot_mat_driving_down8  )

        rot_mat_driving_down8 = rot_mat_driving_down8.unsqueeze(1).repeat( 1, d_m ,1,1,1,1)
        rot_mat_driving_down8 = rot_mat_driving_down8.reshape(bs,-1,3,3)
        grid_driving = grid_driving.unsqueeze(-1)
        grid_driving = torch.matmul(rot_mat_driving_down8, grid_driving)
        grid_driving = grid_driving.squeeze(-1) + t_down8_driving
        grid_driving = grid_driving.reshape(bs,  d_m, h, w , 3)

 
        w_s_d = w_s_id + grid_driving
        v_s_d = F.grid_sample( v_s_no.unsqueeze(1) , w_s_d ,align_corners=False)
        out = self.decoder( v_s_d.squeeze(1) ,encoder_map)
 
        return out
     

