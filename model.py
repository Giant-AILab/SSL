import torch
from models import Pose_Base,Pose_Base_Z
from utils import pose_transformation_mat
from models import Co_Speech

 

class GeneratorFullModel(torch.nn.Module):
    def __init__(self  ):
        super(GeneratorFullModel, self).__init__()

 
        
        gen_config = {
                    'image_channel' : 3, 
                    'block_expansion' :64 , 
                    'max_features' : 512, 
                    'num_down_blocks' : 3, 
                    }
        
        pose_config = {
                'block_expansion': 64, 
                'image_channel': 3, 
                'num_bins': 66
                      }

 
        self.pose_base = Pose_Base(**pose_config)
        # self.z_exp_m = Pose_Base_Z(block_expansion=64)
        self.generator = Co_Speech(**gen_config)
 

    def _source_forward(self, x ):             
        _ , v_s_no , encoder_map = self.generator._source_forward(x['source'])
        return v_s_no , encoder_map 

    
    def _driving_forward(self, x , v_s_no , encoder_map ):
        z_pose_dev = self.pose_base(x['driving']) 
        # z_fet_pose = self.z_exp_m(x['driving'])       

        dev_z = {}
        dev_z['z_pose_dev'] = z_pose_dev
        # he_driving['z_fet_pose'] = z_fet_pose
        # he_driving['z_sor_fet_pose'] = he_source['z_sor_fet_pose'].clone()

        out = self.generator._driving_forward(  v_s_no, dev_z=dev_z,encoder_map_init=encoder_map )
        
        return out
    
    def forward(self, xor_img,dev_img ):

        v_s_no , encoder_map = self._source_forward( {'source':xor_img} )
        out = self._driving_forward( {'driving':dev_img} ,v_s_no , encoder_map )
        return {"prediction":out}
           
 
    def _get_img_feat(self, x   ):
        bs = len(x)
        z_pose_dev = self.pose_base( x ) 

        # aff_z_pose = self.aff_z_pose( torch.cat([ z_sor_fet_pose,z_fet_pose ],-1 ))
        z_feat = pose_transformation_mat( z_pose_dev ).reshape(bs,-1) ##  16 16 9 
        t_feat = z_pose_dev['t_down8'].reshape(bs,3) 
        z_feat = torch.cat([z_feat , t_feat] , -1)
        return z_feat
 

    def _driving_gs (self,   v_s_no , encoder_map , z_pose_dev:torch.Tensor ):

        bs = len(z_pose_dev)
        z_p_dict = {}
        t_feat = z_pose_dev[:,-3:].unsqueeze(1)
        z_feat = z_pose_dev[:,:-3].reshape( bs,16,16,3,3 )
        z_p_dict['rot_mat_driving_down8'] = z_feat
        z_p_dict['t_down8_driving'] = t_feat
        out = self.generator._driving_gs(  v_s_no,  encoder_map_init=encoder_map,z_p_dict=z_p_dict )

        return out

