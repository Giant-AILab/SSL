
import os
import cv2
import torch
import imageio
import numpy as np
from tqdm import tqdm
from model import GeneratorFullModel


def _load_cpk_model( _model_:torch.nn.Module , load_model_path ,model_key=None,rep_mm=None ):
    import os
    if model_key is not None:
        print ('  model_key  ',model_key)
    if os.path.isfile(load_model_path):
        cpk = torch.load(load_model_path,map_location='cpu' )
        if model_key is None:
            pass
        else:
            cpk = cpk[model_key]
        new_state_dict = {}
        for key in cpk:
            if key[:7] == 'module.':
                new_key = key[7:]
            else:
                new_key = key
            if rep_mm is None:
                pass
            else:
                new_key = new_key[len(rep_mm):]
            new_state_dict[new_key] = cpk[key]
        _model_.load_state_dict(new_state_dict)
        print ('模型载入成功 ')
    else:
        print ('model no: ',load_model_path)
        pass
    return _model_




def read_img_cv2(  path ,img_sizes=(256 , 256 )  ):
    '''
    loc_np x h x h
    '''
    img = cv2.imread( path )
    if img is None:
        return None
    img = img[:,:,::-1] 
    img = cv2.resize(img , ( img_sizes[1], img_sizes[0] ))
    img = img / 255.0
    return img
    
os.system('rm -rf 77777.mp4')
os.system('rm -rf 77777ss.mp4')

if __name__=='__main__':
 
    device = 'cuda'
    gen_model = GeneratorFullModel()
    _load_cpk_model(gen_model,
        './cpks/model.my2.pt',model_key='state_dict')
    gen_model = gen_model.to(device)
    gen_model.eval()
    
    so_img_path = './test_datas/seth#99853/0000000.png'
    root_path = os.path.dirname(so_img_path)
    de_img_paths = [os.path.join(root_path ,'{}.png'.format(_).zfill(11) ) for _ in range(500)][:]

    res = []
    so_img = read_img_cv2( so_img_path )
    so_ten = torch.FloatTensor(so_img).unsqueeze(0).to(device)
    so_ten = so_ten.permute(0,3,1,2)
    so_devs = []
    with torch.no_grad():
        x = {'source':so_ten}
        v_s_no , encoder_map_so = gen_model._source_forward(x)
        for dev_path in tqdm(de_img_paths):
            dev_img = read_img_cv2( dev_path )
            if dev_img is None:
                break
            so_devs.append( (dev_img * 255).astype(np.uint8) )
            dev_ten = torch.FloatTensor(dev_img).unsqueeze(0).to(device)
            dev_ten = dev_ten.permute(0,3,1,2)

            x_dev = {'driving':dev_ten}
            encoder_map = [oo.clone() for oo in encoder_map_so]
            out = gen_model._driving_forward(x_dev  , v_s_no , encoder_map )  

            out = out.permute(0,2,3,1)
            res.append( out.cpu().data.numpy() )
        torch.cuda.synchronize()
    res = (np.concatenate(res , 0) * 255).astype(np.uint8)


so_devs = np.stack(so_devs)  
res = np.concatenate([res,so_devs],2)
imageio.mimwrite( '77777.mp4', res  , fps=25 )


