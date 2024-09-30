import torch
import torch.nn.functional as F


def pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    # pred = F.softmax(pred)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat



def pose_transformation_mat(  he ):

    yaw, pitch, roll = he['yaw_down8'], he['pitch_down8'], he['roll_down8']
    b,_,h,w = yaw.shape
    yaw = yaw.permute(0,2,3,1).reshape(b*h*w , -1)
    pitch = pitch.permute(0,2,3,1).reshape(b*h*w , -1)
    roll = roll.permute(0,2,3,1).reshape(b*h*w , -1)

    yaw = pred_to_degree(yaw)
    pitch = pred_to_degree(pitch)
    roll = pred_to_degree(roll)

    rot_mat = get_rotation_matrix(yaw, pitch, roll)   
    rot_mat = rot_mat.reshape( b,h,w,3,3 )

    return rot_mat

