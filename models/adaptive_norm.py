import torch
from torch import nn
 
class AdaptiveNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-4):
        super(AdaptiveNorm2d, self).__init__()
        self.num_features = num_features
        self.norm_layer = nn.InstanceNorm3d(num_features, eps=eps, affine=False)
    def forward(self, input ,w_b:torch.Tensor):
        out = self.norm_layer(input)
        output = out * w_b[:, :self.num_features, None,None] + w_b[:, self.num_features:, None,None] 
        return output
