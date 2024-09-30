from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from utils import ResBottleneck


class Pose_Base(nn.Module):
    def __init__(self, block_expansion, 
                 image_channel, 
                 num_bins=66
                 ):
        super(Pose_Base, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=image_channel, out_channels=block_expansion, kernel_size=7, padding=3, stride=2)
        self.norm1 = BatchNorm2d(block_expansion, affine=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=block_expansion, out_channels=256, kernel_size=1)
        self.norm2 = BatchNorm2d(256, affine=True)

        self.block1 = nn.Sequential()
        for i in range(3):
            self.block1.add_module('b1_'+ str(i), ResBottleneck(in_features=256, stride=1))

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.norm3 = BatchNorm2d(512, affine=True)
        self.block2 = ResBottleneck(in_features=512, stride=2)



        self.block3 = nn.Sequential()
        for i in range(3):
            self.block3.add_module('b3_'+ str(i), ResBottleneck(in_features=512, stride=1))

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.norm4 = BatchNorm2d(1024, affine=True)
        self.block4 = ResBottleneck(in_features=1024, stride=2)

        self.block5 = nn.Sequential()
        for i in range(5):
            self.block5.add_module('b5_'+ str(i), ResBottleneck(in_features=1024, stride=1))

        self.fc_roll_down8 = nn.Conv2d(1024, num_bins,1)
        self.fc_pitch_down8 = nn.Conv2d(1024, num_bins,1)
        self.fc_yaw_down8 = nn.Conv2d(1024, num_bins,1)
        self.fc_t_down8 = nn.Sequential(
                nn.Conv2d(1024, 3,1),
                nn.Tanh()
                )


    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        out = self.block1(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = F.relu(out)
        out = self.block2(out)





        out = self.block3(out)

        out = self.conv4(out)
        out = self.norm4(out)
        out = F.relu(out)
        out = self.block4(out)

        out = self.block5(out)
        down_8feat = out.clone()
        

        yaw_down8 = self.fc_roll_down8(down_8feat)
        pitch_down8 = self.fc_pitch_down8(down_8feat)
        roll_down8 = self.fc_yaw_down8(down_8feat)
        t_down8 = self.fc_t_down8( F.adaptive_avg_pool2d( down_8feat , 1) )

        out_dict = {}
        out_dict['yaw_down8'] = yaw_down8
        out_dict['pitch_down8'] = pitch_down8
        out_dict['roll_down8'] = roll_down8
        out_dict['t_down8'] = t_down8.permute(0,2,3,1)

        return out_dict
    