# save this code in a file named `upsample_downsample.py`

import torch.nn as nn
from einops.einops import rearrange

class UpsampleDownsample(nn.Module):
    def __init__(self, config):
        super().__init__()
        initial_dim_16 = config['d_model_16']
        initial_dim_8 = config['d_model_8']
        initial_dim_4 = config['d_model_4']
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = nn.Conv2d(in_channels=initial_dim_16, out_channels=initial_dim_8, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(initial_dim_8)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=initial_dim_16, out_channels=initial_dim_8, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(initial_dim_8)
        self.relu1 = nn.ReLU(inplace=True)
        self.interpolate0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.interpolate1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(in_channels=initial_dim_4, out_channels=initial_dim_8, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(initial_dim_8)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=initial_dim_4, out_channels=initial_dim_8, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(initial_dim_8)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, feat_c0_16, feat_c1_16, feat_c0_4, feat_c1_4, feat_c0_8, feat_c1_8):

        feat_c0_16 = self.interpolate0(feat_c0_16)
        feat_c1_16 = self.interpolate1(feat_c1_16)
        feat_c0_16 = self.conv0(feat_c0_16)
        feat_c0_16 = self.bn0(feat_c0_16)
        feat_c0_16 = self.relu0(feat_c0_16)
        feat_c1_16 = self.conv1(feat_c1_16)
        feat_c1_16 = self.bn1(feat_c1_16)
        feat_c1_16 = self.relu1(feat_c1_16)
        feat_c0_4 = self.pool0(feat_c0_4)
        feat_c1_4= self.pool1(feat_c1_4)
        feat_c0_4 = self.conv2(feat_c0_4)
        feat_c0_4 = self.bn2(feat_c0_4)
        feat_c0_4 = self.relu2(feat_c0_4)
        feat_c1_4 = self.conv3(feat_c1_4)
        feat_c1_4 = self.bn3(feat_c1_4)
        feat_c1_4 = self.relu3(feat_c1_4)

        feat_c0_8_fused = feat_c0_8*0.5 + feat_c0_16*0.25 + feat_c0_4*0.25
        feat_c1_8_fused = feat_c1_8*0.5 + feat_c1_16*0.25 + feat_c1_4*0.25

        return feat_c0_8_fused, feat_c1_8_fused
