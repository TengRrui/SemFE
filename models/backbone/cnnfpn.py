import math

import torch.nn as nn
import torch.nn.functional as F
import torch
from einops.einops import rearrange
import copy

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv1x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes, stride=1)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class CNNFPN(nn.Module):
    """
    ResNet+FPN, output resolution are 1/16 and 1/4.
    Each block has 2 layers.
    """

    def __init__(self, config):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        # Class Variable
        self.in_planes = initial_dim
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = self._make_blocks(block, initial_dim, block_dims[0], stride=1)  # 1/2
        self.stage2 = self._make_blocks(block, block_dims[0], block_dims[1], stride=2)  # 1/4
        self.stage3 = self._make_blocks(block, block_dims[1], block_dims[2], stride=2)  # 1/8
        self.stage4 = self._make_blocks(block, block_dims[2], block_dims[3], stride=2)  # 1/16

        # 3. FPN upsample
        self.layer4_outconv = conv1x1(block_dims[3], block_dims[3])
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[3])
        self.layer3_outconv2 = nn.Sequential(
            conv3x3(block_dims[3], block_dims[3]),
            nn.BatchNorm2d(block_dims[3]),
            nn.LeakyReLU(),
            conv3x3(block_dims[3], block_dims[2]),
        )

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )

        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_blocks(self, block, dim1, dim, stride=1):
            block1 = block(dim1, dim, stride=stride)
            block2 = block(dim, dim, stride=1)
            blocks = (block1, block2)
            return nn.Sequential(*blocks)

    def forward(self, x):

        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.stage1(x0)  # 1/2
        x2 = self.stage2(x1)  # 1/4
        x3 = self.stage3(x2)  # 1/8
        x4 = self.stage4(x3)  # 1/16

        # FPN
        x4_out = self.layer4_outconv(x4) # 1/16

        x4_out_2x = F.interpolate(x4_out, scale_factor=2., mode='bilinear', align_corners=True)  # 1/8
        x3_out = self.layer3_outconv(x3)  # 1/8
        x3_out = self.layer3_outconv2(x3_out+x4_out_2x)  # 1/8

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)  # 1/4
        x2_out = self.layer2_outconv(x2)  # 1/4
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)  # 1/4

        x1_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)  # 1/2
        x1_out = self.layer1_outconv(x1)  # 1/2
        x1_out = self.layer1_outconv2(x1_out + x1_out_2x)  # 1/2

        return [x4_out, x3_out, x2_out, x1_out]


"""def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

config = {
    'initial_dim': 128,  # 适当填写初始维度的值
    'block_dims': [128, 176, 256, 392]  # 适当填写每个 block 的维度列表
}

model = CNNFPN(config)
total_params = count_parameters(model)
print(f"Total trainable parameters in the model: {total_params}")
"""
