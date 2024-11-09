import torch
import torch.nn as nn
from einops.einops import rearrange
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(out+residual)
# 定义模型

class FTM_module(nn.Module):
    def __init__(self):
        super(FTM_module, self).__init__()
        initial_dim = 256
        out_initial = 256
        self.conv1 = nn.Conv2d(initial_dim, out_initial, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_initial)
        self.relu = nn.ReLU(inplace=True)
        self.stage1 = self.make_layer(out_initial, out_initial, num_blocks=1, stride=2)  # 40
        self.stage2 = self.make_layer(out_initial, out_initial, num_blocks=1, stride=2)  # 20
        self.stage3 = self.make_layer(out_initial, out_initial, num_blocks=1, stride=2)  # 10

    def forward(self,x):

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.stage1(output)
        output = self.stage2(output)
        output = self.stage3(output)
        classnum = output.shape[2]*output.shape[3]
        output = rearrange(output, 'n c h w -> n c (h w)')
        output = output.transpose(1, 2)
        return output, classnum

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)


