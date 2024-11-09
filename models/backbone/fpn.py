import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from einops.einops import rearrange
import copy
from models.loftr_module.linear_attention import LinearAttention, FullAttention
from models.position import PositionEmbedding2D, PositionEmbedding1D

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        if feat0.size(0) == 2:  # 如果feat0的第一个维度是2
            feat0_1, feat0_2 = feat0.chunk(2, dim=0)  # 将feat0分割成两个子张量
            # 对每个子张量进行操作
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0_1 = layer(feat0_1, feat0_1, mask0, mask0)
                    feat0_2 = layer(feat0_2, feat0_2, mask0, mask0)
                else:
                    raise KeyError
            # 将操作后的子张量合并
            feat0 = torch.cat([feat0_1, feat0_2], dim=0)

        else:  # 如果feat0的第一个维度不是2，则直接进行操作
            for layer, name in zip(self.layers, self.layer_names):
                if name == 'self':
                    feat0 = layer(feat0, feat0, mask0, mask0)
                else:
                    raise KeyError

        return feat0

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

class FPN(nn.Module):
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
        self.position1d = PositionEmbedding1D(d_model=392, max_len=400)

        self.stage1 = self._make_blocks(block, initial_dim, block_dims[0], stride=1)  # 1/2
        self.stage2 = self._make_blocks(block, block_dims[0], block_dims[1], stride=2)  # 1/4
        self.stage3 = self._make_blocks(block, block_dims[1], block_dims[2], stride=2)  # 1/8
        self.stage4 = self._make_blocks(block, block_dims[2], block_dims[3], stride=2)  # 1/16
        self.attention = LocalFeatureTransformer(config)
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
        data = []
        data = {'h0_c_16': x4_out.size(2), 'w0_c_16': x4_out.size(3)}
        x4_out = rearrange(x4_out, 'n c h w -> n (h w) c', h=data['h0_c_16'], w=data['w0_c_16'])
        # local_position = self.position1d(x4_out)
        # x4_out = x4_out + local_position
        x4_out = self.attention(x4_out)
        x4_out = rearrange(x4_out, 'n (h w) c -> n c h w', h=data['h0_c_16'], w=data['w0_c_16'])

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
