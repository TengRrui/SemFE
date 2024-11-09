import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=False, normtype=False):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),#to_2tuple(k) =  (k,k) ; to_ntuple：n=几 复制几次并生成一个元组
                              to_2tuple(1), #stride=1
                              to_2tuple(k // 2), #padding = 3//2=1
                              groups=dim)
        self.normtype = normtype
        if self.normtype == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.normtype == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        if self.normtype == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.normtype == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))#[B,C,H,W]->[B,C,N]->[B,N,C]
        else:#flatten(n)：从第n个维度展开成一维
            feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)
        return x

class ChannelAttention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]#[B,h,N,C//h]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v#[B,h,C//h,N] @ #[B,h,N,C//h] ->[B,h,C//h,C//h]
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)#[B,h,C//h,C//h] @ [B,h,C//h,N]->[B,h,C//h,N]->[B,h,N,C//h]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, cpe_act=False):
        super().__init__()

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3, act=cpe_act),
                                  ConvPosEnc(dim=dim, k=3, act=cpe_act)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size

class ChannelTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.dim = 256
        self.num_heads = 8
        self.stages = 8

        self.channel_blocks0 = nn.ModuleList([
            ChannelBlock(dim=self.dim,
                         num_heads=self.num_heads,
                         drop_path=0.)
                         for _ in range(self.stages)
        ])

        self.channel_blocks1 = nn.ModuleList([
            ChannelBlock(dim=self.dim,
                         num_heads=self.num_heads,
                         drop_path=0.)
                         for _ in range(self.stages)
        ])

    def forward(self, feat0, feat1):
        
        # feat0 feat1 是 [b,c,h,w]
        b,c,h,w = feat0.shape
        feat0_size = (h,w)
        feat1_size = (h,w)
        feat0 = feat0.flatten(2).transpose(1,2) #[b,n,c]
        feat1 = feat1.flatten(2).transpose(1,2) 
        for idx in range(self.stages):
            feat0,feat0_size = self.channel_blocks0[idx](feat0,feat0_size)
            feat1,feat1_size = self.channel_blocks1[idx](feat1,feat1_size)

        #channel_block 要求输入是 [b,n,c]

        return feat0,feat1  # 返回的是[1,1600,256]==[b,n,c]
