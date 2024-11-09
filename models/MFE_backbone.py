import torch.nn as nn
import torch.nn.functional as F
from .backbone import build_backbone
from models.backbone import build_backbone
from models.position_encoding import PositionEncodingSine
from models.loftr_module.transformer import LocalFeatureTransformer
from models.upsample_downsample import UpsampleDownsample
from einops.einops import rearrange, repeat

class MFE_backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.backbone = build_backbone(config)
        self.start_coarse = LocalFeatureTransformer(config['start'])
        self.updown_sample = UpsampleDownsample(config['updown'])
        # Class Variable

    def forward(self, x1, x2):
        data = {}
        (feat_c0_16, feat_c0_8, feat_c0_4, feat_f0_2), (feat_c1_16, feat_c1_8, feat_c1_4, feat_f1_2) = self.backbone(
            x1), self.backbone(x2)
        data['hw0_c_16'] = feat_c0_16.shape[2:]
        data['hw1_c_16'] = feat_c1_16.shape[2:]
        data['hw0_c_8'] = feat_c0_8.shape[2:]
        data['hw1_c_8'] = feat_c1_8.shape[2:]
        data['hw0_c_4'] = feat_c0_4.shape[2:]
        data['hw1_c_4'] = feat_c1_4.shape[2:]
        data['hw0_f_2'] = feat_f0_2.shape[2:]
        data['hw1_f_2'] = feat_f1_2.shape[2:]
        data['h0_c_16'] = feat_c0_16.size(2)
        data['h1_c_16'] = feat_c1_16.size(2)
        data['h0_c_8'] = feat_c0_8.size(2)
        data['h1_c_8'] = feat_c1_8.size(2)
        data['h0_c_4'] = feat_c0_4.size(2)
        data['h1_c_4'] = feat_c1_4.size(2)
        data['h0_f_2'] = feat_f0_2.size(2)
        data['h1_f_2'] = feat_f1_2.size(2)
        data['w0_c_16'] = feat_c0_16.size(3)
        data['w1_c_16'] = feat_c1_16.size(3)
        data['w0_c_8'] = feat_c0_8.size(3)
        data['w1_c_8'] = feat_c1_8.size(3)
        data['w0_c_4'] = feat_c0_4.size(3)
        data['w1_c_4'] = feat_c1_4.size(3)
        data['w0_f_2'] = feat_f0_2.size(3)
        data['w1_f_2'] = feat_f1_2.size(3)
        feat_c0_8 = rearrange(self.pos_encoding(feat_c0_8), 'n c h w -> n (h w) c')
        feat_c1_8 = rearrange(self.pos_encoding(feat_c1_8), 'n c h w -> n (h w) c')
        feat_c0_8_01 = feat_c0_8
        feat_c1_8_01 = feat_c1_8
        feat_c0_8, feat_c1_8 = self.start_coarse(feat_c0_8, feat_c1_8)
        feat_c0_8_01 = (feat_c0_8_01 + feat_c0_8) / 2
        feat_c1_8_01 = (feat_c1_8_01 + feat_c1_8) / 2
        feat_c0_8 = rearrange(feat_c0_8_01, 'n (h w) c -> n c h w', h=data['h0_c_8'], w=data['w0_c_8'])
        feat_c1_8 = rearrange(feat_c1_8_01, 'n (h w) c -> n c h w', h=data['h1_c_8'], w=data['w1_c_8'])
        feat_c0_8_02 = feat_c0_8
        feat_c1_8_02 = feat_c1_8
        feat_c0_8, feat_c1_8 = self.updown_sample(feat_c0_16, feat_c1_16, feat_c0_4, feat_c1_4, feat_c0_8, feat_c1_8)
        feat_c0_8 = (feat_c0_8_02 + feat_c0_8) / 2
        feat_c1_8 = (feat_c1_8_02 + feat_c1_8) / 2

        return [feat_c0_8, feat_c1_8, feat_f0_2, feat_f1_2]
