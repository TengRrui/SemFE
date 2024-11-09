import torch.nn as nn
import torch.nn.functional as F
from models.position_encoding import PositionEncodingSine
from models.loftr_module.transformer import LocalFeatureTransformer
from models.FTM import FTM_module
from einops.einops import rearrange, repeat

class SFE_module(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Config
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        self.FTM_module_1 = FTM_module()
        self.FTM_module_2 = FTM_module()
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        # Class Variable

    def forward(self, x1, x2):
        data = {}
        data['h0_c_8'] = x1.size(2)
        data['h1_c_8'] = x2.size(2)
        data['w0_c_8'] = x1.size(3)
        data['w1_c_8'] = x2.size(3)
        feat_c0_8_1 = rearrange(self.pos_encoding(x1), 'n c h w -> n (h w) c')
        feat_c1_8_1 = rearrange(self.pos_encoding(x2), 'n c h w -> n (h w) c')
        mask_c0 = mask_c1 = None
        feat_c0_8_2, feat_c1_8_2 = self.loftr_coarse(feat_c0_8_1, feat_c1_8_1, mask_c0, mask_c1)
        feat_c0_8_2 = (feat_c0_8_2 + feat_c0_8_1) / 2
        feat_c1_8_2 = (feat_c1_8_2 + feat_c1_8_1) / 2
        feat_c0_8_2 = rearrange(feat_c0_8_2, 'n (h w) c -> n c h w', h=data['h0_c_8'], w=data['w0_c_8'])
        feat_c1_8_2 = rearrange(feat_c1_8_2, 'n (h w) c -> n c h w', h=data['h1_c_8'], w=data['w1_c_8'])
        Sem_K_query, classnum_query = self.FTM_module_1(feat_c0_8_2)
        Sem_K_refer, classnum_refer = self.FTM_module_2(feat_c1_8_2)

        return [feat_c0_8_2, feat_c1_8_2, Sem_K_query, classnum_query, Sem_K_refer, classnum_refer]
