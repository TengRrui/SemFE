from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4
from .cnnfpn import CNNFPN
from .fpn import  FPN

def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_16_4(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    elif  config['backbone_type'] == 'CnnFpn':
        return CNNFPN(config['resnetfpn'])
    elif  config['backbone_type'] == 'fpn':
        return FPN(config['fpn'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
