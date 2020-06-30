"""MobileNet3 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.base import BaseModel
from model.layers.basic import _ConvBNReLU, _Hsigmoid
from model.layers.LRASPP import _Head

import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

from torch.quantization import QuantStub, DeQuantStub

__all__ = ['MobileNetV3Seg', 'get_mobilenet_v3_large_seg', 'get_mobilenet_v3_small_seg']


class _MobileNetV3Seg(BaseModel):
    def __init__(self, nclass, backbone, pretrained_base=False, 
                 dataset ='city', crop_scale = 1.0, **kwargs):
        super(_MobileNetV3Seg, self).__init__(nclass, backbone, pretrained_base, **kwargs)
        mode = backbone.split('_')[-1]
        in_channels = 960//2 if mode == 'large' else 576//2
        inter_channels = 40 if mode.startswith('large') else 24    
        self.head = _Head(nclass, in_channels, inter_channels, mode = mode, dataset = dataset, **kwargs)
        self.quant = QuantStub()
        self.dequant1 = DeQuantStub()
        self.dequant2 = DeQuantStub()     
    def forward(self, x):
        x = self.quant(x)
        c1, _, _, c4 = self.base_forward(x)
        c1, c4 = self.head(c1, c4)   
        c1 = self.dequant1(c1)
        c4 = self.dequant2(c4)
        return c1, c4

    def fuse_model(self):
        self.pretrained.fuse_model()
        self.head.fuse_model()
        
class MobileNetV3Seg(nn.Module): 
    def __init__(self, nclass, backbone, pretrained_base=False, 
                 dataset ='city', crop_scale = 1.0, **kwargs):
        super(MobileNetV3Seg, self).__init__() 
        mode = backbone.split('_')[-1]
        in_channels = 960//2 if mode == 'large' else 576//2
        inter_channels = 40 if mode.startswith('large') else 24                 
        self.quantized = _MobileNetV3Seg(nclass, backbone, pretrained_base, dataset, **kwargs)
        self.project = nn.Conv2d(256//2, nclass, 1, 1)
        self.auxlayer = nn.Conv2d(inter_channels, nclass, 1, 1) 
        
    def forward(self, x):
        size = x.size()[2:]
        c1, c4 = self.quantized(x)
        c4 = self.project(c4)
        c1 = self.auxlayer(c1)
        out = torch.add(c1,c4)  
        x = F.interpolate(out, size, mode='bilinear', align_corners=True)    
        return x
    
def get_mobilenet_v3_large_seg(args, pretrained=False, root='~/.torch/models',
                               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    
    model = MobileNetV3Seg(args.classes, backbone='mobilenetv3_large',
                           pretrained_base=pretrained_base, dataset = args.dataset, **kwargs)
    return model


def get_mobilenet_v3_small_seg(args, pretrained=False, root='~/.torch/models',
                               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    model = MobileNetV3Seg(args.classes, backbone='mobilenetv3_small',
                           pretrained_base=pretrained_base, dataset = args.dataset, **kwargs)
    return model

def get_mobilenet_v3_RE_large_seg(args, pretrained=False, root='~/.torch/models',
                               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    
    model = MobileNetV3Seg(args.classes, backbone='mobilenetv3_RE_large',
                           pretrained_base=pretrained_base, dataset = args.dataset, **kwargs)
    return model


def get_mobilenet_v3_RE_small_seg(args, pretrained=False, root='~/.torch/models',
                               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    model = MobileNetV3Seg(args.classes, backbone='mobilenetv3_RE_small',
                           pretrained_base=pretrained_base, dataset = args.dataset, **kwargs)
    return model

if __name__ == '__main__':
    model = get_mobilenet_v3_small_seg()