"""MobileNetV2 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.base import BaseModel
from model.layers.basic import _FCNHead, _ConvBNReLU, _Hsigmoid
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

__all__ = ['MobileNetV2Seg', 'get_mobilenet_v2_seg']


class _MobileNetV2Seg(BaseModel):
    def __init__(self, nclass, backbone, pretrained_base=False, 
                  dataset="city", width_multi = 1.0, **kwargs):
        super(_MobileNetV2Seg, self).__init__(nclass, backbone, pretrained_base, **kwargs)
        self.width_multi = width_multi
        in_channels = int(320//2*width_multi)
        inter_channels = int(24*width_multi)         
        self.head = _Head(nclass, in_channels,inter_channels, dataset = dataset, width_multi = width_multi,  **kwargs)
        self.quant = QuantStub()
        self.dequant1 = DeQuantStub()
        self.dequant2 = DeQuantStub()     
        
    def forward(self, x):
        x = self.quant(x)
        c1, c2, c3, c4 = self.base_forward(x)
        c1, c4 = self.head(c1, c4)   
        c1 = self.dequant1(c1)
        c4 = self.dequant2(c4)
        return c1, c4

    def fuse_model(self):
        self.pretrained.fuse_model()
        self.head.fuse_model()
        
class MobileNetV2Seg(nn.Module): 
    def __init__(self, nclass, backbone, pretrained_base=False, 
                 dataset ='city',  width_multi = 1.0, **kwargs):
        super(MobileNetV2Seg, self).__init__() 
        in_channels = int(320//2*width_multi)
        inter_channels = int(24*width_multi)                 
        self.quantized = _MobileNetV2Seg(nclass, backbone, pretrained_base, dataset,width_multi, **kwargs)
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

def get_mobilenet_v2_seg(args, pretrained=False, root='~/.torch/models',
                         pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    model = MobileNetV2Seg(args.classes, backbone='mobilenetv2',
                           pretrained_base=pretrained_base, dataset = args.dataset, 
                           crop_scale = args.crop_scale, **kwargs)
    return model

def get_mobilenet_v2_1_0_seg(args, pretrained=False, root='~/.torch/models',
                         pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    model = MobileNetV2Seg(args.classes, backbone='mobilenet_v2_1_0',
                           pretrained_base=pretrained_base, dataset = args.dataset, 
                           width_multi = 1.0, **kwargs)
    return model

def get_mobilenet_v2_0_35_seg(args, pretrained=False, root='~/.torch/models',
                         pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    model = MobileNetV2Seg(args.classes, backbone='mobilenet_v2_0_35',
                           pretrained_base=pretrained_base, dataset = args.dataset,
                           width_multi = 0.35, **kwargs)
    return model

def get_mobilenet_v2_0_5_seg(args, pretrained=False, root='~/.torch/models',
                         pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    model = MobileNetV2Seg(args.classes, backbone='mobilenet_v2_0_5',
                           pretrained_base=pretrained_base, dataset = args.dataset, 
                           width_multi = 0.5, **kwargs)
    return model

if __name__ == '__main__':
    model = get_mobilenet_v2_seg()