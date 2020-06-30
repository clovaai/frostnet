import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.base import BaseModel
from model.layers.basic import _ConvBNReLU,_ConvBN, _Hsigmoid

class _Head(nn.Module):
    def __init__(self, nclass, in_channels, inter_channels, dataset='city', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()  
        self.lr_aspp = _LRASPP(in_channels, norm_layer, dataset, **kwargs)          
    def forward(self, c1, c4):
        c4 = self.lr_aspp(c4)
        c4 = F.interpolate(c4, c1.size()[2:], mode='bilinear', align_corners=True) 
        return c1, c4
    
    def fuse_model(self):
        self.lr_aspp.fuse_model()
        
class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, dataset, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 256//2
        self.b0 = _ConvBNReLU(in_channels, out_channels, 1,1)
        if dataset=='city':
            self.b1 = nn.Sequential(        
                    nn.AvgPool2d((37,37),(12,12)),            
                    _ConvBN(in_channels, out_channels, 1, 1, bias=False),
                    _Hsigmoid()
                    )             
        else:
            self.b1 = nn.Sequential(        
                    nn.AvgPool2d((25,25),(8,8)),            
                    _ConvBN(in_channels, out_channels, 1, 1, bias=False),
                    _Hsigmoid()
                    ) 
        self.quant_mul = nn.quantized.FloatFunctional() 
        
    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = self.quant_mul.mul(feat1, feat2)        
        return x
    def fuse_model(self):
        self.b0.fuse_model()
        self.b1[1].fuse_model()