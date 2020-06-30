import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.base import BaseModel
from model.layers.basic import _ConvBNReLU,_ConvBN, _Hsigmoid

class _Head(nn.Module):
    def __init__(self, nclass, in_channels, inter_channels, dataset='city', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()  
        atrous_rates = (6, 12, 18)  
        self.aspp = ASPP(in_channels, atrous_rates, norm_layer, **kwargs)       
        self.auxlayer = _ConvBNReLU(inter_channels, 48, 1, 1)
        self.project = _ConvBNReLU(304, 256, 3, 3)
        self.reduce_conv = nn.Conv2d(256, nclass, 1, 1)        
        self.quant_cat = nn.quantized.FloatFunctional()   
        
    def forward(self, c1, c4):
        c4 = self.aspp(c4)
        c4 = F.interpolate(c4, c1.size()[2:], mode='bilinear', align_corners=True) 
        c1 = self.auxlayer(c1)
        out = self.quant_cat.cat([c1,c4],dim=1)    
        out = self.project(out)
        out = self.reduce_conv(out)
        return out
    
    def fuse_model(self):
        self.aspp.fuse_model()
        self.project.fuse_model()
        self.auxlayer.fuse_model()
        
class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvBNReLU(in_channels, out_channels, 1)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out
    
    def fuse_model(self):
        self.gap[1].fuse_model()

class _RASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_RASPP, self).__init__()
        out_channels = 256
        self.b0 = _ConvBNReLU(in_channels, out_channels, 1)


        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ConvBNReLU(in_channels, out_channels, 3, padding=rate1, dilation=rate1, norm_layer=norm_layer)
        self.b2 = _ConvBNReLU(in_channels, out_channels, 3, padding=rate2, dilation=rate2, norm_layer=norm_layer)
        self.b3 = _ConvBNReLU(in_channels, out_channels, 3, padding=rate3, dilation=rate3, norm_layer=norm_layer)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        self.project = nn.Sequential(
            _ConvBNReLU(5 * out_channels, out_channels, 1),
            nn.Dropout2d(0.1)
        )
        self.quant_cat = nn.quantized.FloatFunctional()
    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = self.quant_cat.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x
    def fuse_model(self):
        self.b0.fuse_model()
        self.b1.fuse_model()
        self.b2.fuse_model()
        self.b3.fuse_model()
        self.b4.fuse_model()
        self.project[0].fuse_model()
