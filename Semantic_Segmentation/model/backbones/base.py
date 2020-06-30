"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from model.backbones import mobilenet_v2_1_0,mobilenet_v2_0_35, mobilenet_v2_0_5, mobilenet_v3_small_1_0, mobilenet_v3_large_1_0, mobilenet_v3_RE_small_1_0, mobilenet_v3_RE_large_1_0


class BaseModel(nn.Module):
    def __init__(self, nclass, backbone='mobilenet', pretrained_base=False, **kwargs):
        super(BaseModel, self).__init__()
        self.nclass = nclass
        self.backbone = backbone
        if backbone == 'mobilenet_v2_1_0':
            self.pretrained = mobilenet_v2_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'mobilenet_v2_0_35':
            self.pretrained = mobilenet_v2_0_35(dilated=True, pretrained=pretrained_base,**kwargs) 
        elif backbone == 'mobilenet_v2_0_5':
            self.pretrained = mobilenet_v2_0_5(dilated=True, pretrained=pretrained_base, **kwargs)              
        elif backbone == 'mobilenetv3_small':
            self.pretrained = mobilenet_v3_small_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'mobilenetv3_large':
            self.pretrained = mobilenet_v3_large_1_0(dilated=True, pretrained=pretrained_base,**kwargs)
        elif backbone == 'mobilenetv3_RE_small':
            self.pretrained = mobilenet_v3_RE_small_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'mobilenetv3_RE_large':
            self.pretrained = mobilenet_v3_RE_large_1_0(dilated=True, pretrained=pretrained_base,**kwargs)            
        else:
            raise RuntimeError("Unknown backnone: {}".format(backbone))
     
    def base_forward(self, x):
        """forwarding pre-trained network"""
        if self.backbone in ['mobilenet_v2_1_0', 'mobilenet_v2_0_35','mobilenet_v2_0_5','mobilenetv3_small', 'mobilenetv3_large','mobilenetv3_RE_small', 'mobilenetv3_RE_large']:
            c1, c2, c3, c4 = self.pretrained(x)

        elif self.backbone in ['shufflenetv2']:
            x = self.pretrained.conv1(x)
            c1 = self.pretrained.maxpool(x)
            c2 = self.pretrained.stage2(c1)
            c3 = self.pretrained.stage3(c2)
            c4 = self.pretrained.stage4(c3)
        else:
            raise ValueError
        
        return c1, c2, c3, c4
