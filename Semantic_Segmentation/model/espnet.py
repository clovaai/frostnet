import torch
import torch.nn as nn
import torch.quantization

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

from model.layers.espnet_utils import *
from model.backbones.espnet import *
from torch.nn import functional as F

class ESPNet(nn.Module):
    def __init__(self, classes=20, p=2, q=3):
        super().__init__()
        self.encoder = ESPNet_Encoder(classes, p, q)      
        # load the encoder modules

        # light-weight decoder
        self.level3_C = C(128 + 3, classes, 1, 1)
        self.b = CB(classes,classes,1,1)        
        self.conv = CBR(19 + classes, classes, 3, 1)
        self.up_l3 = CBR(classes, classes, 1, stride=1, padding=0, bias=False)
        
        self.combine_l2_l3 = DilatedParllelResidualBlockB(2*classes , classes, add=False)
        self.up_l2 = CBR(classes, classes, 1, stride=1, padding=0, bias=False)

        #self.classifier = C(classes, classes, 1, stride=1, padding=0)
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.quant_cat1 = nn.quantized.FloatFunctional()
        self.quant_cat2 = nn.quantized.FloatFunctional()
        self.quant_cat3 = nn.quantized.FloatFunctional()
        self.quant_cat4 = nn.quantized.FloatFunctional()
        self.quant_cat5 = nn.quantized.FloatFunctional()
        
    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    
    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        input = self.quant(input)
        
        output0 = self.encoder.level1(input)
        inp1 = self.encoder.sample1(input)
        inp2 = self.encoder.sample2(input)

        output0_cat = self.encoder.b1(self.quant_cat1.cat([output0, inp1], 1))
        output1_0 = self.encoder.level2_0(output0_cat)  # down-sampled
        for i, layer in enumerate(self.encoder.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.encoder.b2(self.quant_cat2.cat([output1, output1_0, inp2], 1))

        output2_0 = self.encoder.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.encoder.b3(self.quant_cat3.cat([output2_0, output2], 1))
        
        l3 = self.upsample(self.b(self.encoder.classifier(output2_cat)))
        output2_c = self.up_l3(l3) #RUM

        output1_C = self.level3_C(output1_cat) # project to C-dimensional space
        
        l2 = self.upsample(self.combine_l2_l3(self.quant_cat4.cat([output1_C, output2_c], 1)))
        comb_l2_l3 = self.up_l2(l2) #RUM

        concat_features = self.conv(self.quant_cat5.cat([comb_l2_l3, output0_cat], 1))
        concat_features = self.upsample(concat_features)
        
        classifier = concat_features
        classifier = self.dequant(classifier)
        
        return classifier
    
    def fuse_model(self):
        self.encoder.level1.fuse_model()
        self.encoder.level2_0.fuse_model()   
        for i, layer in enumerate(self.encoder.level2):
            layer.fuse_model()
        self.encoder.level3_0.fuse_model() 
        for i, layer in enumerate(self.encoder.level3):
            layer.fuse_model()               
        self.up_l3.fuse_model()             
        self.up_l2.fuse_model()   
        self.conv.fuse_model()
        self.combine_l2_l3.fuse_model()
        self.encoder.b1.fuse_model()
        self.encoder.b2.fuse_model()        
        self.encoder.b3.fuse_model()
        self.b.fuse_model()

class ESPNetSeg(nn.Module):
    def __init__(self, classes=20, p=2, q=3):
        super().__init__()
        self.quantized = ESPNet(classes,p,q)
        self.classifier = C(classes, classes, 1, stride=1, padding=0)        
    def forward(self, input):
        x = self.quantized(input)
        x = self.classifier(x)
        return x
        
def espnet_seg(args):
    classes = args.classes
    p = 2
    q = 8
    model = ESPNetSeg(classes,p,q)

    return model