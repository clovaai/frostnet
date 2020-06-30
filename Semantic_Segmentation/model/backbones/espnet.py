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
from torch.nn import functional as F

class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.cbr = CBR(nOut,nOut,1,1)        
        self.quant_cat = nn.quantized.FloatFunctional()
        self.quant_add2 = nn.quantized.FloatFunctional()
        self.quant_add3 = nn.quantized.FloatFunctional()
        self.quant_add4 = nn.quantized.FloatFunctional()
        
    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = self.quant_add2.add(add1, d4)
        add3 = self.quant_add3.add(add2, d8)
        add4 = self.quant_add4.add(add3, d16)

        combine = self.quant_cat.cat([d1, add1, add2, add3, add4],1)
        output = combine
        output = self.cbr(combine)
        return output
    
    def fuse_model(self):
        self.cbr.fuse_model()
        
class DilatedParllelResidualBlockB(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = C(nIn, n, 1, 1)
        self.d1 = CDilated(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16) # dilation rate of 2^4
        self.add = add
        if self.add:
            self.skip_add = nn.quantized.FloatFunctional()   
        self.cbr = CBR(nOut, nOut, 1, 1)     
        self.quant_cat = nn.quantized.FloatFunctional()
        self.quant_add2 = nn.quantized.FloatFunctional()
        self.quant_add3 = nn.quantized.FloatFunctional()
        self.quant_add4 = nn.quantized.FloatFunctional()
        
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # hierarchical fusion for de-gridding
        add1 = d2
        add2 = self.quant_add2.add(add1, d4)
        add3 = self.quant_add3.add(add2, d8)
        add4 = self.quant_add4.add(add3, d16)

        #merge
        combine = self.quant_cat.cat([d1, add1, add2, add3, add4], 1)
    
        # if residual version
        if self.add:
            combine = self.skip_add.add(input, combine)
        output = self.cbr(combine)
        
        return output
    
    def fuse_model(self):
        self.cbr.fuse_model()
        
class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, classes=20, p=5, q=3):
        '''
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        '''
        super().__init__()
        self.level1 = CBR(3, 16, 3, 2)
        self.sample1 = InputProjectionA(1)        
        self.sample2 = InputProjectionA(2)
        self.b1 = CBR(16 + 3,16 + 3, 1, 1)
        self.level2_0 = DownSamplerB(16 +3, 64)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(DilatedParllelResidualBlockB(64 , 64))
        self.b2 = CBR(128 + 3, 128+3, 1, 1)
        self.level3_0 = DownSamplerB(128 + 3, 128)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(DilatedParllelResidualBlockB(128 , 128))
        self.b3 = CBR(256, 256, 1, 1)
        self.classifier = C(256, classes, 1, 1)
        self.quant_cat1 = nn.quantized.FloatFunctional()
        self.quant_cat2 = nn.quantized.FloatFunctional()
        self.quant_cat3 = nn.quantized.FloatFunctional()
        
    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(self.quant_cat1.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(self.quant_cat2.cat([output1,  output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat) # down-sampled
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(self.quant_cat3.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        return classifier
    def fuse_model(self):
        self.level1.fuse_model()
        self.level2_0.fuse_model()
        for i, layer in enumerate(self.level2):
            layer.fuse_model()
        self.level3_0.fuse_model()
        for i, layer in enumerate(self.level3):
            layer.fuse_model()
        self.b1.fuse_model()
        self.b2.fuse_model()
        self.b3.fuse_model()