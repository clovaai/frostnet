
import torch
import torch.nn as nn
import torch.quantization
import math
from torch.nn import functional as F

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

class hard_sigmoid(nn.Module):
    '''
    This class defines the ReLU6 activation to replace sigmoid
    '''

    def __init__(self):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=False)
        self.quant_mul = nn.quantized.FloatFunctional()
        self.quant_add = nn.quantized.FloatFunctional()        
    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = input
        output = self.quant_add.add_scalar(output, 3)
        output = self.relu6(output)
        output = self.quant_mul.mul_scalar(output, 1/6)

        return output
    
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 4, 8)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([C(features, features, 3, 1, groups=features) for size in sizes])
        self.project = CBR(features * (len(sizes) + 1), out_features, 1, 1)
        self.quant_cat = nn.quantized.FloatFunctional()
        
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        out = [feats]
        for stage in self.stages:
            feats = F.avg_pool2d(feats, kernel_size=3, stride=2, padding=1)
            upsampled = F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True)
            out.append(upsampled)
        return self.project(self.quant_cat.cat(out, dim=1))
    def fuse_model(self):
        self.project.fuse_model()
    
class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kernel_size, stride=1,  padding=0, groups=1, bias=False):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.cbr = nn.Sequential(nn.Conv2d(nIn, nOut, (kernel_size,kernel_size), stride=stride, 
                                           padding=(padding, padding),groups=groups, bias=False),
                                 nn.BatchNorm2d(nOut, eps=1e-03),
                                 nn.ReLU(inplace=False)
        )

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.cbr(input)

        return output

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cbr, ['0', '1','2'], inplace=True)



class CB(nn.Module):
    '''
       This class groups the convolution and batch normalization
    '''

    def __init__(self,nIn, nOut, kernel_size, stride=1,  padding=0, groups=1, bias=False):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optinal stide for down-sampling
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.cb = nn.Sequential(nn.Conv2d(nIn, nOut, (kernel_size,kernel_size), stride=stride, 
                                          padding=(padding, padding), groups=groups, bias=False),
                                nn.BatchNorm2d(nOut, eps=1e-03)
        )
    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.cb(input)
        return output

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cb, ['0','1'], inplace=True)

class C(nn.Module):
    '''
    This class is for a convolutional layer.
    '''

    def __init__(self,nIn, nOut, kernel_size, stride=1,  padding=0, groups=1, bias=False):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, (kernel_size, kernel_size), 
                              stride=stride, padding=(padding, padding),groups=groups, bias=False)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kernel_size, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kernel_size, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class CDilatedB(nn.Module):
    '''
    This class defines the dilated convolution with batch normalization.
    '''

    def __init__(self, nIn, nOut, kernel_size, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d
        self.cb = nn.Sequential(
                        nn.Conv2d(nIn, nOut,kernel_size, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups),
                        nn.BatchNorm2d(nOut)
                        )

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.cb(input)
        return output

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cb, ['0','1'], inplace=True)

class CDilatedBR(nn.Module):
    '''
    This class defines the dilated convolution with batch normalization.
    '''

    def __init__(self, nIn, nOut, kernel_size, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kernel_size: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kernel_size - 1) / 2) * d
        self.cbr = nn.Sequential(
                        nn.Conv2d(nIn, nOut,kernel_size, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups),
                        nn.BatchNorm2d(nOut),
                        nn.ReLU(inplace=False))

    def forward(self, input):
        '''

        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.cbr(input)
        return output

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cbr, ['0','1','2'], inplace=True)

class Shuffle(nn.Module):
    '''
    This class implements Channel Shuffling
    '''
    def __init__(self, groups):
        '''
        :param groups: # of groups for shuffling
        '''
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x       
    
class DWSepConv(nn.Module):
    def __init__(self, nin, nout):
        super(DWSepConv, self).__init__()   
        self.dwc = CBR(nin, nin, kernel_size=3, stride=1, groups=nin)
        self.pwc = CBR(nin, nout, kernel_size=1, stride=1, groups=1)
    def forward(self, x):         
        x = self.dwc(x)
        x = self.pwc(x)
        return x

    def fuse_model(self):  
        self.dwc.fuse_model()
        self.pwc.fuse_model()  

class EfficientPyrPool(nn.Module):
    """Efficient Pyramid Pooling Module"""

    def __init__(self, in_planes, proj_planes, out_planes, scales=[2.0, 1.5, 1.0, 0.5, 0.1], last_layer_br=True):
        super(EfficientPyrPool, self).__init__()
        self.stages = nn.ModuleList()
        scales.sort(reverse=True)

        self.projection_layer = CBR(in_planes, proj_planes, 1, 1)
        for _ in enumerate(scales):
            self.stages.append(C(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes))

        self.merge_layer = nn.Sequential(
            CBR(proj_planes * len(scales),proj_planes * len(scales),1,1),            
            Shuffle(groups=len(scales)),
            CBR(proj_planes * len(scales), proj_planes, 3, 1, groups=proj_planes)        
        )
        if last_layer_br:
            self.br = CBR(proj_planes, out_planes, kernel_size=1, stride=1, bias=not last_layer_br)
        else:
            self.br = nn.Conv2d(proj_planes, out_planes, kernel_size=1, stride=1, bias=not last_layer_br)
            
        self.last_layer_br = last_layer_br
        self.scales = scales
        self.quant_cat = nn.quantized.FloatFunctional()
    def forward(self, x):
        hs = []
        x = self.projection_layer(x)
        height, width = x.size()[2:]
        for i, stage in enumerate(self.stages):
            h_s = int(math.ceil(height * self.scales[i]))
            w_s = int(math.ceil(width * self.scales[i]))
            h_s = h_s if h_s > 5 else 5
            w_s = w_s if w_s > 5 else 5
            if self.scales[i] < 1.0:
                h = F.adaptive_avg_pool2d(x, output_size=(h_s, w_s))
                h = stage(h)
                h = F.interpolate(h, (height, width), mode='bilinear', align_corners=True)
            elif self.scales[i] > 1.0:
                h = F.interpolate(x, (h_s, w_s), mode='bilinear', align_corners=True)
                h = stage(h)
                h = F.adaptive_avg_pool2d(h, output_size=(height, width))
            else:
                h = stage(x)
            hs.append(h)

        out = self.quant_cat.cat(hs, dim=1)
        out = self.merge_layer(out)
        return self.br(out)
    
    def fuse_model(self):
        self.projection_layer.fuse_model() 
        self.merge_layer[0].fuse_model()
        self.merge_layer[2].fuse_model()
        if self.last_layer_br:
            self.br.fuse_model()
                