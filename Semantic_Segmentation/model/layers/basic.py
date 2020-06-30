import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

__all__ = ['_ConvBNReLU', '_DWConvBNReLU', 'InvertedResidual', '_ASPP', '_FCNHead',
           '_Hswish', '_ConvBNHswish', 'SEModule', 'Bottleneck', 'ShuffleNetUnit',
           'ShuffleNetV2Unit', 'InvertedIGCV3', 'MBConvBlock']


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.relu6 = relu6
        self.cbr = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                           padding, dilation, groups, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU6(True) if relu6 else nn.ReLU(False)
                                )

    def forward(self, x):
        x = self.cbr(x)
        return x
    
    def fuse_model(self):
        if self.relu6:
            torch.quantization.fuse_modules(self.cbr, ['0', '1'], inplace=True)
        else:
            torch.quantization.fuse_modules(self.cbr, ['0', '1','2'], inplace=True)            

class _ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBN, self).__init__()
        self.cb = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                           padding, dilation, groups, bias=False),
                                 nn.BatchNorm2d(out_channels)
                                )

    def forward(self, x):
        x = self.cb(x)
        return x
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self.cb, ['0', '1'], inplace=True)
        
class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, inter_channels, 3, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)

    def fuse_model(self):
        self.block[0].fuse_model()
        
# -----------------------------------------------------------------
#                      For MobileNet
# -----------------------------------------------------------------
class _DWConvBNReLU(nn.Module):
    """Depthwise Separable Convolution in MobileNet.
    depthwise convolution + pointwise convolution
    """

    def __init__(self, in_channels, dw_channels, out_channels, stride, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DWConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_channels, dw_channels, 3, stride, dilation, dilation, in_channels, norm_layer=norm_layer),
            _ConvBNReLU(dw_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)
    
    def fuse_model(self):
        self.conv[0].fuse_model()
        self.conv[1].fuse_model()

# -----------------------------------------------------------------
#                      For MobileNetV2
# -----------------------------------------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        self.expand_ratio = expand_ratio
        if expand_ratio != 1:
            # pw
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1, relu6=True, norm_layer=norm_layer))
        layers.extend([
            # dw
            _ConvBNReLU(inter_channels, inter_channels, 3, stride, dilation, dilation,
                        groups=inter_channels, relu6=True, norm_layer=norm_layer),
            # pw-linear
            _ConvBN(inter_channels, out_channels, 1)])
        self.conv = nn.Sequential(*layers)
        if self.use_res_connect:
            self.skip_add = nn.quantized.FloatFunctional()
            
    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)
        
    def fuse_model(self):
        self.conv[0].fuse_model()
        self.conv[1].fuse_model()
        if self.expand_ratio != 1:
            self.conv[2].fuse_model()

# -----------------------------------------------------------------
#                      For MobileNetV3
# -----------------------------------------------------------------

class _Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(_Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace)
        self.quant_add = nn.quantized.FloatFunctional()
        self.quant_mul = nn.quantized.FloatFunctional()
    def forward(self, x):
        out = self.quant_add.add_scalar(x, 3.0)
        out = self.relu6(out)
        out = self.quant_mul.mul_scalar(out,1/6)
        return out

class _Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(_Hswish, self).__init__()
        self.sigmoid = _Hsigmoid()
        self.quant_mul = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        out = self.sigmoid(x)
        out = self.quant_mul.mul(x,out)        
        return out

class _ConvBNHswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNHswish, self).__init__()
        self.cb = _ConvBN(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.act = _Hswish(True)

    def forward(self, x):
        x = self.cb(x)
        x = self.act(x)
        return x
    def fuse_model(self):
        self.cb.fuse_model()

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction, in_channels, bias=False),          
            _Hsigmoid(True)
        )
        self.quant_mul = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return self.quant_mul.mul(x, out.expand_as(x))
    def fuse_model(self):
        torch.quantization.fuse_modules(self.fc, ['0', '1'], inplace=True)

class Identity(nn.Module):
    def __init__(self, in_channels):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, dilation=1, se=False, nl='RE',
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            _ConvBNHswish(in_channels, exp_size, 1) if nl == 'HS' else _ConvBNReLU(in_channels, exp_size, 1), 
            # dw
            _ConvBNHswish(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation,
                      dilation, groups=exp_size) if nl == 'HS' else _ConvBNReLU(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation,
                      dilation, groups=exp_size),
            SELayer(exp_size),
            # pw-linear
            _ConvBN(exp_size, out_channels, 1)
        )
        self.se = se
        if self.use_res_connect:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self):
        self.conv[0].fuse_model()
        self.conv[1].fuse_model()
        if self.se:
            self.conv[2].fuse_model()
        self.conv[3].fuse_model()
        