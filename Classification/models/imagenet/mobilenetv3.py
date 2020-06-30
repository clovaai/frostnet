import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

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
        
class _Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(_Hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)
        self.quant_mul1 = nn.quantized.FloatFunctional()
        self.quant_mul2 = nn.quantized.FloatFunctional()
        self.quant_add = nn.quantized.FloatFunctional()
        
    def forward(self, x):
        out = self.quant_add.add_scalar(x, 3.0)
        out = self.relu6(out)
        out = self.quant_mul1.mul(x,out)
        out = self.quant_mul2.mul_scalar(out, 1/6)
        return out


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
        if nl == 'HS':
            act = _Hswish
        else:
            act = nn.ReLU
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            _ConvBNHswish(in_channels, exp_size, 1) if nl == 'HS' else _ConvBNReLU(in_channels, exp_size, 1), 
            # dw
            _ConvBN(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation,
                      dilation, groups=exp_size),
            SELayer(exp_size),
            act(True),
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
        self.conv[4].fuse_model()        

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=1000, mode='large', width_mult=1.0, dilated=False,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(MobileNetV3, self).__init__()

        if mode == 'large':
            layer1_setting = [
                    # k, exp_size, c, se, nl, s
                    [3, 16, 16, False, 'RE', 1],
                    [3, 64, 24, False, 'RE', 2],
                    [3, 72, 24, False, 'RE', 1],]
            layer2_setting = [
                    [5, 72, 40, True, 'RE', 2],
                    [5, 120, 40, True, 'RE', 1],
                    [5, 120, 40, True, 'RE', 1],]
            
            layer3_setting = [
                [3, 240, 80, False, 'HS', 2],                
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],]
            if dilated: #Reduce by Factor of 2
                layer4_setting = [
                    [5, 672, 160, True, 'HS', 2],
                    [5, 960, 160, True, 'HS', 1],
                    [5, 960//2, 160//2, True, 'HS', 1], ]
            else:
                layer4_setting = [
                    [5, 672, 160, True, 'HS', 2],
                    [5, 960, 160, True, 'HS', 1],
                    [5, 960, 160, True, 'HS', 1], ]                
            
        elif mode == 'small':
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, True, 'RE', 2], ]

            layer2_setting = [
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1], ]                
            layer3_setting = [
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1], ]
            if dilated:
                layer4_setting = [
                    [5, 288, 96, True, 'HS', 2],
                    [5, 576, 96, True, 'HS', 1],
                    [5, 576//2, 96//2, True, 'HS', 1], ]
            else:
                layer4_setting = [
                    [5, 288, 96, True, 'HS', 2],
                    [5, 576, 96, True, 'HS', 1],
                    [5, 576, 96, True, 'HS', 1], ]
        else:
            raise ValueError('Unknown mode.')

        # building first layer
        self.in_channels = int(16 * width_mult) if width_mult > 1.0 else 16
        self.conv1 = _ConvBNHswish(3, self.in_channels, 3, 2, 1, norm_layer=norm_layer)

        # building bottleneck blocks
        self.layer1 = self._make_layer(Bottleneck, layer1_setting,
                                       width_mult, norm_layer=norm_layer)
        self.layer2 = self._make_layer(Bottleneck, layer2_setting,
                                       width_mult, norm_layer=norm_layer)
        self.layer3 = self._make_layer(Bottleneck, layer3_setting,
                                       width_mult, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(Bottleneck, layer4_setting,
                                           width_mult, dilation=2, norm_layer=norm_layer)
        else:
            self.layer4 = self._make_layer(Bottleneck, layer4_setting,
                                           width_mult, norm_layer=norm_layer)

        # building last several layers
        classifier = list()
        if mode == 'large':
            if dilated:
                last_bneck_channels = int(960//2 * width_mult) if width_mult > 1.0 else 960//2
            else:
                last_bneck_channels = int(960 * width_mult) if width_mult > 1.0 else 960
            self.layer5 = _ConvBNHswish(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(_Hswish(True))
            classifier.append(nn.Conv2d(1280, num_classes, 1))
        elif mode == 'small':
            if dilated:
                last_bneck_channels = int(576//2 * width_mult) if width_mult > 1.0 else 576//2                
            else:
                last_bneck_channels = int(576 * width_mult) if width_mult > 1.0 else 576
            self.layer5 = _ConvBNHswish(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            classifier.append(SEModule(last_bneck_channels))
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1024, 1))
            classifier.append(_Hswish(True))
            classifier.append(nn.Conv2d(1024, num_classes, 1))
        else:
            raise ValueError('Unknown mode.')
        self.mode = mode
        self.classifier = nn.Sequential(*classifier)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self._init_weights()

    def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for k, exp_size, c, se, nl, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if (dilation == 1) else 1
            exp_channels = int(exp_size * width_mult)
            layers.append(block(self.in_channels, out_channels, exp_channels, k, stride, dilation, se, nl, norm_layer))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)          
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.classifier(x)
        x = self.dequant(x)
        x = x.view(x.size(0), x.size(1))
        return x
    
    def fuse_model(self):
        self.conv1.fuse_model()
        for layer in self.layer1:        
            layer.fuse_model()
        for layer in self.layer2:        
            layer.fuse_model()
        for layer in self.layer3:        
            layer.fuse_model() 
        for layer in self.layer4:        
            layer.fuse_model()        
        self.layer5.fuse_model()            
        if self.mode == 'small':        
            self.classifier[0].fuse_model()   
            
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def get_mobilenet_v3(mode='small', width_mult=1.0, pretrained=False, root='~/,torch/models', **kwargs):
    model = MobileNetV3(mode=mode, width_mult=width_mult, **kwargs)
    if pretrained:
        raise ValueError("Not support pretrained")
    return model

def mobilenet_v3_large(**kwargs):
    return get_mobilenet_v3('large', 1.0, **kwargs)


def mobilenet_v3_small(**kwargs):
    return get_mobilenet_v3('small', 1.0, **kwargs)