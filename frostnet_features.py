import torch.nn as nn
import torch
import math
import os
import torch.nn.functional as F
from collections import OrderedDict
from ..builder import BACKBONES


def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        print("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                           padding, dilation, groups, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(False)
                                )
    def forward(self, x):
        x = self.conv(x)
        return x
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self.conv, ['0', '1','2'], inplace=True)            

class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                           padding, dilation, groups, bias=False),
                                 nn.ReLU(False)
                                )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self.conv, ['0', '1'], inplace=True)

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                           padding, dilation, groups, bias=False),
                                 nn.BatchNorm2d(out_channels)
                                )

    def forward(self, x):
        x = self.conv(x)
        return x
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self.conv, ['0', '1'], inplace=True)    

def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
        
class CascadePreExBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels, quantized = False,
                 kernel_size=3, stride=1, dilation=1,expand_ratio=6,
                 reduce_factor = 4, block_type = 'CAS'):
        super(CascadePreExBottleneck, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.quantized = quantized
        if in_channels//reduce_factor < 8:
            block_type = 'MB'
        self.block_type = block_type
        
        r_channels = _make_divisible(in_channels//reduce_factor)
         
        if stride == 1 and in_channels==out_channels:
            self.reduction = False         
        else:
            self.reduction = True
            
        if self.expand_ratio == 1:
            self.squeeze_conv = None
            self.conv1 = None    
            n_channels = in_channels
        else:
            if block_type == 'CAS':
                self.squeeze_conv = ConvBNReLU(in_channels,r_channels, 1) 
                n_channels = r_channels + in_channels  
            else:
                n_channels = in_channels
            self.conv1 = ConvBNReLU(n_channels,n_channels*expand_ratio, 1) 
        self.conv2 = ConvBNReLU(n_channels*expand_ratio, n_channels*expand_ratio, kernel_size, stride, 
                              (kernel_size - 1) // 2 , 1,
                              groups=n_channels*expand_ratio)
        self.reduce_conv = ConvBN(n_channels*expand_ratio, out_channels, 1)
        if self.quantized:
            self.skip_add = nn.quantized.FloatFunctional()
            self.quant_cat = nn.quantized.FloatFunctional()

    def forward(self, x):        
        if not self.expand_ratio == 1:
            if self.block_type == 'CAS':
                squeezed = self.squeeze_conv(x)
                if self.quantized:
                    out = self.quant_cat.cat([squeezed,x],1)
                else:
                    out = torch.cat([squeezed,x],1)
            else:
                out = x
            out = self.conv1(out)
        else:
            out = x
        out = self.conv2(out)
        out = self.reduce_conv(out)

        if not self.reduction:          
            if self.quantized:
                out = self.skip_add.add(x,out)                
            else:
                out = torch.add(x,out)
        return out 

@BACKBONES.register_module()
class FrostNet(nn.Module):
    def __init__(self, mode='large', width_mult=1.0, bottleneck=CascadePreExBottleneck,quantized =False,pretrained='', **kwargs):
        super(FrostNet, self).__init__()
        
        self.quantized = quantized
        if mode == 'large':
            layer1_setting = [
                    # kernel_size, c, e, r, s
                    [3, 16, 1, 1, 1], #0
                    [3, 24, 6, 4, 2], #1
                    #[, , , , ],      #2
                    #[, , , , ],      #3        
                    [3, 24, 3, 4, 1], #4                
                    ]
            layer2_setting = [
                    [5, 40, 6, 4, 2], #5
                    #[, , , , ],      #6
                    #[, , , , ],      #7                
                    [3, 40, 3, 4, 1], #8

                    ]
            
            layer3_setting = [
                    [5, 80, 6, 4, 2], #9               
                    #[, , , , ],      #10  
                    [5, 80, 3, 4, 1], #11 
                    [5, 80, 3, 4, 1], #12 
                
                    [5, 96, 6, 4, 1], #13
                    #[, , , , ],      #14
                    [5, 96, 3, 4, 1], #15
                    [3, 96, 3, 4, 1], #16       
                    [3, 96, 3, 4, 1], #17                    
                    ]

            layer4_setting = [
                    [5, 192, 6, 2, 2], #18
                    [5, 192, 6, 4, 1], #19
                    [5, 192, 6, 4, 1], #20
                    [5, 192, 3, 4, 1], #21
                    [5, 192, 3, 4, 1], #22                
                    ]   

            layer5_setting = [
                    [5, 320, 6, 2, 1], #23
                    ]         
            
        elif mode == 'base':
            layer1_setting = [
                    # kernel_size, c, e, r, s
                    [3, 16, 1, 1, 1], #0
                    [5, 24, 6, 4, 2], #1
                    #[, , , , ],      #2
                    #[, , , , ],      #3        
                    [3, 24, 3, 4, 1], #4                
                    ]
            layer2_setting = [
                    [5, 40, 3, 4, 2], #5
                    #[, , , , ],      #6
                    [5, 40, 3, 4, 1], #7
                    #[, , , , ],      #8
                    ]
            
            layer3_setting = [
                    [5, 80, 3, 4, 2], #9               
                    #[, , , , ],      #10  
                    #[, , , , ],      #11 
                    [3, 80, 3, 4, 1], #12 
                
                    [5, 96, 3, 2, 1], #13
                    [3, 96, 3, 4, 1], #14
                    [5, 96, 3, 4, 1], #15
                    [5, 96, 3, 4, 1], #16              
                    ]

            layer4_setting = [
                    [5, 192, 6, 2, 2], #17
                    [5, 192, 3, 2, 1], #18
                    [5, 192, 3, 2, 1], #19
                    [5, 192, 3, 2, 1], #20
                    ]   

            layer5_setting = [
                    [5, 320, 6, 2, 1], #21
                    ] 
            
        elif mode == 'small':
            layer1_setting = [
                    # kernel_size, c, e, r, s
                    [3, 16, 1, 1, 1], #0
                    [5, 24, 3, 4, 2], #1
                    [3, 24, 3, 4, 1],      #2
                    #[, , , , ],      #3                       
                    ]
            layer2_setting = [
                    [5, 40, 3, 4, 2], #4
                    #[, , , , ],      #5
                    #[, , , , ],      #6
                    ]
            
            layer3_setting = [
                    [5, 80, 3, 4, 2], #7              
                    [5, 80, 3, 4, 1], #8 
                    [3, 80, 3, 4, 1], #9 
                
                    [5, 96, 3, 2, 1], #10
                    [5, 96, 3, 4, 1], #11
                    [5, 96, 3, 4, 1], #12             
                    ]

            layer4_setting = [
                    [5, 192, 6, 4, 2], #13
                    [5, 192, 6, 4, 1], #14
                    [5, 192, 6, 4, 1], #15
                    ]   

            layer5_setting = [
                    [5, 320, 6, 2, 1], #16
                    ]          
        else:
            raise ValueError('Unknown mode.')
        # building first layer

        self.in_channels = _make_divisible(int(32*min(1.0,width_mult)))

      
        self.conv1 = ConvBNReLU(3, self.in_channels, 3, 2, 1)          
        
        # building bottleneck blocks
        self.layer1 = self._make_layer(bottleneck, layer1_setting, width_mult, 1)    
        self.layer2 = self._make_layer(bottleneck, layer2_setting, width_mult, 1)          
        self.layer3 = self._make_layer(bottleneck, layer3_setting, width_mult, 1)           
        self.layer4 = self._make_layer(bottleneck, layer4_setting, width_mult, 1)        
        self.layer5 = self._make_layer(bottleneck, layer5_setting, width_mult, 1)

        
        # building last several layers
        last_in_channels = self.in_channels

        self.mode = mode  
        #self.init_weights(pretrained)
        #self._freeze_stages()

    def init_weights(self, pretrained):
        if pretrained !='':
            load_checkpoint(self, pretrained, use_ema=True, strict=False)     
        else:
            print('No pretrained backbone provided')            
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
                        
    def _make_layer(self, block, block_setting, width_mult, dilation=1):
        layers = list()
        for k, c, e, r, s in block_setting:
            out_channels = _make_divisible(int(c * width_mult))
            stride = s if (dilation == 1) else 1
            layers.append(block(self.in_channels, out_channels, quantized = self.quantized, kernel_size = k, 
                                stride=s, dilation=dilation, expand_ratio=e, reduce_factor = r))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x) 
        x1 = self.layer1(x)  
        x2 = self.layer2(x1)     
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        
        features = [x1, x2, x3, x5]

        return features

    def _freeze_stages(self):
        '''Freeze BatchNorm layers.'''
        print('Freeze BatchNorm layers.')
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()  
                


