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

import torch
from torch.nn import init
from model.layers.espnet_utils import *
from model.backbones.espnetv2 import *
from utilities.print_utils import *
from torch.nn import functional as F
from model.backbones.espnetv2 import *

class ESPNetv2Segmentation(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the Semantic Segmenation
    '''

    def __init__(self, args, classes=21, dataset='pascal'):
        super().__init__()

        # =============================================================
        #                       BASE NETWORK
        # =============================================================
        classificationNet = EESPNet(args) #imagenet model
        self.net = classificationNet        
        del self.net.classifier
        del self.net.level5
        del self.net.level5_0
 
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        #=============================================================
        #                   SEGMENTATION NETWORK
        #=============================================================

        self.quant_cat1 = nn.quantized.FloatFunctional()
        self.quant_cat2 = nn.quantized.FloatFunctional()
        self.quant_cat3 = nn.quantized.FloatFunctional() 

        if args.s <=0.5:
            p = 0.1
        else:
            p = 0.2        
        self.proj_L4_C = CBR(self.net.level4[-1].act_out,
                             self.net.level3[-1].act_out, 1, 1)
        pspSize = 2*self.net.level3[-1].act_out
        self.pspMod = nn.Sequential(EESP(pspSize, pspSize //2, stride=1, k=4, r_lim=7),
                                    PSPModule(pspSize // 2, pspSize //2))
        
        #self.project_l3 = nn.Sequential(nn.Dropout2d(p=p), C(pspSize // 2, classes, 1, 1))
        self.project_l3 =  CBR(pspSize // 2, classes, 1, 1)  
        self.act_l3 = CBR(classes,classes,1,1)
        self.project_l2 = CBR(self.net.level2_0.act_out + classes, classes, 1, 1)
        
        #self.project_l1 = nn.Sequential(nn.Dropout2d(p=p), 
        #                                C(self.net.level1_act_out + classes, classes, 1, 1))
    
        self.init_params()
        
    def forward(self, x):
        '''
        :param x: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        x = self.quant(x)
        out_l1, out_l2, out_l3, out_l4 = self.net(x, seg=True)
        out_l4_proj = self.proj_L4_C(out_l4)
        up_l4_to_l3 = F.interpolate(out_l4_proj, scale_factor=2, mode='bilinear', align_corners=True)
        merged_l3_upl4 = self.pspMod(self.quant_cat1.cat([out_l3, up_l4_to_l3], 1))
        proj_merge_l3_bef_act = self.project_l3(merged_l3_upl4)
        proj_merge_l3 = self.act_l3(proj_merge_l3_bef_act)
        out_up_l3 = F.interpolate(proj_merge_l3, scale_factor=2, mode='bilinear', align_corners=True)
        merge_l2 = self.project_l2(self.quant_cat2.cat([out_l2, out_up_l3], 1))
        out_up_l2 = F.interpolate(merge_l2, scale_factor=2, mode='bilinear', align_corners=True)
        output = self.dequant(self.quant_cat3.cat([out_l1, out_up_l2], 1))        
        
        return output
    
    def fuse_model(self):
        self.net.fuse_model(seg=True)
        self.proj_L4_C.fuse_model()
        self.pspMod[0].fuse_model()
        self.pspMod[1].fuse_model()
        self.act_l3.fuse_model()
        self.project_l3.fuse_model()        
        self.project_l2.fuse_model()
        
    def hierarchicalUpsample(self, x, factor=3):
        for i in range(factor):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

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

                    
class ESPNetv2Seg(nn.Module):
    def __init__(self,args, classes=20, dataset='pascal'):
        super().__init__()
        self.quantized = ESPNetv2Segmentation(args, classes=classes, dataset=dataset)
        self.classifier = C(self.quantized.net.level1_act_out + classes, classes, 1, 1) 
        
    def forward(self, input):
        x = self.quantized(input)
        x = self.classifier(x)
        x = F.interpolate(x , scale_factor=2, mode='bilinear', align_corners=True)        
        return x                    
        
def espnetv2_seg(args):
    classes = args.classes
    scale=args.s
    dataset=args.dataset
    model = ESPNetv2Seg(args, classes=classes, dataset=dataset)

    return model

if __name__ == "__main__":

    from utilities.utils import compute_flops, model_parameters
    import torch
    import argparse

    parser = argparse.ArgumentParser(description='Testing')
    args = parser.parse_args()

    args.classes = 21
    args.s = 2.0
    args.weights='../classification/model_zoo/espnet/espnetv2_s_2.0_imagenet_224x224.pth'
    args.dataset='pascal'

    input = torch.Tensor(1, 3, 384, 384)
    model = espnetv2_seg(args)
    from utilities.utils import compute_flops, model_parameters
    print_info_message(compute_flops(model, input=input))
    print_info_message(model_parameters(model))
    out = model(input)
    print_info_message(out.size())
