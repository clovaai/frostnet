import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from layers import *
from data import TDSOD_voc, TDSOD_coco
import os

from torch.quantization import QuantStub, DeQuantStub

from tqdm import tqdm
import time

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

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

class conv_bn(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, inp, oup, stride=1, k_size=3, padding=1, group=1):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()

        self.cbr = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=k_size, stride=stride, padding=padding, groups=group, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=False))

    def forward(self, x):
        return self.cbr(x)

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cbr, ['0', '1','2'], inplace=True)


class conv_bn_no_relu(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, inp, oup, stride=1, k_size=3, padding=1, group=1):
        super().__init__()

        self.cb = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=k_size, stride=stride, padding=padding, groups=group, bias=False),
            nn.BatchNorm2d(oup))

    def forward(self, x):
        return self.cb(x)

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cb, ['0', '1'], inplace=True)


class dwd_block(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, inp, oup):
        super().__init__()

        self.dwd1 = conv_bn(inp=inp, oup=oup, stride=1, k_size=1, padding=0)
        self.dwd2 = conv_bn(inp=oup, oup=oup, stride=1, k_size=3, padding=1, group=oup)

    def forward(self, x):
        return self.dwd2(self.dwd1(x))

    def fuse_model(self):
        self.dwd1.fuse_model()
        self.dwd2.fuse_model()

class trans_block(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, inp, oup):
        super().__init__()

        self.trn1 = conv_bn(inp=inp, oup=oup, stride=1, k_size=1, padding=0)
        self.trn2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x):
        return self.trn2(self.trn1(x))

    def fuse_model(self):
        self.trn1.fuse_model()


class downsample_0(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, in_channels, out_channels):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()

        self.dwn1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, ceil_mode=True)
        self.conv1 = conv_bn(inp=in_channels, oup=out_channels, stride=1, k_size=1, padding=0)

    def forward(self, x):
        return self.conv1(self.dwn1(x))

    def fuse_model(self):
        self.conv1.fuse_model()


class downsample_1(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv2 = conv_bn_no_relu(inp=in_channels, oup=out_channels, stride=1, k_size=1, padding=0)
        self.conv3 = conv_bn(inp=out_channels, oup=out_channels, stride=2, k_size=3, padding=1, group=out_channels)

    def forward(self, x):
        return self.conv3(self.conv2(x))

    def fuse_model(self):
        self.conv2.fuse_model()
        self.conv3.fuse_model()

class upsample(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = conv_bn(inp=in_channels, oup=in_channels, stride=1, k_size=3, padding=1, group=in_channels)

    def forward(self, x):
        return self.conv1(x)

    def fuse_model(self):
        self.conv1.fuse_model()


class baseNet(nn.Module):
    '''
        This class defines the basenet
    '''
    def __init__(self):
        super().__init__()

        self.base1 = conv_bn(inp=3, oup=64, stride=2, k_size=3, padding=1)
        self.base2 = conv_bn(inp=64, oup=64, stride=1, k_size=1, padding=0)
        self.base3 = conv_bn(inp=64, oup=64, stride=1, k_size=3, padding=1, group=64)
        self.base4 = conv_bn(inp=64, oup=128, stride=1, k_size=1, padding=0)
        self.base5 = conv_bn(inp=128, oup=128, stride=1, k_size=3, padding=1, group=128)
        self.max = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.base1(x)
        x = self.base2(x)
        x = self.base3(x)
        x = self.base4(x)
        x = self.base5(x)
        x = self.max(x)
        return x

    def fuse_model(self):
        self.base1.fuse_model()
        self.base2.fuse_model()
        self.base3.fuse_model()
        self.base4.fuse_model()
        self.base5.fuse_model()


class QSSD_TDSOD_Feat(nn.Module):

    def __init__(self, size, num_classes):
        super().__init__()

        self.size = size
        self.num_classes = num_classes
        self.cfg = (TDSOD_coco, TDSOD_voc)[num_classes == 21]
        self.num_feat = len(self.cfg['feature_maps'])

        self.quant = QuantStub()
        self.dequant_list = []
        for idx in range(self.num_feat):
            self.dequant_list.append(DeQuantStub())
        self.dequant_list = nn.ModuleList(self.dequant_list)

        self.base = baseNet()

        # list of ddb and trans layers
        self.ddb_0 = []
        self.quant_list_0 = []
        inp = 128
        for it in range(4):
            if it == 0:
                self.ddb_0.append(dwd_block(inp=inp, oup=32))
            else:
                inp += 32
                self.ddb_0.append(dwd_block(inp=inp, oup=32))
            self.quant_list_0.append(nn.quantized.FloatFunctional())
        self.quant_list_0 = nn.ModuleList(self.quant_list_0)
        self.ddb_0 = nn.ModuleList(self.ddb_0)
        self.trans_0 = trans_block(inp=256, oup=128)  # output: 38x38

        self.ddb_1 = []
        self.quant_list_1 = []
        inp = 128
        for it in range(6):
            if it == 0:
                self.ddb_1.append(dwd_block(inp=inp, oup=48))
            else:
                inp += 48
                self.ddb_1.append(dwd_block(inp=inp, oup=48))
            self.quant_list_1.append(nn.quantized.FloatFunctional())
        self.quant_list_1 = nn.ModuleList(self.quant_list_1)
        self.ddb_1 = nn.ModuleList(self.ddb_1)
        self.trans_1 = trans_block(inp=416, oup=128)  # output: 19x19

        self.ddb_2 = []
        self.quant_list_2 = []
        inp = 128
        for it in range(6):
            if it == 0:
                self.ddb_2.append(dwd_block(inp=inp, oup=64))
            else:
                inp += 64
                self.ddb_2.append(dwd_block(inp=inp, oup=64))
            self.quant_list_2.append(nn.quantized.FloatFunctional())
        self.quant_list_2 = nn.ModuleList(self.quant_list_2)
        self.ddb_2 = nn.ModuleList(self.ddb_2)
        self.trans_2 = conv_bn(inp=512, oup=256, stride=1, k_size=1, padding=0)  # output: 19x19

        self.ddb_3 = []
        self.quant_list_3 = []
        inp = 256
        for it in range(6):
            if it == 0:
                self.ddb_3.append(dwd_block(inp=inp, oup=80))
            else:
                inp += 80
                self.ddb_3.append(dwd_block(inp=inp, oup=80))
            self.quant_list_3.append(nn.quantized.FloatFunctional())
        self.quant_list_3 = nn.ModuleList(self.quant_list_3)
        self.ddb_3 = nn.ModuleList(self.ddb_3)
        self.trans_3 = conv_bn(inp=736, oup=64, stride=1, k_size=1, padding=0)  # output: 19x19

        # list of upsample and downsample layers
        self.downfeat_0 = []
        self.downfeat_1 = []
        for it in range(5):
            if it == 1:
                self.downfeat_0.append(downsample_0(in_channels=128 + 64, out_channels=64))
                self.downfeat_1.append(downsample_1(in_channels=128 + 64, out_channels=64))
            else:
                self.downfeat_0.append(downsample_0(in_channels=128, out_channels=64))
                self.downfeat_1.append(downsample_1(in_channels=128, out_channels=64))

        self.upfeat = []
        for it in range(5):
            self.upfeat.append(upsample(in_channels=128))

        self.downfeat_0 = nn.ModuleList(self.downfeat_0)
        self.downfeat_1 = nn.ModuleList(self.downfeat_1)
        self.upfeat = nn.ModuleList(self.upfeat)

        self.qadd1 = nn.quantized.FloatFunctional()
        self.qadd2 = nn.quantized.FloatFunctional()
        self.qadd3 = nn.quantized.FloatFunctional()
        self.qadd4 = nn.quantized.FloatFunctional()
        self.qadd5 = nn.quantized.FloatFunctional()

        self.qcat0 = nn.quantized.FloatFunctional()
        self.qcat1 = nn.quantized.FloatFunctional()
        self.qcat2 = nn.quantized.FloatFunctional()
        self.qcat3 = nn.quantized.FloatFunctional()
        self.qcat4 = nn.quantized.FloatFunctional()
        self.qcat5 = nn.quantized.FloatFunctional()


    def forward(self, x):
        """applies network layers and ops on input image(s) x.

        args:
            x: input image or batch of images. shape: [batch,3,300,300].

        return:
            depending on phase:
            test:
                variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, shape: [batch*num_priors,num_classes]
                    2: localization layers, shape: [batch,num_priors*4]
                    3: priorbox layers, shape: [2,num_priors*4]
        """
        sources = list()

        x = self.quant(x)

        # apply base networks
        x = self.base(x)  # output: 75x75

        # apply dwd_block 0
        for idx, blc in enumerate(self.ddb_0):
            x = self.quant_list_0[idx].cat((x, blc(x)), 1)

        x = self.trans_0(x)  # output: 38x38

        infeat_1 = x
        # apply dwd_block 1
        for idx, blc in enumerate(self.ddb_1):
            x = self.quant_list_1[idx].cat((x, blc(x)), 1)

        x = self.trans_1(x)  # output: 19x19

        # apply dwd_block 2
        for idx, blc in enumerate(self.ddb_2):
            x = self.quant_list_2[idx].cat((x, blc(x)), 1)

        x = self.trans_2(x)  # output: 19x19

        # apply dwd_block 3
        for idx, blc in enumerate(self.ddb_3):
            x = self.quant_list_3[idx].cat((x, blc(x)), 1)

        x = self.trans_3(x)  # output: 19x19
        infeat_2 = x
        infeat_3 = self.qcat0.cat((self.downfeat_0[0](infeat_1), self.downfeat_1[0](infeat_1)), 1)

        sz_x = infeat_3.size()[2]
        sz_y = infeat_3.size()[3]

        # why should we use sz_x, sz_y?
        s0 = self.qcat1.cat((infeat_3[:, :, :sz_x, :sz_y], infeat_2[:, :, :sz_x, :sz_y]), 1)
        s1 = self.qcat2.cat((self.downfeat_0[1](s0), self.downfeat_1[1](s0)), 1)
        s2 = self.qcat3.cat((self.downfeat_0[2](s1), self.downfeat_1[2](s1)), 1)
        s3 = self.qcat4.cat((self.downfeat_0[3](s2), self.downfeat_1[3](s2)), 1)
        s4 = self.qcat5.cat((self.downfeat_0[4](s3), self.downfeat_1[4](s3)), 1)

        sources.append(s4)

        u1 = self.qadd1.add(self.upfeat[0](F.interpolate(s4, size=(s3.size()[2], s3.size()[3]), mode='bilinear')), s3)
        sources.append(u1)
        u2 = self.qadd2.add(self.upfeat[1](F.interpolate(u1, size=(s2.size()[2], s2.size()[3]), mode='bilinear')), s2)
        sources.append(u2)
        u3 = self.qadd3.add(self.upfeat[2](F.interpolate(u2, size=(s1.size()[2], s1.size()[3]), mode='bilinear')), s1)
        sources.append(u3)
        u4 = self.qadd4.add(self.upfeat[3](F.interpolate(u3, size=(infeat_3.size()[2], infeat_3.size()[3]), mode='bilinear')), infeat_3)
        sources.append(u4)
        u5 = self.qadd5.add(self.upfeat[4](F.interpolate(u4, size=(infeat_1.size()[2], infeat_1.size()[3]), mode='bilinear')), infeat_1)
        sources.append(u5)

        ####### Features should be reversed order
        sources = sources[::-1]  ####### #reverse order

        output = []
        for idx, source in enumerate(sources):
            output.append(self.dequant_list[idx](source))

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            mdata = torch.load(base_file,
                               map_location=lambda storage, loc: storage)
            weights = mdata['weight']
            epoch = mdata['epoch']
            self.load_state_dict(weights)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')
        return epoch

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            # m.bias.data.zero_()

    def fuse_model(self):
        self.base.fuse_model()

        for itm in self.ddb_0:
            itm.fuse_model()
        self.trans_0.fuse_model()

        for itm in self.ddb_1:
            itm.fuse_model()
        self.trans_1.fuse_model()

        for itm in self.ddb_2:
            itm.fuse_model()
        self.trans_2.fuse_model()

        for itm in self.ddb_3:
            itm.fuse_model()
        self.trans_3.fuse_model()

        for itm in self.downfeat_0:
            itm.fuse_model()

        for itm in self.downfeat_1:
            itm.fuse_model()

        for itm in self.upfeat:
            itm.fuse_model()


class QSSD_TDSOD_HEAD(nn.Module):
    def __init__(self, phase='train', num_classes=21, cfg=[4, 6, 6, 6, 4, 4]):
        super().__init__()

        self.phase = phase
        self.num_classes = num_classes

        self.loc_layers = []
        self.conf_layers = []

        self.cfg = (TDSOD_coco, TDSOD_voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.get_prior()
        self.priors.requires_grad = False

        self.loc_layers += [conv_bn_no_relu(inp=128, oup=cfg[0] * 4, stride=1, k_size=3, padding=1)]
        self.conf_layers += [conv_bn_no_relu(inp=128, oup=cfg[0] * num_classes, stride=1, k_size=3, padding=1)]

        for k in range(1, 6):
            self.loc_layers += [conv_bn_no_relu(inp=128, oup=cfg[k] * 4, stride=1, k_size=3, padding=1)]
            self.conf_layers += [conv_bn_no_relu(inp=128, oup=cfg[k] * num_classes, stride=1, k_size=3, padding=1)]

        self.loc_layers = nn.ModuleList(self.loc_layers)
        self.conf_layers = nn.ModuleList(self.conf_layers)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def xavier(self, param):
        init.xavier_uniform(param)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            self.xavier(m.weight.data)
            # m.bias.data.zero_()

    def forward(self, sources):
        loc = []
        conf = []

        # apply multibox head to source layers
        for idx, (x, l, c) in enumerate(zip(sources, self.loc_layers, self.conf_layers)):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            output = self.detect.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                                       self.num_classes)),  # conf preds
                self.priors.type(type(x.data))  # default boxes
            )

        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output


def build_tdsod(phase, size=300, num_classes=21):

    return QSSD_TDSOD_Feat(size=size, num_classes=num_classes), \
           QSSD_TDSOD_HEAD(phase=phase, num_classes=num_classes, cfg=[4, 6, 6, 6, 4, 4])


if __name__ == '__main__':
    net, head = build_tdsod(phase='test')


    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(net, inplace=True)
    print(net)
    print_size_of_model(net)

    try:
        trained_model_net = 'weights/ssd300_f_170000.pth'
        trained_model_head = 'weights/ssd300_h_170000.pth'

        net.load_state_dict(torch.load(trained_model_net, map_location=torch.device('cpu')))
        head.load_state_dict(torch.load(trained_model_head, map_location=torch.device('cpu')))
    except:
        print('no pre-trained weights')

    net.train()

    quantized_net = torch.quantization.convert(net.cpu().eval(), inplace=False)

    print_size_of_model(net)
    print_size_of_model(quantized_net)

    inp = torch.rand((1, 3, 300, 300))

    tot_time = 0.
    for i in tqdm(range(1000)):
        tic = time.time()
        _ = head(quantized_net(inp))
        toc = time.time() - tic
        tot_time += toc

    tot_time = tot_time / 1000.

    print(tot_time)