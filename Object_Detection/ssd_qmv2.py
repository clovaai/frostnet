
from torchvision.models.mobilenet import InvertedResidual, ConvBNReLU, MobileNetV2
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from data import voc, coco
import os

from tqdm import tqdm
import time

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def _make_divisible(v, divisor, min_value=None):
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


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
        if dilation>1:
            padding = dilation
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )

class ConvBN(nn.Module):
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

        self.cb = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=k_size, stride=stride, padding=padding, groups=group, bias=False),
            nn.BatchNorm2d(oup))

        self.out_channels = oup

    def forward(self, x):
        return self.cb(x)

    def fuse_model(self):
        torch.quantization.fuse_modules(self.cb, ['0', '1'], inplace=True)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, dilation=dilation, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, momentum=0.1),
        ])
        self.conv = nn.Sequential(*layers)

        # Replace torch.add with floatfunctional
        if self.use_res_connect:
            self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s, d
                [1, 16, 1, 1, 1],
                [6, 24, 2, 2, 1],
                [6, 32, 3, 2, 1],
                [6, 64, 4, 2, 1],
                [6, 96, 3, 1, 1],
                [6, 160, 3, 1, 2],
                [6, 320, 1, 1, 2],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 5:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s, d in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, dilation=d, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # weight initialization
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
                nn.init.zeros_(m.bias)

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == InvertedResidual:
                for idx in range(len(m.conv)):
                    if type(m.conv[idx]) == nn.Conv2d:
                        torch.quantization.fuse_modules(m.conv, [str(idx), str(idx + 1)], inplace=True)


def add_extras_bn_group(cfg, strides, i, batch_norm=True):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if v == 'P':
            layers += [nn.AvgPool2d(3,3)]
        else:
            convbn = ConvBN(inp=in_channels, oup=v, stride=strides[k], padding=(0,1)[strides[k] == 2], k_size=(1, 3)[flag], group=(1,in_channels)[flag])

            layers.append(convbn)
        flag = not flag
        in_channels = v
    return layers


class SSD_MobileNetV2_Feat(nn.Module):
    def __init__(self, size, base, extras, extras_head_pos, use_final_conv):
        super(SSD_MobileNetV2_Feat, self).__init__()

        #Quant, Dequant
        self.size = size
        self.quant = QuantStub()
        self.dequant_list = []
        for idx in range(len(extras_head_pos)+2):
            self.dequant_list.append(DeQuantStub())
        self.dequant_list = nn.ModuleList(self.dequant_list)

        # SSD network
        self.vgg = base
        if use_final_conv == False:
            self.vgg.finalconv = None
        # Layer learns to scale the l2 normalized features from conv4_3
        self.extras = nn.ModuleList(extras)
        self.extras_head_pos = extras_head_pos

    def forward(self, x):

        sources = list()

        x = self.quant(x)
        # apply vgg up to conv4_3 relu
        for k in range(7):
            x = self.vgg.features[k](x)

        s = x  ## when L2 norm is not used - L2Norm depreciated
        sources.append(s)

        # apply vgg up to fc7
        for k in range(7, len(self.vgg.features)):
            x = self.vgg.features[k](x)

        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k in self.extras_head_pos:
                sources.append(x)

        output = []
        for idx, source in enumerate(sources):
            output.append(self.dequant_list[idx](source))
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

    def fuse_model(self):
        self.vgg.fuse_model()
        for v, extra in enumerate(self.extras):
            if v <len(self.extras)-1:
                extra.fuse_model()



class SSD_MobileNetV2_HEAD(nn.Module):
    def __init__(self, base, extra_layers, cfg, extras_head_pos, num_classes, phase='train'):
        super(SSD_MobileNetV2_HEAD, self).__init__()
        self.data_cfg = (coco, voc)[num_classes == 21]
        self.phase = phase
        self.num_classes = num_classes
        self.loc_layers = []
        self.conf_layers = []

        self.priorbox = PriorBox(self.data_cfg)
        self.priors = self.priorbox.get_prior()
        self.priors.requires_grad = False

        self.loc_layers += [ConvBN(inp=base.features[6].conv[2].out_channels, oup=cfg[0] * 4, k_size=3, stride=1)]
        self.conf_layers += [ConvBN(inp=base.features[6].conv[2].out_channels, oup=cfg[0] * num_classes, k_size=3, stride=1)]
        self.loc_layers += [ConvBN(inp=base.features[-1][0].out_channels, oup=cfg[1] * 4, k_size=3, stride=1)]
        self.conf_layers += [ConvBN(inp=base.features[-1][0].out_channels, oup=cfg[1] * num_classes, k_size=3, stride=1)]

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

        idx = 2
        for k, v in enumerate(extras_head_pos):
            try:
                self.loc_layers += [ConvBN(inp=extra_layers[v].out_channels, oup=cfg[idx] * 4, k_size=3, stride=1)]
                self.conf_layers += [ConvBN(inp=extra_layers[v].out_channels, oup=cfg[idx] * num_classes, k_size=3, stride=1)]
                idx += 1
            except:
                self.loc_layers += [ConvBN(inp=extra_layers[v-1].out_channels, oup=cfg[idx] * 4, k_size=3, stride=1)]
                self.conf_layers += [ConvBN(inp=extra_layers[v-1].out_channels, oup=cfg[idx] * num_classes, k_size=3, stride=1)]
                idx += 1

        self.loc_layers = nn.ModuleList(self.loc_layers)
        self.conf_layers = nn.ModuleList(self.conf_layers)

    def forward(self, sources):

        loc = []
        conf = []
        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc_layers, self.conf_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
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



def build_ssd(phase, size=300, num_classes=21):
    model = MobileNetV2()
    weights = torch.load('weights/mobilenet_v2-float.pth')
    model.load_state_dict(weights, strict=False)


    use_final_conv = True
    batch_norm_on_extra_layers = True
    final_feat_dim = 1280


    extras_ = add_extras_bn_group([32, 128, 32, 128, 32, 128, 'P'],
                                  [1,2,1,2,1,2,1],
                                  final_feat_dim,
                                  batch_norm=batch_norm_on_extra_layers)

    return SSD_MobileNetV2_Feat(size, model, extras_, [1, 3, 5, 6], use_final_conv), \
           SSD_MobileNetV2_HEAD(model,
                                add_extras_bn_group([32, 128, 32, 128, 32, 128, 'P'],
                                [1,2,1,2,1,2,1],
                                final_feat_dim,
                                batch_norm=batch_norm_on_extra_layers),
                                [4, 6, 6, 6, 4, 4],
                                [1, 3, 5, 6],
                                num_classes,
                                phase=phase)


if __name__ == '__main__':
    net, head = build_ssd(phase='test')

    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(net, inplace=True)
    print(net)
    print_size_of_model(net)

    try:
        trained_model_net = 'weights/ssd300_f_110000.pth'
        trained_model_head = 'weights/ssd300_h_110000.pth'

        net.load_state_dict(torch.load(trained_model_net, map_location=torch.device('cpu')), strict=False)
        head.load_state_dict(torch.load(trained_model_head, map_location=torch.device('cpu')), strict=False)
    except:
        print('no pre-trained weight')

    net.train()

    quantized_net = torch.quantization.convert(net.cpu().eval(), inplace=False)

    print_size_of_model(net)
    print_size_of_model(quantized_net)

    inp = torch.rand((1, 3, 300, 300))
    head.eval()
    tot_time = 0.
    for i in tqdm(range(1000)):
        tic = time.time()
        _ = head(quantized_net(inp))
        toc = time.time() - tic
        tot_time += toc

    tot_time = tot_time / 1000.

    print(tot_time)
