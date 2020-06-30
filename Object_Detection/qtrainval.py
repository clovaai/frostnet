from data import TDSOD_voc, voc, VOCDetection, detection_collate, MEANS

from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from qtdsod import build_tdsod
from ssd_qmv2 import build_ssd
import os

import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
from utils.optimizer import get_optimizer


DATASET_PATH =''
VOC_ROOT = '/Data/VOC0712/train'

print('Run in Local')

# for evaluation
from qeval import evaluator

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--optimizer', default='QSGD', choices=['SGD', 'QSGD'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=20, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='pretrained/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--net_type', default='qssd', type=str,
                    help='networktype - tdsod qssd') # tdsosd, qssd
parser.add_argument('--expname', default='TEST', type=str,
                    help='name of experiment')
parser.add_argument('--scratch', dest='scratch', default=True, type=str2bool,
                    help='to train a model from scratch or not')
parser.add_argument('--warmup', default=True, type=str2bool,
                    help='require warm up epoch')
parser.add_argument('--quant', default=True, type=str2bool,
                    help='require warm up epoch')
parser.add_argument('--save_iter', default=10000, type=int,
                    help='save and evaluation per iteration')

args = parser.parse_args()

#QSGD params
args.toss_coin = True
args.nesterov = False
args.clip_by = 1e-3
args.noise_decay = 1e-2


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

num_gpu = torch.cuda.device_count()
print('num_gpu %d' % num_gpu)
args.save_folder = os.path.join(args.save_folder)

def train():
    if args.dataset == 'COCO':
        print("WARNING: Using default COCO dataset_root because " + "--dataset_root was not specified.")
        exit()
    elif args.dataset == 'VOC':
        if args.net_type == 'tdsod':
            print('tdsod setting')
            cfg = TDSOD_voc
        else:
            cfg = voc
        dataset = VOCDetection(root=VOC_ROOT, transform=SSDAugmentation(cfg['min_dim'], MEANS))



    if args.net_type == 'tdsod':
        print('Build TDSOD')
        net, head = build_tdsod(phase='train')
    elif args.net_type == 'qssd':
        print('Build SSD')
        net, head = build_ssd(phase='train')
    else:
        print('we only support tdsod and qssd. Thank you')
        exit()

    if args.cuda:
        if num_gpu > 1:
            net = torch.nn.DataParallel(net)
            head = torch.nn.DataParallel(head)
        cudnn.benchmark = True

    print(net)
    print(head)
    print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
    print('training step: ',cfg['lr_steps'])
    print('max step: ', cfg['max_iter'])

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)

    if args.cuda:
        net = net.cuda()
        head = head.cuda()


    if args.scratch:
        if args.net_type == 'tdsod':
            print('Initializing weights for training from SCRATCH - TDSOD')
            net.apply(weights_init)
            head.apply(weights_init)
        elif args.net_type == 'qssd':
            print('Initializing weights for training from SCRATCH - QSSD')
            net.extras.apply(weights_init)
            head.apply(weights_init)
    else:
        print('This code only support scratch mode')
        exit()

    optimizer = get_optimizer(args.optimizer, list(net.parameters())+list(head.parameters()), args=args)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()
    head.train()

    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0

    if not os.path.exists('weights'):
        os.makedirs('weights')
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size # epoch_Size가 아니라 iteration_size일듯 
    print('Dataset size, Total epoch:', len(dataset), (cfg['max_iter']-args.start_iter)/epoch_size)
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    # assign fp warmup
    # create batch iterator
    if args.quant == True:
        if args.warmup == True:
            print('start optimizer warm-up')
            batch_iterator = None
            start_time = time.time()
            for iteration in range(args.start_iter, 2*epoch_size):
                #     print(not batch_iterator, iteration % epoch_size)
                t0 = time.time()
                if (not batch_iterator) or (iteration % epoch_size == 0):
                    batch_iterator = iter(data_loader)
                if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
                    # reset epoch loss counters
                    loc_loss = 0
                    conf_loss = 0

                if iteration in cfg['lr_steps']:
                    step_index += 1
                    adjust_learning_rate(optimizer, args.gamma, step_index)

                # load train data
                images, targets = next(batch_iterator)

                if args.cuda:
                    images = Variable(images.cuda())
                    with torch.no_grad():
                        targets = [Variable(ann.cuda()) for ann in targets]
                else:
                    images = Variable(images)
                    with torch.no_grad():
                        targets = [Variable(ann) for ann in targets]
                # forward
                t1 = time.time()
                out = head(net(images))

                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()
                t2 = time.time()
                loc_loss += loss_l.item()
                conf_loss += loss_c.item()

                if iteration % 100 == 0:
                    current_LR = get_learning_rate(optimizer)[0]
                    print(
                        'iter ' + repr(iteration) + '|| LR: ' + repr(current_LR) + ' || Loss: %.4f ||' % (loss.item()),
                        'batch/forw. time: %.2f' % (t2 - t0), '%.2f:' % (t2 - t1),
                        'avg. time: %.4f' % ((time.time() - start_time) / (iteration + 1)))
            start_iter = iteration
        else:
            start_iter = 0

        # quantization on
        if num_gpu > 1:
            net.module.fuse_model()
            net.module.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
            torch.quantization.prepare_qat(net.module, inplace=True)
            print('quant on')
        else:
            net.fuse_model()
            net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
            torch.quantization.prepare_qat(net, inplace=True)
            print('quant on')
    else:
        print('run FP only model')
        start_iter = 0

    # create batch iterator
    batch_iterator = None  
    start_time = time.time()
    for iteration in range(start_iter, cfg['max_iter']):

        t0 = time.time()
        if (not batch_iterator) or (iteration % epoch_size == 0):
            batch_iterator = iter(data_loader)
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]
        # forward
        t1 = time.time()
        out = head(net(images))

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t2 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 100 == 0:
            current_LR = get_learning_rate(optimizer)[0]

            print('iter ' + repr(iteration) + '|| LR: ' + repr(current_LR) + ' || Loss: %.4f ||' % (loss.item()),
                    'batch/forw. time: %.2f' % (t2 - t0),  '%.2f:' % (t2-t1), 'avg. time: %.4f' %((time.time()-start_time)/(iteration+1)))


        if iteration != 0 and iteration % args.save_iter == 0:
            print('Saving state and evaluation at iter:', iteration)
            if num_gpu>1:
                torch.save(net.module.state_dict(), 'weights/ssd300_f_' + repr(iteration) + '.pth')
                torch.save(head.module.state_dict(), 'weights/ssd300_h_' + repr(iteration) + '.pth')
            else:
                torch.save(net.state_dict(), 'weights/ssd300_f_' + repr(iteration) + '.pth')
                torch.save(head.state_dict(), 'weights/ssd300_h_' + repr(iteration) + '.pth')
            mean_AP = evaluator(args.net_type, 'VOC0712',
                                'weights/ssd300_f_' + repr(iteration) + '.pth',
                                'weights/ssd300_h_' + repr(iteration) + '.pth',
                                cuda=args.cuda,
                                quant=args.quant,
                                verbose=False)


    if num_gpu > 1:
        torch.save(net.module.state_dict(), 'weights/ssd300_f_final.pth')
        torch.save(head.module.state_dict(), 'weights/ssd300_f_final.pth')
    else:
        torch.save(net.state_dict(), 'weights/ssd300_f_final.pth')
        torch.save(head.state_dict(), 'weights/ssd300_h_final.pth')


def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
#        print(m, m.bias, hasattr(m.bias, 'data'))
        if hasattr(m.bias, 'data'):
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()



if __name__ == '__main__':
    train()
    temp = evaluator(args.net_type, 'VOC0712', 'weights/ssd300_f_final.pth', 'weights/ssd300_h_final.pth', cuda=args.cuda, quant=args.quant, verbose=True)
