import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utilities.utils import save_checkpoint, model_parameters, compute_flops
from utilities.train_eval_seg import train_seg as train
from utilities.train_eval_seg import val_seg as val
from utilities.lr_scheduler import get_lr_scheduler

#from torch.utils.tensorboard import SummaryWriter
from loss_fns.segmentation_loss import SegmentationLoss
import random
import math
import time
import numpy as np
from utilities.print_utils import *
from utilities.optimizer import *
from utilities.Tensor_logger import Logger
import torch.quantization

from torch.quantization.fake_quantize import FakeQuantize

def disable_observer(mod):
    if type(mod) == FakeQuantize:
        if mod.scale is not None and mod.zero_point is not None:
            mod.disable_observer()
        else:
            pass
            #print("Can't disable {} : scale or zero_point is None.".format(mod))
            
def main(args):
    logdir = args.savedir+'/logs/'
    if not os.path.isdir(logdir):
        os.makedirs(logdir)    

    my_logger = Logger(60066, logdir)
        
    if args.dataset == 'pascal':  
        crop_size = (512, 512) 
        args.scale = (0.5, 2.0)        
    elif args.dataset == 'city':        
        crop_size = (768, 768)
        args.scale = (0.5, 2.0)         

    print_info_message('Running Model at image resolution {}x{} with batch size {}'.format(crop_size[1], crop_size[0], args.batch_size))
    if not os.path.isdir(args.savedir):
        os.makedirs(args.savedir)

    if args.dataset == 'pascal':
        from data_loader.segmentation.voc import VOCSegmentation, VOC_CLASS_LIST
        train_dataset = VOCSegmentation(root=args.data_path, train=True, crop_size=crop_size, scale=args.scale,
                                        coco_root_dir=args.coco_path)
        val_dataset = VOCSegmentation(root=args.data_path, train=False, crop_size=crop_size, scale=args.scale)
        seg_classes = len(VOC_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
    elif args.dataset == 'city':
        from data_loader.segmentation.cityscapes import CityscapesSegmentation, CITYSCAPE_CLASS_LIST
        train_dataset = CityscapesSegmentation(root=args.data_path, train=True, size=crop_size, scale=args.scale, coarse=args.coarse)
        val_dataset = CityscapesSegmentation(root=args.data_path, train=False, size=crop_size, scale=args.scale,
                                             coarse=False)
        seg_classes = len(CITYSCAPE_CLASS_LIST)
        class_wts = torch.ones(seg_classes)
        class_wts[0] = 2.8149201869965
        class_wts[1] = 6.9850029945374
        class_wts[2] = 3.7890393733978
        class_wts[3] = 9.9428062438965
        class_wts[4] = 9.7702074050903
        class_wts[5] = 9.5110931396484
        class_wts[6] = 10.311357498169
        class_wts[7] = 10.026463508606
        class_wts[8] = 4.6323022842407
        class_wts[9] = 9.5608062744141
        class_wts[10] = 7.8698215484619
        class_wts[11] = 9.5168733596802
        class_wts[12] = 10.373730659485
        class_wts[13] = 6.6616044044495
        class_wts[14] = 10.260489463806
        class_wts[15] = 10.287888526917
        class_wts[16] = 10.289801597595
        class_wts[17] = 10.405355453491
        class_wts[18] = 10.138095855713
        class_wts[19] = 0.0
    else:
        print_error_message('Dataset: {} not yet supported'.format(args.dataset))
        exit(-1)

    print_info_message('Training samples: {}'.format(len(train_dataset)))
    print_info_message('Validation samples: {}'.format(len(val_dataset)))

    if args.model == 'espnetv2':
        from model.espnetv2 import espnetv2_seg
        args.classes = seg_classes
        model = espnetv2_seg(args)
    elif args.model == 'espnet':
        from model.espnet import espnet_seg
        args.classes = seg_classes
        model = espnet_seg(args)
    elif args.model == 'mobilenetv2_1_0':
        from model.mobilenetv2 import get_mobilenet_v2_1_0_seg
        args.classes = seg_classes
        model = get_mobilenet_v2_1_0_seg(args)
    elif args.model == 'mobilenetv2_0_35':
        from model.mobilenetv2 import get_mobilenet_v2_0_35_seg
        args.classes = seg_classes
        model = get_mobilenet_v2_0_35_seg(args)   
    elif args.model == 'mobilenetv2_0_5':
        from model.mobilenetv2 import get_mobilenet_v2_0_5_seg
        args.classes = seg_classes
        model = get_mobilenet_v2_0_5_seg(args)           
    elif args.model == 'mobilenetv3_small':
        from model.mobilenetv3 import get_mobilenet_v3_small_seg
        args.classes = seg_classes
        model = get_mobilenet_v3_small_seg(args) 
    elif args.model == 'mobilenetv3_large':
        from model.mobilenetv3 import get_mobilenet_v3_large_seg
        args.classes = seg_classes
        model = get_mobilenet_v3_large_seg(args)
    elif args.model == 'mobilenetv3_RE_small':
        from model.mobilenetv3 import get_mobilenet_v3_RE_small_seg
        args.classes = seg_classes
        model = get_mobilenet_v3_RE_small_seg(args) 
    elif args.model == 'mobilenetv3_RE_large':
        from model.mobilenetv3 import get_mobilenet_v3_RE_large_seg
        args.classes = seg_classes
        model = get_mobilenet_v3_RE_large_seg(args)           
    else:
        print_error_message('Arch: {} not yet supported'.format(args.model))
        exit(-1)        

    num_gpus = torch.cuda.device_count()
    device = 'cuda' if num_gpus > 0 else 'cpu'

    train_params = []
    params_dict = dict(model.named_parameters()) 
    others= args.weight_decay*0.01    
    for key, value in params_dict.items():
        if len(value.data.shape) == 4:
            if value.data.shape[1] == 1:
                train_params += [{'params': [value], 'lr': args.lr, 'weight_decay': 0.0}]
            else:
                train_params += [{'params': [value], 'lr': args.lr, 'weight_decay': args.weight_decay}]
        else:
            train_params += [{'params': [value],'lr': args.lr, 'weight_decay': others}]  

    args.learning_rate = args.lr
    optimizer = get_optimizer(args.optimizer, train_params, args)
    num_params = model_parameters(model)
    flops = compute_flops(model, input=torch.Tensor(1, 3, crop_size[1], crop_size[0]))
    print_info_message('FLOPs for an input of size {}x{}: {:.2f} million'.format(crop_size[1], crop_size[0], flops))
    print_info_message('Network Parameters: {:.2f} million'.format(num_params))

    start_epoch = 0
    epochs_len = args.epochs
    best_miou = 0.0

    #criterion = nn.CrossEntropyLoss(weight=class_wts, reduction='none', ignore_index=args.ignore_idx)
    criterion = SegmentationLoss(n_classes=seg_classes, loss_type=args.loss_type,
                                 device=device, ignore_idx=args.ignore_idx,
                                 class_wts=class_wts.to(device))

    if num_gpus >= 1:
        if num_gpus == 1:
            # for a single GPU, we do not need DataParallel wrapper for Criteria.
            # So, falling back to its internal wrapper
            from torch.nn.parallel import DataParallel
            model = DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
        else:
            from utilities.parallel_wrapper import DataParallelModel, DataParallelCriteria
            model = DataParallelModel(model)
            model = model.cuda()
            criterion = DataParallelCriteria(criterion)
            criterion = criterion.cuda()

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True, num_workers=args.workers,drop_last=True)
    if args.dataset == 'city':
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers, drop_last=True)
    else:
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             pin_memory=True, num_workers=args.workers, drop_last=True)        


    lr_scheduler = get_lr_scheduler(args)
    
    print_info_message(lr_scheduler)

    with open(args.savedir + os.sep + 'arguments.json', 'w') as outfile:
        import json
        arg_dict = vars(args)
        arg_dict['model_params'] = '{} '.format(num_params)
        arg_dict['flops'] = '{} '.format(flops)
        json.dump(arg_dict, outfile)

    extra_info_ckpt = '{}_{}_{}'.format(args.model, args.s, crop_size[0])

    if args.fp_epochs > 0:
        print_info_message("========== MODEL FP WARMUP ===========")

        for epoch in range(args.fp_epochs):        
            lr = lr_scheduler.step(epoch)

            for param_group in optimizer.param_groups:         
                param_group['lr'] = lr

            print_info_message(
            'Running epoch {} with learning rates: {:.6f}'.format(epoch, lr))
            start_t = time.time()  
            miou_train, train_loss = train(model, train_loader, optimizer, criterion, 
                                           seg_classes, epoch, device=device)
    if args.optimizer.startswith('Q'):
        optimizer.is_warmup = False
        print('exp_sensitivity calibration fin.')    
        
    if not args.fp_train:
        model.module.quantized.fuse_model()
        model.module.quantized.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(model.module.quantized, inplace=True)
    
    if args.resume:
        start_epoch = args.start_epoch
        if os.path.isfile(args.resume):
            print_info_message('Loading weights from {}'.format(args.resume))
            weight_dict = torch.load(args.resume, device)
            model.module.load_state_dict(weight_dict)
            print_info_message('Done')
        else:
            print_warning_message('No file for resume. Please check.')

        
    for epoch in range(start_epoch, args.epochs):
        lr = lr_scheduler.step(epoch)
        for param_group in optimizer.param_groups:         
            param_group['lr'] = lr

        print_info_message(
            'Running epoch {} with learning rates: {:.6f}'.format(epoch, lr))
        miou_train, train_loss = train(model, train_loader, optimizer, criterion, seg_classes, epoch, device=device)
        miou_val, val_loss = val(model, val_loader, criterion, seg_classes, device=device)

        # remember best miou and save checkpoint
        is_best = miou_val > best_miou
        best_miou = max(miou_val, best_miou)

        weights_dict = model.module.state_dict() if device == 'cuda' else model.state_dict()
        save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.model,
                'state_dict': weights_dict,
                'best_miou': best_miou,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.savedir, extra_info_ckpt)
        if is_best:
            model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
            torch.save(weights_dict, model_file_name)
            print('weights saved in {}'.format(model_file_name))
        info = {
            'Segmentation/LR': round(lr, 6),
            'Segmentation/Loss/train': train_loss,
            'Segmentation/Loss/val': val_loss,
            'Segmentation/mIOU/train': miou_train,
            'Segmentation/mIOU/val': miou_val,
            'Segmentation/Complexity/Flops': best_miou,
            'Segmentation/Complexity/Params': best_miou,
            }        

        for tag, value in info.items():
            if tag == 'Segmentation/Complexity/Flops':
                my_logger.scalar_summary(tag, value, math.ceil(flops)) 
            elif tag == 'Segmentation/Complexity/Params':
                my_logger.scalar_summary(tag, value, math.ceil(num_params)) 
            else:
                my_logger.scalar_summary(tag, value, epoch + 1)   
                


        
    print_info_message("========== TRAINING FINISHED ===========")

    
if __name__ == "__main__":
    from commons.general_details import segmentation_models, segmentation_schedulers, segmentation_loss_fns, \
        segmentation_datasets

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch resume from')    
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--ignore_idx', type=int, default=255, help='Index or label to be ignored during training')
    parser.add_argument('--fp_train', action='store_true', default=False, help='Train model in FP mode')

    # dataset and result directories
    parser.add_argument('--dataset', type=str, default='pascal', choices=segmentation_datasets, help='Datasets')
    parser.add_argument('--data_path', type=str, default='', help='dataset path')
    parser.add_argument('--coco_path', type=str, default='', help='MS COCO dataset path')
    parser.add_argument('--savedir', type=str, default='./results_segmentation', help='Location to save the results')
    ## only for cityscapes
    parser.add_argument('--coarse', action='store_true', default=False, help='Want to use coarse annotations or not')
    
    # scheduler details
    parser.add_argument('--optimizer', default='QSGD', 
                        help='Optimizers')    
    parser.add_argument('--nesterov', action='store_true', default=True, help='nesterov')
    parser.add_argument('--amsgrad', action='store_true', default=False, help='amsgrad')    
    parser.add_argument('--clip_by', default=1e-3, type=float, help='clip_by')
    parser.add_argument('--noise_decay', default=1e-2, type=float, help='noise_decay') 
    parser.add_argument('--toss_coin', action='store_true', default=True, help='toss_coin')    
    parser.add_argument('--scheduler', default='hybrid', choices=segmentation_schedulers,
                        help='Learning rate scheduler (fixed, clr, poly)')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')  
    parser.add_argument('--step_size', default=51, type=int, help='steps at which lr should be decreased')
    parser.add_argument('--lr', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--lr_mult', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='factor by which lr should be decreased')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=4e-5, type=float, help='weight decay (default: 4e-5)')
    # for Polynomial LR
    parser.add_argument('--power', default=0.9, type=float, help='power factor for Polynomial LR')

    # for hybrid LR
    parser.add_argument('--clr_max', default=61, type=int, help='Max number of epochs for cylic LR before '
                                                                'changing last cycle to linear')
    parser.add_argument('--cycle_len', default=5, type=int, help='Duration of cycle')

    # input details
    parser.add_argument('--batch_size', type=int, default=40, help='list of batch sizes')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                        help='list of image crop sizes, with each item storing the crop size (should be a tuple).')
    parser.add_argument('--loss_type', default='ce', choices=segmentation_loss_fns, help='Loss function (ce or miou)')

    # model related params
    parser.add_argument('--s', type=float, default=2.0, help='Factor by which channels will be scaled')
    parser.add_argument('--model', default='espnetv2', choices=segmentation_models,
                        help='Which model? basic= basic CNN model, res=resnet style)')
    parser.add_argument('--channels', default=3, type=int, help='Input channels')
    parser.add_argument('--num_classes', default=1000, type=int,
                        help='ImageNet classes. Required for loading the base network')
    parser.add_argument('--model_width', default=224, type=int, help='Model width')
    parser.add_argument('--model_height', default=224, type=int, help='Model height')
    parser.add_argument('--fp_epochs', default=1, type=int, help='FP warmup epoch')   

    args = parser.parse_args()

    random.seed(1882)
    torch.manual_seed(1882)

    assert args.data_path != '', 'Dataset path is an empty string. Please check.'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    args.savedir = '{}/model_{}_{}/sch_{}_loss_{}/{}'.format(args.savedir, args.model, args.dataset, args.scheduler, args.loss_type, timestr)
    main(args)
