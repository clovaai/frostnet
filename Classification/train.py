
import random
import os
import time
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import models
from utils import *
from utils.utils import *
from utils.Tensor_logger import Logger
from utils.data_functions import *
from utils.helper_functions import *
from utils.optimizer import *
from utils.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
import torch.quantization

#replacing original torch.quantization.disable_observer because of error shooting issue.
from torch.quantization.fake_quantize import FakeQuantize

def disable_observer(mod):
    if type(mod) == FakeQuantize:
        if mod.scale is not None and mod.zero_point is not None:
            mod.disable_observer()
        else:
            pass
            #print("Can't disable {} : scale or zero_point is None.".format(mod))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='./setting/train.json', help='JSON file for configuration')
    random.seed(1882)
    torch.manual_seed(1882)
    
    num_gpu = torch.cuda.device_count()
    print("Number of gpu : %d" % num_gpu)
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    ############### setting framework ##########################################
    with open(args.config) as fin:
        config = json.load(fin)
    train_config = config['train_config']
    data_config = config['data_config']

    args.optim = train_config["optim"]
    args.lrsch = train_config["lrsch"]
    args.warmup_epochs = train_config["warmup_epoch"]
    args.epochs = train_config["epochs"]
    args.weight_decay = train_config["weight_decay"]
    args.restart_epochs = train_config["restart_epoch"]
    args.batch_size = train_config["batch_size"]
    args.warmup_lr = train_config["warmup_lr"]
    args.learning_rate = train_config['learning_rate']
    args.anneal = train_config['annealing']
    args.noise_decay = train_config['noise_decay']    
    args.toss_coin = train_config["toss_coin"]
    args.clip_by = train_config['clip_by']
    args.nesterov = train_config['nesterov']
    args.amsgrad = train_config['amsgrad']
    args.lr = args.learning_rate
    others= args.weight_decay*0.01

    if not os.path.isdir(train_config['save_dir']):
        os.mkdir(train_config['save_dir']) 
        

    model_name = train_config["Model"]
    
    if data_config["dataset_name"].startswith('cifar'):
        model = models.cifar.__dict__[model_name](num_classes=data_config["num_classes"])
    else:
        model = models.imagenet.__dict__[model_name](num_classes=data_config["num_classes"], pretrained=True)
        
    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    batch = torch.FloatTensor(1, 3, data_config["w"], data_config["h"])
    out = model_eval(batch)
    N_flop = model.compute_average_flops_cost()
    total_parameters = netParams(model)           
    print("num_classes: {}".format(data_config["num_classes"]))  
    print("total_parameters: {}".format(total_parameters))     
    if use_cuda:
        if num_gpu > 1:
            model = torch.nn.DataParallel(model)
            print("make DataParallel")
        model = model.cuda()

    start_epoch = 0
    Max_val_acc1 = 0.0
    Max_val_acc5 = 0.0    
    Max_name = ''

    data_dir = data_config['data_dir']
    logger, this_savedir = info_setting(train_config['save_dir'], 
                                        model_name, total_parameters, 
                                        N_flop, args.optim)
    logger.flush()
    logdir = this_savedir.split(train_config['save_dir'])[1]

    my_logger = Logger(8097, './logs/' + logdir)
    
    dataset, dataset_test = download_data(data_config["dataset_name"],
                                          data_config["data_dir"])
    
    args.dataset_len = len(dataset)//args.batch_size
    
    trainLoader, valLoader = prepare_data_loaders(dataset, dataset_test, 
                                                  args.batch_size, args.batch_size)

    criteria = nn.CrossEntropyLoss()
   
    if num_gpu > 0:
        criteria = criteria.cuda()

    params_set = []
    names_set = []    

    if num_gpu > 1:
        params_dict = dict(model.module.named_parameters())
    else:
        params_dict = dict(model.named_parameters())
        
    for key, value in params_dict.items():
        if len(value.data.shape) == 4:
            if value.data.shape[1] == 1:
                params_set += [{'params': [value], 'weight_decay': 0.0}]
                # names_set.append(key)
            else:
                params_set += [{'params': [value], 'weight_decay': args.weight_decay}]
        else:
            params_set += [{'params': [value], 'weight_decay': others}]
    
    
    optimizer = get_optimizer(args.optim, params_set, args)  
                
    init_lr = train_config["learning_rate"]
       
    print("init_lr: " + str(train_config["learning_rate"]) + "   batch_size : " + str(args.batch_size)
          +' '+args.lrsch + "_sch")
    print("logs saved in " + logdir + "\tlr sch: " + args.lrsch + "\toptim method: " + args.optim 
           + "\tbn-weight : " + str(others))
 
    if train_config["warmup_epoch"] > 0:
        print("========== MODEL FP WARMUP ===========")

        for epoch in range(start_epoch, train_config["FP_epoch"]):    
            lr = 0             
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Learning rate: " + str(lr))
            start_t = time.time()            
            lossTr, acc1_tr, acc5_tr = train(num_gpu, trainLoader, 
                                    model, criteria, optimizer, 
                                    epoch, train_config["FP_epoch"], args)
            
    if args.optim.startswith('Q'):
        optimizer.is_warmup = False
        print('exp_sensitivity calibration fin.')
             
    if num_gpu > 1:
        model.module.fuse_model()
        model.module.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(model.module, inplace=True)
    else:
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        torch.quantization.prepare_qat(model, inplace=True)
        

    print("========== MODEL TRAINING ===========")

    for epoch in range(start_epoch, train_config["epochs"]):    
        lr = 0             
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print("Learning rate: " + str(lr))
        
        start_t = time.time()            
        lossTr, acc1_tr, acc5_tr = train(num_gpu, trainLoader, 
                                model, criteria, optimizer, 
                                epoch, train_config["epochs"], args )
        
        lossVal, acc1_val, acc5_val = val(num_gpu, valLoader, model, criteria)
        end_t = time.time()

        # save the model also
        if num_gpu > 1:
            this_state_dict = model.module.state_dict()
        else:
            this_state_dict = model.state_dict()

        if (Max_val_acc1 < acc1_val):
            model_file_name = this_savedir + '/model_' + str(epoch + 1) + '.pth'

            torch.save(this_state_dict, model_file_name)
            Max_val_acc1 = acc1_val
            Max_val_acc5 = acc5_val            
            Max_name = model_file_name
            print("new max accuracy : {} Top1: {:.4f} Top5: {:.4f}".format(Max_name, Max_val_acc1, Max_val_acc5)) 

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f\t\t%.2f" % (
            epoch+1, lossTr, lossVal, acc1_tr, acc5_tr, acc1_val, acc5_val, lr, (end_t - start_t)))
        logger.flush()
        print("Epoch : " + str(epoch+1) + ' Details')
        print("Epoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t top1_acc(tr) = %.4f\t top5_acc(tr) = %.4f\t top1_acc(val) = %.4f\t top5_acc(val) = %.4f \n" % (
            epoch+1, lossTr, lossVal, acc1_tr, acc5_tr, acc1_val, acc5_val))

        save_checkpoint({
            'epoch': epoch + 1, 'arch': str(model),
            'state_dict': this_state_dict,
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr, 'lossVal': lossVal,
            'acc1Tr': acc1_tr, 'acc1Val': acc1_val,
            'acc5Tr': acc5_tr, 'acc5Val': acc5_val,            
            'lr': lr,
            'Max_name': Max_name, 'Max_val_top1_acc': Max_val_acc1,'Max_val_top5_acc': Max_val_acc5
        }, this_savedir + '/checkpoint.pth.tar')
        
        info = {
                'train_loss': lossTr,
                'val_loss': lossVal,
                'train_top1_acc': acc1_tr,
                'train_top5_acc': acc5_tr,            
                'val_top1_acc': acc1_val,
                'val_top5_acc': acc5_val,            
                'lr': lr
            }

        for tag, value in info.items():
            my_logger.scalar_summary(tag, value, epoch + 1)

    logger.close()
    print("========== TRAINING FINISHED ===========")
    print("max accuracy : {} Top1: {:.4f} Top5: {:.4f}".format(Max_name, Max_val_acc1,Max_val_acc5))





