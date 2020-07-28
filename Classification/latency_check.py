import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
num_gpu = 0

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--config', type=str, default='./setting/evaluate.json', help='JSON file for configuration')

    args = parser.parse_args()

    ############### setting framework ##########################################
    with open(args.config) as fin:
        config = json.load(fin)
    test_config = config['test_config']
    data_config = config['data_config']

    args.optim = test_config["optim"]
    args.lrsch = test_config["lrsch"]
    args.warmup_epochs = test_config["warmup_epoch"]
    args.epochs = test_config["epochs"]
    args.weight_decay = test_config["weight_decay"]
    args.restart_epochs = test_config["restart_epoch"]
    args.batch_size = test_config["batch_size"]
    args.warmup_lr = test_config["warmup_lr"]
    args.learning_rate = test_config['learning_rate']
    args.anneal = test_config['annealing']
    args.noise_decay = test_config['noise_decay']    
    args.toss_coin = test_config["toss_coin"]
    args.clip_by = test_config['clip_by']
    args.nesterov = test_config['nesterov']
    args.amsgrad = test_config['amsgrad']
    args.lr = args.learning_rate
    others= args.weight_decay*0.01

        
    if not os.path.isdir(test_config['save_dir']):
        os.mkdir(test_config['save_dir'])

    print("Run : " + test_config["Model"])
    model_name = test_config["Model"]
    if data_config["dataset_name"].startswith('cifar'):
        model = models.cifar.__dict__[model_name](num_classes=data_config["num_classes"])
    else:
        model = models.imagenet.__dict__[model_name](num_classes=data_config["num_classes"])
      
    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    batch = torch.FloatTensor(1, 3, data_config["w"], data_config["h"])
    out = model_eval(batch)
    N_flop = model.compute_average_flops_cost()
    total_parameters = netParams(model)   
    
    print("num_classes: {}".format(data_config["num_classes"]))  
    print("total_parameters: {}".format(total_parameters))  
    
    dataset, dataset_test = download_data(data_config["dataset_name"],
                                          data_config["data_dir"])
    
    trainLoader, valLoader = prepare_data_loaders(dataset, dataset_test, 
                                                  args.batch_size, 1)
    args.dataset_len = len(dataset)//args.batch_size
    criteria = nn.CrossEntropyLoss()  
    
    params_dict = dict(model.named_parameters())
    params_set = []
    names_set = []      
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
                
    init_lr = test_config["learning_rate"]

    def print_size_of_model(model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')
        
    print('========== ORIGINAL MODEL SIZE ==========')
    print_size_of_model(model)
    with torch.no_grad():
        qat_lossVal, qat_acc1_val, qat_acc5_val = latency_val(num_gpu, valLoader, model, criteria)    
    print('========== QAT MODEL SIZE ==========')
    print_size_of_model(model)        
    model.fuse_model()
    model.qconfig =  torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(model, inplace=True)
    with torch.no_grad():
        qat_lossVal, qat_acc1_val, qat_acc5_val = latency_val(num_gpu, valLoader, model, criteria)
    torch.quantization.convert(model.eval(),inplace = True)
    print("========== QUANTIZED MODEL SIZE ==========")
    print_size_of_model(model)
    with torch.no_grad():
        lossVal, acc1_val, acc5_val = latency_val(num_gpu, valLoader, model, criteria)










