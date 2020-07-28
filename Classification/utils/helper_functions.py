import datetime
import os
import time
import torch
import sys
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5

def load_model(model_file):
    model = MobileNetV2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model

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

def train(num_gpu, train_loader, model, criterion, optimizer, epoch, total_ep, args):
    '''

    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()

    mAcc1 = []
    mAcc5 = []
    epoch_loss = []

    total_time = 0
    total_batches = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        start_time = time.time()

        if num_gpu > 0:
            input = input.cuda()
            target = target.cuda()
            
        if args.lrsch == 'cos_lr':
            current_lr = adjust_learning_rate_cosine(optimizer, epoch, i, args.dataset_len, args)
        elif args.lrsch == 'linear_lr':
            current_lr = adjust_learning_rate_linear(optimizer, epoch, i, args.dataset_len, args)
        else:
            current_lr = adjust_learning_rate(optimizer, epoch, i, args.dataset_len, args)
            
            
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        if num_gpu > 0:
            input_var = input_var.cuda()
            target_var = target_var.cuda()
            
        optimizer.zero_grad()        
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        mAcc1.append(acc1.item())
        mAcc5.append(acc5.item())
        
        time_taken = time.time() - start_time
        total_time += time_taken
        # compute the confusion matrix
        progress_bar(i, len(train_loader), 'Loss: %.3f | Top1 Acc: %.3f | Top5 Acc: %.3f'
            % (sum(epoch_loss) / len(epoch_loss), sum(mAcc1) / len(mAcc1), sum(mAcc5) / len(mAcc5)))
        
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    print('[%d/%d] loss: %.3f time:%.2f Top1 acc(tr): %.3f Top5 acc(tr): %.3f' % (
            epoch + 1, total_ep, average_epoch_loss_train, 
            total_time / total_batches, 
            sum(mAcc1) / len(mAcc1), sum(mAcc5) / len(mAcc5)))

    return average_epoch_loss_train, sum(mAcc1) / len(mAcc1), sum(mAcc5) / len(mAcc5)


def train_one_iter(num_gpu, train_loader, model, criterion, optimizer, epoch, total_ep, args):
    '''

    :param train_loader: loaded for training dataset
    :param model: model
    :param criterion: loss function
    :param optimizer: optimization algo, such as ADAM or SGD
    :param epoch: epoch number
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to train mode
    model.train()

    mAcc1 = []
    mAcc5 = []
    epoch_loss = []

    total_time = 0
    total_batches = len(train_loader)
    for i, (input, target) in enumerate(train_loader):
        start_time = time.time()

        if num_gpu > 0:
            input = input.cuda()
            target = target.cuda()
            
        if args.lrsch == 'cos_lr':
            current_lr = adjust_learning_rate_cosine(optimizer, epoch, i, args.dataset_len, args)
        elif args.lrsch == 'linear_lr':
            current_lr = adjust_learning_rate_linear(optimizer, epoch, i, args.dataset_len, args)
        else:
            current_lr = adjust_learning_rate(optimizer, epoch, i, args.dataset_len, args)
            
            
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        if num_gpu > 0:
            input_var = input_var.cuda()
            target_var = target_var.cuda()
            
        optimizer.zero_grad()        
        output = model(input_var)
        loss = criterion(output, target_var)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        mAcc1.append(acc1.item())
        mAcc5.append(acc5.item())
        
        time_taken = time.time() - start_time
        total_time += time_taken
        # compute the confusion matrix
        progress_bar(i, len(train_loader), 'Loss: %.3f | Top1 Acc: %.3f | Top5 Acc: %.3f'
            % (sum(epoch_loss) / len(epoch_loss), sum(mAcc1) / len(mAcc1), sum(mAcc5) / len(mAcc5)))
        break
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    print('[%d/%d] loss: %.3f time:%.2f Top1 acc(tr): %.3f Top5 acc(tr): %.3f' % (
            epoch + 1, total_ep, average_epoch_loss_train, 
            total_time / total_batches, 
            sum(mAcc1) / len(mAcc1), sum(mAcc5) / len(mAcc5)))

    return average_epoch_loss_train, sum(mAcc1) / len(mAcc1), sum(mAcc5) / len(mAcc5)
def adjust_learning_rate_cosine(optimizer, epoch, iter, dataset_len, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.anneal:
        epoch = epoch%args.restart_epochs
        epochs = args.restart_epochs
        '''
        if args.optim.startswith('Q'):
            if epoch == 0:
                for param_group in optimizer.param_groups:
                    for p in param_group['params']:
                        state = optimizer.state[p]
                        if p.grad is None:
                            continue
                        state['restart_step'] = 0
       '''                
    else:
        epochs = args.epochs
    total_iter = (epochs - args.warmup_epochs) * dataset_len
    current_iter = iter + (epoch - args.warmup_epochs) * dataset_len

    """Warmup"""
    if epoch < args.warmup_epochs:
        lr = args.warmup_lr \
             + (args.lr - args.warmup_lr) * float(iter + epoch * dataset_len) / (args.warmup_epochs * dataset_len)
    else:
        lr = args.lr / 2 * (np.cos(np.pi * current_iter / total_iter) + 1)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def adjust_learning_rate_linear(optimizer, epoch, iter, dataset_len, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.anneal:
        epoch = epoch%args.restart_epochs        
        epochs = args.restart_epochs
        '''
        if args.optim.startswith('Q'):
            if epoch == 0:
                for param_group in optimizer.param_groups:
                    param_group['restart_step'] = 0
       '''             
    else:
        epochs = args.epochs    
    total_iter = (epochs - args.warmup_epochs) * dataset_len
    current_iter = iter + (epoch - args.warmup_epochs) * dataset_len

    """Warmup"""
    if epoch < args.warmup_epochs:
        lr = args.warmup_lr \
             + (args.lr - args.warmup_lr) * float(iter + epoch * dataset_len) / (args.warmup_epochs * dataset_len)
    else:
        lr = args.lr * (1 - current_iter / total_iter)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def adjust_learning_rate(optimizer, epoch, iter, dataset_len, args):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    """Warmup"""
    if epoch < args.warmup_epochs:
        lr = args.warmup_lr \
             + (args.lr - args.warmup_lr) * float(iter + epoch * dataset_len) / (args.warmup_epochs * dataset_len)
    else:
        lr = args.lr * (0.1 ** factor)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
def val(num_gpu, val_loader, model, criterion):
    '''
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to evaluation mode
    model.eval()
    total_time = 0
    epoch_loss = []
    total_batches = len(val_loader)
    mAcc1 = []
    mAcc5 = []
    for i, (input, target) in enumerate(val_loader):
        start_time = time.time()

        if num_gpu > 0:
            input = input.cuda()
            target = target.cuda()

        with torch.no_grad():
            input_var, target_var = torch.autograd.Variable(input),\
                                    torch.autograd.Variable(target)
            # run the model
            output = model(input_var)
            # compute the loss
            loss = criterion(output, target_var)
        epoch_loss.append(loss.item())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        mAcc1.append(acc1.item())
        mAcc5.append(acc5.item())

        time_taken = time.time() - start_time
        total_time += time_taken
        # compute the confusion matrix

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    print('loss: %.4f time:%.2f Top1 acc(val): %.2f Top5 acc(val): %.2f' % (
            average_epoch_loss_val, total_time / total_batches, 
            sum(mAcc1) / len(mAcc1),sum(mAcc5) / len(mAcc5)))
    
    return average_epoch_loss_val, sum(mAcc1) / len(mAcc1),sum(mAcc5) / len(mAcc5)


def latency_val(num_gpu, val_loader, model, criterion):
    '''
    :param val_loader: loaded for validation dataset
    :param model: model
    :param criterion: loss function
    :return: average epoch loss, overall pixel-wise accuracy, per class accuracy, per class iu, and mIOU
    '''
    # switch to evaluation mode
    model.eval()
    total_time = 0
    epoch_loss = []
    total_batches = 100
    mAcc1 = []
    mAcc5 = []
    for i, (input, target) in enumerate(val_loader):
        if num_gpu > 0:
            input = input.cuda()
            target = target.cuda()

        with torch.no_grad():
            input_var, target_var = torch.autograd.Variable(input),\
                                    torch.autograd.Variable(target)
            # run the model
            start_time = time.time()            
            output = model(input_var)
            time_taken = time.time() - start_time
            total_time += time_taken            
            # compute the loss
            loss = criterion(output, target_var)
        epoch_loss.append(loss.item())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        mAcc1.append(acc1.item())
        mAcc5.append(acc5.item())

        if i >= total_batches: 
            break
        # compute the confusion matrix

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)

    print('loss: %.4f time:%.4f Top1 acc(val): %.2f Top5 acc(val): %.2f' % (
            average_epoch_loss_val, total_time / total_batches, 
            sum(mAcc1) / len(mAcc1),sum(mAcc5) / len(mAcc5)))
    
    return average_epoch_loss_val, sum(mAcc1) / len(mAcc1),sum(mAcc5) / len(mAcc5)

def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    '''
    helper function to save the checkpoint
    :param state: model state
    :param filenameCheckpoint: where to save the checkpoint
    :return: nothing
    '''
    torch.save(state, filenameCheckpoint)


def netParams(model):
    '''
    helper function to see total network parameters
    :param model: model
    :return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p
        # print(total_paramters)

    return total_paramters


def info_setting(save_dir, model_name, Nparam, Flop, optim):
    now = datetime.datetime.now()
    time_str = now.strftime("%m-%d_%H%M")
    this_savedir = os.path.join(save_dir, model_name+time_str+'_'+optim)
    if not os.path.isdir(this_savedir):
        os.mkdir(this_savedir)

    logFileLoc = this_savedir + "/trainValLog.txt"

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(Nparam)))
        logger.write("FLOP: %s" % (str(Flop)))

        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % (
            'Epoch', 'Loss(Tr)', 'Loss(val)', 'top1_acc(tr)','top5_acc(tr)','top1_acc(val)','top5_acc(val)', 'lr', 'time'))
    return logger, this_savedir



def colormap_cityscapes(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([128, 64, 128])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([70, 70, 70])
    cmap[3, :] = np.array([102, 102, 156])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])

    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([107, 142, 35])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])

    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([255, 0, 0])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])

    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([119, 11, 32])
    cmap[19, :] = np.array([0, 0, 0])

    return cmap

class Colorize:

    def __init__(self, n=22):
        #self.cmap = colormap(256)
        self.cmap = colormap_cityscapes(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        #print(size)
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        #color_image = torch.ByteTensor(3, size[0], size[1]).fill_(0)

        #for label in range(1, len(self.cmap)):
        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            #mask = gray_image == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
    
term_width = 80   
TOTAL_BAR_LENGTH = 20.
last_time = time.time()
begin_time = last_time  

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f    