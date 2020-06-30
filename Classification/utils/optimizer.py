import math
import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

def get_optimizer(optim, params_set, args):

    if optim == "SGD":
        optimizer = torch.optim.SGD(params_set, args.learning_rate, 
                                    momentum=0.9, weight_decay=args.weight_decay, 
                                    nesterov=args.nesterov)
    elif optim == "RMS":
        optimizer = torch.optim.RMSprop(params_set, args.learning_rate, 
                                        alpha=0.9, momentum=0.9, eps=1e-8, 
                                        weight_decay=args.weight_decay)
    elif optim == "Adam":
        optimizer = torch.optim.Adam(params_set, args.learning_rate, 
                                     betas=(0.9, 0.999), eps=1e-08, 
                                     weight_decay=args.weight_decay)        
    elif optim == "AdamW":
        optimizer = torch.optim.AdamW(params_set, args.learning_rate, 
                                      betas=(0.9, 0.999), eps=1e-08,
                                      weight_decay=args.weight_decay,
                                      amsgrad=args.amsgrad)
    elif optim == "QSGD":
        optimizer = QSGD(params_set, args.learning_rate, 
                         momentum=0.9,weight_decay=args.weight_decay, 
                         nesterov=args.nesterov, clip_by = args.clip_by, 
                         toss_coin = args.toss_coin, noise_decay=args.noise_decay)
    elif optim == "QRMS":
        optimizer = QRMSprop(params_set, args.learning_rate, 
                             alpha=0.9, momentum=0.9, eps=1e-8, 
                             weight_decay=args.weight_decay, 
                             clip_by = args.clip_by, toss_coin = args.toss_coin,
                             noise_decay=args.noise_decay)     
    elif optim == "QAdam":
        optimizer = QAdam(params_set, args.learning_rate, 
                           betas=(0.9, 0.999), eps=1e-08, 
                           weight_decay=args.weight_decay,
                           amsgrad=args.amsgrad, clip_by = args.clip_by, 
                           toss_coin = args.toss_coin,  noise_decay=args.noise_decay)   
    elif optim == "QAdamW":
        optimizer = QAdamW(params_set, args.learning_rate, 
                           betas=(0.9, 0.999), eps=1e-08, 
                           weight_decay=args.weight_decay,
                           amsgrad=args.amsgrad, clip_by = args.clip_by, 
                           toss_coin = args.toss_coin, noise_decay=args.noise_decay)      
    return optimizer

class QSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v_{t+1} = \mu * v_{t} + g_{t+1} \\
                  p_{t+1} = p_{t} - lr * v_{t+1}

        where p, g, v and :math:`\mu` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v_{t+1} = \mu * v_{t} + lr * g_{t+1} \\
             p_{t+1} = p_{t} - v_{t+1}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, beta = 0.9, eps=1e-8, 
                 clip_by = 1e-3, toss_coin = True, noise_decay = 1e-2):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, beta=beta, 
                        eps=eps, clip_by = clip_by, toss_coin = toss_coin, 
                        noise_decay = noise_decay)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.is_warmup = True
        super(QSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(QSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # State initialization
             
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            toss_coin = group['toss_coin']
            noise_decay = group['noise_decay'] 

            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    continue
                if len(state) == 0:
                    state['step'] = 0
                    state['restart_step'] = 0
                    state['exp_min'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_max'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) 
                    if toss_coin:
                        state['coin_toss'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)                    
                exp_min, exp_max = state['exp_min'], state['exp_max']
                if toss_coin:                
                    coin_toss = state['coin_toss']                   
                grad = p.grad.data
                beta1 = group['beta']
                clip_by = group['clip_by']

                
                state['step'] += 1             
                bias_correction1 = 1 - beta1 ** state['step']
                
                if self.is_warmup:
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1,new_min).div_(bias_correction1)        
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1,new_max).div_(bias_correction1) 
                    
                else:
                    state['restart_step'] +=1
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1,new_min).div_(bias_correction1)        
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1,new_max).div_(bias_correction1) 
                    noise_scale = (1 - noise_decay) ** state['restart_step']
                    grad_sensitivity = (exp_max - exp_min) * noise_scale
                    noise = np.random.laplace(0.0, 1.0, p.data.size())
                    noise = np.abs(noise)
                    noise = torch.from_numpy(noise).float().cuda() 
                    sign = grad.sign()
                    noise.mul_(grad_sensitivity)
                    if toss_coin:
                        coin_toss.random_(2)                
                        noise.mul_(coin_toss)
                    noise.mul_(sign)
                    if clip_by > 0.0:
                        noise.clamp_(-clip_by,clip_by)
                    grad.add_(noise)
                            
                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, grad)
                    if nesterov:
                        grad = grad.add(momentum, buf)
                    else:
                        grad = buf 
                p.data.add_(-group['lr'], grad)

        return loss

class QRMSprop(Optimizer):
    r"""Implements QRMSprop algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, 
                 centered=False,  beta = 0.9, clip_by = 1e-3,
                 toss_coin = True,  noise_decay = 1e-2):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered,
                        weight_decay=weight_decay, beta=beta,
                        clip_by = clip_by, toss_coin = toss_coin,
                        noise_decay = noise_decay)
        self.is_warmup = True
        super(QRMSprop, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(QRMSprop, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]
                toss_coin = group['toss_coin'] 
                noise_decay = group['noise_decay']
            
                # State initialization
                if len(state) == 0:
                    state['step'] = 0  
                    state['restart_step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['step'] = 0
                    state['exp_min'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_max'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)   
                    if toss_coin:
                        state['coin_toss'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                exp_min, exp_max = state['exp_min'], state['exp_max']
                if toss_coin:                
                    coin_toss = state['coin_toss']                 
                beta1 = group['beta']
                clip_by = group['clip_by']
             
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                
                if self.is_warmup:
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1,new_min).div_(bias_correction1)        
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1,new_max).div_(bias_correction1) 
                    
                else:
                    state['restart_step'] +=1
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1,new_min).div_(bias_correction1)        
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1,new_max).div_(bias_correction1) 
                    noise_scale = (1 - noise_decay) ** state['restart_step']
                    grad_sensitivity = (exp_max - exp_min) * noise_scale
                    noise = np.random.laplace(0.0, 1.0, p.data.size())
                    noise = np.abs(noise)
                    noise = torch.from_numpy(noise).float().cuda() 
                    sign = grad.sign()
                    noise.mul_(grad_sensitivity)
                    if toss_coin:
                        coin_toss.random_(2)                
                        noise.mul_(coin_toss)
                    noise.mul_(sign)
                    if clip_by > 0.0:
                        noise.clamp_(-clip_by,clip_by)
                    grad.add_(noise)         
            
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt_().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss 
    
class QAdam(Optimizer):
    r"""Implements AdamW algorithm.
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, 
                 clip_by = 1e-3, toss_coin = True, 
                 noise_decay = 1e-2):

        #sigma slightly defines maximum amount of noise.
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, 
                        clip_by = clip_by, toss_coin = toss_coin,
                        noise_decay = noise_decay)
        self.is_warmup = True
        super(QAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(QAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                toss_coin = group['toss_coin'] 
                noise_decay = group['noise_decay']    

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0 
                    state['restart_step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    
                    state['exp_min'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_max'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) 
                    if toss_coin:
                        state['coin_toss'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq  = state['exp_avg'], state['exp_avg_sq']
                exp_min, exp_max = state['exp_min'], state['exp_max']
                if toss_coin:                
                    coin_toss = state['coin_toss']    
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                clip_by = group['clip_by']
                    
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                    
                if self.is_warmup:
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1,new_min).div_(bias_correction1)        
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1,new_max).div_(bias_correction1) 
                    
                else:
                    state['restart_step'] +=1
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1,new_min).div_(bias_correction1)        
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1,new_max).div_(bias_correction1) 
                    noise_scale = (1 - noise_decay) ** state['restart_step']
                    grad_sensitivity = (exp_max - exp_min) * noise_scale
                    noise = np.random.laplace(0.0, 1.0, p.data.size())
                    noise = np.abs(noise)
                    noise = torch.from_numpy(noise).float().cuda() 
                    sign = grad.sign()
                    noise.mul_(grad_sensitivity)
                    if toss_coin:
                        coin_toss.random_(2)                
                        noise.mul_(coin_toss)
                    noise.mul_(sign)
                    if clip_by > 0.0:
                        noise.clamp_(-clip_by,clip_by)
                    grad.add_(noise)         
                    
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
    
class QAdamW(Optimizer):
    r"""Implements AdamW algorithm.
    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, clip_by = 1e-3,
                 toss_coin = True,noise_decay = 1e-2):

        #sigma slightly defines maximum amount of noise.
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,weight_decay=weight_decay, 
                        amsgrad=amsgrad, clip_by = clip_by, 
                        toss_coin = toss_coin,
                        noise_decay = noise_decay)
        self.is_warmup = True
        super(QAdamW, self).__init__(params, defaults)

        
    def __setstate__(self, state):
        super(QAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue
                                
                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                toss_coin = group['toss_coin']
                noise_decay = group['noise_decay']
 
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['restart_step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    
                    state['exp_min'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_max'] = torch.zeros_like(p.data, memory_format=torch.preserve_format) 
                    if toss_coin:
                        state['coin_toss'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq  = state['exp_avg'], state['exp_avg_sq']
                exp_min, exp_max = state['exp_min'], state['exp_max']
                
  
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                clip_by = group['clip_by']
                if toss_coin:                
                    coin_toss = state['coin_toss'] 
                    
                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                if self.is_warmup:
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1,new_min).div_(bias_correction1)        
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1,new_max).div_(bias_correction1) 
                    
                else:
                    state['restart_step'] +=1
                    new_min = torch.min(exp_min, torch.abs(grad))
                    exp_min.mul_(beta1).add_(1 - beta1,new_min).div_(bias_correction1)        
                    new_max = torch.max(exp_max, torch.abs(grad))
                    exp_max.mul_(beta1).add_(1 - beta1,new_max).div_(bias_correction1) 
                    noise_scale = (1 - noise_decay) ** state['restart_step']
                    grad_sensitivity = (exp_max - exp_min) * noise_scale
                    noise = np.random.laplace(0.0, 1.0, p.data.size())
                    noise = np.abs(noise)
                    noise = torch.from_numpy(noise).float().cuda() 
                    sign = grad.sign()
                    noise.mul_(grad_sensitivity)
                    if toss_coin:
                        coin_toss.random_(2)                
                        noise.mul_(coin_toss)
                    noise.mul_(sign)
                    if clip_by > 0.0:
                        noise.clamp_(-clip_by,clip_by)
                    grad.add_(noise)            
                    
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
