import torch

from torchflow.tool import parser
from torchflow import environment

""" This file wraps the learning rate schedulers used in the script.
"""


def set_step_interval_iters(x):
    def decorator(func):
        def wrapper(*args, **kwargs):
            _optimizer = func(*args, **kwargs)
            setattr(_optimizer, 'step_interval_iters', x)
            return _optimizer
        return wrapper
    return decorator


# ---------------------------------------------------------------------
# Wrapper of Learning Rate Scheduler
# ---------------------------------------------------------------------

def steplr(args):
    """ Wrapper of torch.optim.lr_scheduler.StepLR (PyTorch >= 1.0.0).

    Sets the learning rate of each parameter group to the initial lr decayed by gamma every 
    step_size epochs. When last_epoch=-1, sets initial lr as lr.
    """
    args.step_size = parser.fetch_arg(args.step_size, 1)
    args.gamma = parser.fetch_arg(args.gamma, 0.1)
    args.last_epoch = parser.fetch_arg(args.last_epoch, -1)
    args.step_interval_iters = parser.fetch_arg(args.step_interval_iters, 1)

    @set_step_interval_iters(x=args.step_interval_iters)
    def steplr_wrapper(optimizer):
        environment.pytorch_support(required_version='1.0.0', message='LRScheduler - StepLR')
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=args.last_epoch)
    
    return steplr_wrapper


def multisteplr(args):
    """ Wrapper of torch.optim.lr_scheduler.MultiStepLR (PyTorch >= 1.0.0).

    Set the learning rate of each parameter group to the initial lr decayed by gamma once the 
    number of epoch reaches one of the milestones. When last_epoch=-1, sets initial lr as lr.
    """
    args.milestones = parser.fetch_arg(args.milestones, [])
    args.gamma = parser.fetch_arg(args.gamma, 0.1)
    args.last_epoch = parser.fetch_arg(args.last_epoch, -1)
    args.step_interval_iters = parser.fetch_arg(args.step_interval_iters, 1)

    @set_step_interval_iters(x=args.step_interval_iters)
    def multisteplr_wrapper(optimizer):
        environment.pytorch_support(required_version='1.0.0', message='LRScheduler - MultiStepLR')
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma, last_epoch=args.last_epoch)

    return multisteplr_wrapper
