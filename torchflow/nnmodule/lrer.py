from functools import wraps

import torch

from torchflow.tool import parser
from torchflow import environment

""" This file wraps the learning rate schedulers used in the script.
"""


class WarmupWrapper:
    def __init__(self, lrer, optimizer):
        self.lrer = lrer
        self.optimizer = optimizer
        self.ori_step_func = lrer.step

    def step(self):
        if self.lrer.current_iter < self.lrer.warmup_iters:
            factor = self.lrer.start_factor + (self.lrer.end_factor - self.lrer.start_factor) * (
                self.lrer.current_iter / (self.lrer.warmup_iters - 1)
            )
            factor = min(1, factor)
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.lrer.base_lrs[i] * factor
        elif self.lrer.current_iter == self.lrer.warmup_iters:
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.lrer.base_lrs[i]
        else:
            self.ori_step_func()

        # for i, group in enumerate(self.optimizer.param_groups):
            # print(f' {self.lrer.current_iter} / {self.lrer.warmup_iters} - param_group {i}: lr = {group["lr"]}')

        self.lrer.current_iter += 1

    def __getattr__(self, attr):
        return getattr(self.lrer, attr)


def set_warmup(warmup_iters, start_factor, end_factor):
    def decorator(lrer_func):
        @wraps(lrer_func)
        def wrapper(optimizer, *args, **kwargs):
            lrer = lrer_func(optimizer, *args, **kwargs)
            lrer.current_iter = 0
            lrer.warmup_iters = warmup_iters
            lrer.start_factor = start_factor
            lrer.end_factor = end_factor
            lrer.base_lrs = [group["lr"] for group in optimizer.param_groups]
            return WarmupWrapper(lrer, optimizer)
        return wrapper
    return decorator


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
    args.warmup_iters = parser.fetch_arg(args.warmup_iters, 0)
    args.warmup_start_factor = parser.fetch_arg(args.warmup_start_factor, 0)
    args.warmup_end_factor = parser.fetch_arg(args.warmup_end_factor, 1)

    @set_warmup(warmup_iters=args.warmup_iters, start_factor=args.warmup_start_factor, end_factor=args.warmup_end_factor)
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
    args.warmup_iters = parser.fetch_arg(args.warmup_iters, 0)
    args.warmup_start_factor = parser.fetch_arg(args.warmup_start_factor, 0)
    args.warmup_end_factor = parser.fetch_arg(args.warmup_end_factor, 1)

    @set_warmup(warmup_iters=args.warmup_iters, start_factor=args.warmup_start_factor, end_factor=args.warmup_end_factor)
    @set_step_interval_iters(x=args.step_interval_iters)
    def multisteplr_wrapper(optimizer):
        environment.pytorch_support(required_version='1.0.0', message='LRScheduler - MultiStepLR')
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma, last_epoch=args.last_epoch)

    return multisteplr_wrapper


def cosineannealinglr(args):
    """ Wrapper of torch.optim.lr_scheduler.CosineAnnealingLR (PyTorch >= 2.0.0).

    Set the learning rate of each parameter group using a cosine annealing schedule.
    """
    args.T_max = parser.fetch_arg(args.T_max, required=True)
    args.eta_min = parser.fetch_arg(args.eta_min, 0)
    args.last_epoch = parser.fetch_arg(args.last_epoch, -1)
    args.step_interval_iters = parser.fetch_arg(args.step_interval_iters, 1)
    args.warmup_iters = parser.fetch_arg(args.warmup_iters, 0)
    args.warmup_start_factor = parser.fetch_arg(args.warmup_start_factor, 0)
    args.warmup_end_factor = parser.fetch_arg(args.warmup_end_factor, 1)

    args.T_max = args.T_max - args.warmup_iters

    @set_warmup(warmup_iters=args.warmup_iters, start_factor=args.warmup_start_factor, end_factor=args.warmup_end_factor)
    @set_step_interval_iters(x=args.step_interval_iters)
    def cosineannealinglr_wrapper(optimizer):
        environment.pytorch_support(required_version='2.0.0', message='LRScheduler - CosineAnnealingLR')
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.T_max, eta_min=args.eta_min, last_epoch=args.last_epoch)

    return cosineannealinglr_wrapper
