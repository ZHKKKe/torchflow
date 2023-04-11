import torch

from torchflow.tool import parser
from torchflow import environment


# ---------------------------------------------------------------------
# Wrapper of Optimizer
# ---------------------------------------------------------------------

def sgd(args):
    """ Wrapper of torch.optim.SGD (PyTorch >= 1.0.0).

    Implements stochastic gradient descent (optionally with momentum).
    """
    args.lr = parser.fetch_arg(args.lr, 0.01)
    args.weight_decay = parser.fetch_arg(args.weight_decay, 0)
    args.momentum = parser.fetch_arg(args.momentum, 0)
    args.dampening = parser.fetch_arg(args.dampening, 0)
    args.nesterov = parser.fetch_arg(args.nesterov, False)

    def sgd_wrapper(param_groups):
        environment.pytorch_support(required_version='1.0.0', message='Optimizer - SGD')
        return torch.optim.SGD(
            param_groups, 
            lr=args.lr, momentum=args.momentum, dampening=args.dampening,
            weight_decay=args.weight_decay, nesterov=args.nesterov)

    return sgd_wrapper


def adam(args):
    """ Wrapper of torch.optim.Adam (PyTorch >= 1.0.0).

    Implements Adam algorithm.
    It has been proposed in 'Adam: A Method for Stochastic Optimization'.
    """

    args.lr = parser.fetch_arg(args.lr, 0.001)
    args.beta1 = parser.fetch_arg(args.beta1, 0.9)
    args.beta2 = parser.fetch_arg(args.beta2, 0.999)
    args.eps = parser.fetch_arg(args.eps, 1e-08)
    args.weight_decay = parser.fetch_arg(args.weight_decay, 0)

    def adam_wrapper(param_groups):
        environment.pytorch_support(required_version='1.0.0', message='Optimizer - Adam')
        return torch.optim.Adam(
            param_groups, 
            lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, 
            weight_decay=args.weight_decay)
        
    return adam_wrapper
