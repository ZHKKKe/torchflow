import torch

from torchflow.tool import logger
from . import distributed


__all__ = [
    'distributed',
    'pytorch_support',
    'force_cudnn_initialization',
]


def pytorch_support(required_version='1.0.0', message=''):
    if torch.__version__ < required_version:
        logger.info('{0} required PyTorch >= {1}\n'
                    'However, current PyTorch == {2}\n'
                    .format(message, required_version, torch.__version__))
        return False
    else:
        return True


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    torch.cuda.empty_cache()