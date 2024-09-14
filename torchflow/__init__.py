import os
import torch

from .tool import *
from .template import *
from .worker import *
from .environment import *

from .version import __version__, __name__, __description__, __url__, \
    __license__, __author__, __author_email__, __updates__


def _run(config, dataset_dict, module_dict, flow_dict):
    force_cudnn_initialization()

    args = parser.parse_args(config)

    args.env.config = config.split(os.sep)[-1]

    args.env.rank = 0 if os.environ.get('RANK') is None else int(os.environ['RANK'])
    args.env.local_rank = 0 if os.environ.get('LOCAL_RANK') is None else int(os.environ['LOCAL_RANK'])
    args.env.world_size = 1 if os.environ.get('WORLD_SIZE') is None else int(os.environ['WORLD_SIZE'])

    args.env.backend = parser.fetch_arg(args.env.backend, 'nccl')
    args.env.master_addr = parser.fetch_arg(args.env.master_addr, None)
    args.env.master_port = str(parser.fetch_arg(args.env.master_port, None))

    args.env.find_unused_parameters = parser.fetch_arg(args.env.find_unused_parameters, False)
    args.env.broadcast_buffers = parser.fetch_arg(args.env.broadcast_buffers, False)

    args.env.allow_tf32 = parser.fetch_arg(args.env.allow_tf32, False)

    if args.env.local_rank == 0:
        logger.mode(logger.MODE.INFO)
    else:
        logger.mode(logger.MODE.CRITICAL)

    if args.env.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    distributed.rank = args.env.rank
    distributed.local_rank = args.env.local_rank
    distributed.world_size = args.env.world_size
    distributed.backend = args.env.backend
    distributed.master_addr = args.env.master_addr
    distributed.master_port = args.env.master_port
    distributed.init_process_group()
    distributed.barrier()

    Proxy(args, dataset_dict, module_dict, flow_dict).execute()


def run(config, dataset_dict, module_dict, flow_dict):
    pytorch_support(required_version='2.0.0', message=__name__)

    # Add internal module classes
    module_dict['TorchScriptModule'] = TorchScriptModule

    _run(config, dataset_dict, module_dict, flow_dict)
