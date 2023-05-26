import os
import torch

from .tool import *
from .template import *
from .worker import *
from .environment import *

from .version import __version__, __name__, __description__, __url__, \
    __license__, __author__, __author_email__, __updates__


def _processing(rank, config, dataset_dict, module_dict, flow_dict):
    if rank != 0:
        logger.mode(logger.MODE.CRITICAL)

    args = parser.parse_args(config)

    args.env.rank = rank
    args.env.config = config.split(os.sep)[-1]
    args.env.backend = parser.fetch_arg(args.env.backend, 'nccl')
    args.env.world_size = parser.fetch_arg(args.env.world_size, 1)
    args.env.master_addr = parser.fetch_arg(args.env.master_addr, 'localhost')
    args.env.master_port = str(parser.fetch_arg(args.env.master_port, '10086'))

    args.env.find_unused_parameters = parser.fetch_arg(args.env.find_unused_parameters, False)
    args.env.broadcast_buffers = parser.fetch_arg(args.env.broadcast_buffers, False)

    distributed.rank = args.env.rank
    distributed.world_size = args.env.world_size
    distributed.init_process_group(
        args.env.backend, args.env.world_size, args.env.rank, args.env.master_addr, args.env.master_port)
    distributed.barrier()

    Proxy(args, dataset_dict, module_dict, flow_dict).execute()


# TODO: support `torch.distributed.launch`

def run(config, dataset_dict, module_dict, flow_dict):
    pytorch_support(required_version='1.0.0', message=__name__)

    args = parser.parse_args(config)
    args.env.world_size = parser.fetch_arg(args.env.world_size, 1)

    if args.env.world_size == 1:
        logger.info('Start single process (world_size={0})\n'.format(args.env.world_size))
        _processing(0, config, dataset_dict, module_dict, flow_dict)
    
    else:
        logger.info(
            'Start multiple processes (world_size={0})\n'
            'Subsequent logs are printed by the master process with rank=0\n'.format(args.env.world_size))

        torch.multiprocessing.spawn(
            _processing,
            args=(config, dataset_dict, module_dict, flow_dict),
            nprocs=args.env.world_size,
            join=True
        )