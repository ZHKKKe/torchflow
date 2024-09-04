import os
import torch


# NOTE: distributed global variables
rank = None
local_rank = None
world_size = None
backend = None
master_addr = None
master_port = None


def init_process_group():
    if master_addr is not None:
        os.environ['MASTER_ADDR'] = master_addr
    if master_port is not None:
        os.environ['MASTER_PORT'] = master_port

    torch.distributed.init_process_group(
        backend, 
        init_method='env://',
        world_size=world_size, 
        rank=rank
    )

    print('Initialized process group: rank={0}, world_size={1}'.format(rank, world_size))


def barrier():
    if world_size > 1:
        torch.distributed.barrier()
