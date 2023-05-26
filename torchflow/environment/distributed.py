import os
import torch


# NOTE: distributed global variables
rank = None
world_size = None


# TODO: define all input arguments as global variables?
def init_process_group(backend, world_size, rank, master_addr, master_port):
    
    if world_size > 1:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        torch.distributed.init_process_group(
            backend, world_size=world_size, rank=rank, group_name='torchflow')


def barrier():
    if world_size > 1:
        torch.distributed.barrier()
