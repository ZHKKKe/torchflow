import os
import random
import torch


rank = 0
world_size = 1


def init_process_group(
    backend, world_size, rank, master_addr='localhost', master_port=str(10000+random.randint(0, 10000))):
    
    if world_size > 1:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        torch.distributed.init_process_group(
            backend, world_size=world_size, rank=rank, group_name='torchflow')


def barrier():
    if world_size > 1:
        torch.distributed.barrier()
