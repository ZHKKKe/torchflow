import torch


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __init__(
        self, module, device_ids=None, output_device=None, 
        dim=0, broadcast_buffers=True, process_group=None, 
        bucket_cap_mb=25, find_unused_parameters=False, 
        check_reduction=False, gradient_as_bucket_view=False, 
        static_graph=False):

        super().__init__(
            module, device_ids, output_device, 
            dim, broadcast_buffers, process_group, 
            bucket_cap_mb, find_unused_parameters, 
            check_reduction, gradient_as_bucket_view, 
            static_graph)

    def set_flow(self, name):
        self.module.set_flow(name)

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def optimizable_parameters(self):
        return self.module.optimizable_parameters()
