import torch

from torchflow.tool import *
from torchflow.environment import distributed


class FlowModule:
    def __init__(self):
        self._flow = None
        self._flows = register.Register(self.__class__.__name__)

        self._register_flows()
        self._register_flow(self.forward)
        self.set_flow('forward')

    def _register_flows(self):
        raise NotImplementedError

    def _register_flow(self, target):
        self._flows.register(target)

    def set_flow(self, name):
        if name in self._flows.keys():
            self._flow = self._flows[name]
        else:
            # TODO: raise error
            self._flow = None

    def flow(self, *input, **kwargs):
        if self._flow is None:
            return None
        else:
            return self._flow(*input, **kwargs)


class Flow(torch.nn.Module):
    def __init__(self, args, datasets, dataloaders, modules):
        super().__init__()
        self.args = args
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.modules = modules

        # tools
        self.meter = meter.MeanMeter()
        self.visualizer = {}

        # pre-processing
        for key in self.datasets.keys():
            self.datasets[key].set_flow(vars(self.args.dataset)[key].flow)

    def forward(self):
        pass

    def load(self, dataloader):
        batch = next(dataloader)
        for key in batch:
            batch[key] = batch[key].to(distributed.rank)
        return batch

    def visualization(self, dir, uid):
        pass
            
    def logging(self):
        for key in self.meter.keys():
            logger.log('  - {key}: {value:.6f}'.format(key=key, value=self.meter[key]))