import torch

from torchflow.tool import *
from torchflow.environment import distributed


def recursive_register(ins, cls, func_name):
    if ins.__class__ == cls:
        f = getattr(cls, func_name, None)
        f(ins)

    supers = cls.__bases__

    for s in supers:
        recursive_register(ins, s, func_name)
        f = getattr(s, func_name, None)
        if callable(f):
            f(ins)


class FlowModule:
    def __init__(self, args=None):
        self.args = args

        self._flow = None
        self._flows = register.Register(self.__class__.__name__)

        if self.args is not None:
            # self._register_args()
            recursive_register(self, self.__class__, '_register_args')

        # self._register_flows()
        recursive_register(self, self.__class__, '_register_flows')

    def _register_args(self):
        pass

    def _register_flows(self):
        pass

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

        self._register_args()
        recursive_register(self, self.__class__, '_register_args')
        
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.modules = modules

        # MARK: to compatible with old code, set optimizers by 'register_optimizers'
        self.optimizers = None

        # tools
        self.meter = meter.MeanMeter()
        self.visualizer = {}

        # pre-processing
        for key in self.datasets.keys():
            self.datasets[key].set_flow(vars(self.args.dataset)[key].flow)

    def _register_args(self):
        self.args.interval = parser.fetch_arg(self.args.interval, 1)

    def prepare(self):
        pass

    def forward(self):
        pass

    def postprocess(self):
        pass
    
    def register_optimizers(self, optimizers):
        self.optimizers = optimizers

    def clear_optimizers(self):
        for name in self.optimizers:
            optimizer = self.optimizers[name]
            if optimizer is not None:
                optimizer.zero_grad()

    def run_optimizers(self):
        for name in self.optimizers:
            optimizer = self.optimizers[name]
            if optimizer is not None:
                optimizer.step()

    def load(self, dataloader):

        # in case if the flow of the dataset is changed by other flows
        for key in self.dataloaders.keys():
            if self.dataloaders[key] == dataloader:
                self.datasets[key].set_flow(vars(self.args.dataset)[key].flow)
                break

        batch = next(dataloader)

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(distributed.rank)
        return batch

    def visualization(self, dir, uid):
        pass
            
    def logging(self):
        for key in self.meter.keys():
            logger.log('  - {key}: {value:.6f}'.format(key=key, value=self.meter[key]))