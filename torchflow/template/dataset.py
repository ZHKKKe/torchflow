import copy
import argparse
import numpy as np
import torch

from .flow import FlowModule
from torchflow.tool import parser
from torchflow.environment import distributed


class Dataset(torch.utils.data.Dataset, FlowModule):
    def __init__(self, args=None):
        
        if args is not None:
            args = argparse.Namespace(**args)

        torch.utils.data.Dataset.__init__(self)
        FlowModule.__init__(self, args)

        items = []
        for dir in self.args.dirs:
           items += self._load(dir)
        self.items = np.array(items)
        del items

        self.indexes = np.array([_ for _ in range(0, self.__len__())])

    def _item2device(func):
        def wrapper(self, *args, **kwargs):
            item = func(self, *args, **kwargs)
            for key in item:
                item[key] = item[key].to(distributed.rank)
            return item
        return wrapper

    def __len__(self):
        return len(self.items)
    
    # @_item2device
    def __getitem__(self, index):
        assert index < self.__len__()
        item = copy.deepcopy(self.items[index])
        _item = self.flow(item)
        del item
        item = copy.deepcopy(_item)
        del _item
        _item = None
        
        if item is None:
            raise NotImplementedError
        else:
            return item

    def _load(self, dir):
        raise NotImplementedError

    def _register_args(self):
        self.args.dirs = parser.fetch_arg(self.args.dirs, [''])

    def _register_flows(self):
        pass

    # def forward(self, index):
    #     return {'index': torch.FloatTensor([index])}