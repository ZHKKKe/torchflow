import torch

from torchflow.tool import parser
from .flow import FlowModule


class Module(torch.nn.Module, FlowModule):
    def __init__(self, args=None):
        torch.nn.Module.__init__(self)
        FlowModule.__init__(self, args)

        self.set_flow('forward')

    def __call__(self, *input, flow=None, **kwargs):
        if flow is not None:
            self.set_flow(flow)
        return self.flow(*input, **kwargs)

    def _register_args(self):
        self.args.name = parser.fetch_arg(self.args.name, None)
        self.args.print_params = parser.fetch_arg(self.args.print_params, False)
        self.args.initialization = parser.fetch_arg(self.args.initialization, None)
        self.args.strict_initialization = parser.fetch_arg(self.args.strict_initialization, True)

    def _register_flows(self):
        self._register_flow(self.forward)
    
    def forward(self, *input, **kwargs):
        raise NotImplementedError
    
    def export(self, output_dir):
        raise NotImplementedError

    # NOTE: deprecated interface
    def export2onnx(self, output_dir):
        raise NotImplementedError

    # NOTE: deprecated interface
    def export2coreml(self, output_dir):
        raise NotImplementedError

    def parameters(self):
        return torch.nn.Module.parameters(self)

    def optimizable_parameters(self):
        return torch.nn.Module.parameters(self)
    
    def initialize(self):
        if self.args.initialization is not None:
            print('Initialize module - {0} - from: {1}'.format(self.__class__.__name__, self.args.initialization))
            self.load_state_dict(torch.load(self.args.initialization, map_location='cpu'), strict=self.args.strict_initialization)
