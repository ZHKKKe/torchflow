import torch

from .flow import FlowModule


class Module(torch.nn.Module, FlowModule):
    def __init__(self, args=None):
        torch.nn.Module.__init__(self)
        FlowModule.__init__(self)

        self.args = args

    def __call__(self, *input, flow=None, **kwargs):
        if flow is not None:
            self.set_flow(flow)
        return self.flow(*input, **kwargs)

    def _register_flows(self):
        pass
    
    def forward(self, *input, **kwargs):
        raise NotImplementedError
    
    def export2onnx(self, output_dir):
        raise NotImplementedError

    def export2coreml(self, output_dir):
        raise NotImplementedError

    def parameters(self):
        return torch.nn.Module.parameters(self)
