import torch

from torchflow.environment import distributed
from torchflow.tool import parser
from .flow import FlowModule


class _Module(torch.nn.Module, FlowModule):
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

    def _register_flows(self):
        self._register_flow(self.forward)
    
    def forward(self, *input, **kwargs):
        raise NotImplementedError
    
    def export_with_weight_dtype(self, output_dir, weight_dtype):
        raise NotImplementedError
    
    # NOTE: deprecated interface
    def export(self, output_dir):
        raise NotImplementedError

    # NOTE: deprecated interface
    def export2onnx(self, output_dir):
        raise NotImplementedError

    # NOTE: deprecated interface
    def export2coreml(self, output_dir):
        raise NotImplementedError

    def parameters(self):
        return NotImplementedError

    def optimizable_parameters(self):
        return NotImplementedError
    
    def initialize(self):
        raise NotImplementedError


class Module(_Module):
    def __init__(self, args=None):
        super().__init__(args)

    def _register_args(self):
        self.args.print_params = parser.fetch_arg(self.args.print_params, False)
        self.args.initialization = parser.fetch_arg(self.args.initialization, None)
        self.args.strict_initialization = parser.fetch_arg(self.args.strict_initialization, True)

    def initialize(self):
        if self.args.initialization is not None:
            print('Initialize module - {0} - from: {1}'.format(self.__class__.__name__, self.args.initialization))
            
            state_dict = torch.load(self.args.initialization, map_location='cpu')
            
            processed_state_dict = {}
            for key in state_dict.keys():
                if key.startswith('module.'):
                    processed_state_dict[key[7:]] = state_dict[key]
                else:
                    processed_state_dict[key] = state_dict[key]

            self.load_state_dict(processed_state_dict, strict=self.args.strict_initialization)

    def parameters(self):
        return torch.nn.Module.parameters(self)

    def optimizable_parameters(self):
        return torch.nn.Module.parameters(self)


class TorchScriptModule(_Module):
    def __init__(self, args=None):
        super().__init__(args)

        self.model = None
        self.dtype = None

    def _register_args(self):
        self.args.pt_file = parser.fetch_arg(self.args.pt_file, required=True)
        self.args.input_keys = parser.fetch_arg(self.args.input_keys, required=True)
        self.args.output_keys = parser.fetch_arg(self.args.output_keys, required=True)
        self.args.weight_dtype = parser.fetch_arg(self.args.weight_dtype, required=True, choices=['f16', 'f32'])

    def forward(self, inputs):
        input_values = []
        for ik in self.args.input_keys:
            input_values.append(inputs[ik])

        output_values = self.model(*input_values)
        if not isinstance(output_values, tuple):
            output_values = (output_values, )

        outputs = {}
        assert len(self.args.output_keys) == len(output_values)
        for ok, ov in zip(self.args.output_keys, output_values):
            outputs[ok] = ov

        return outputs
        
    def initialize(self):
        self.model = torch.jit.load(self.args.pt_file)
        self.model.eval()
            
        if self.args.weight_dtype == 'f16':
            self.model.half()
            self.dtype = torch.float16
        else:
            self.model.float()
            self.dtype = torch.float32

    def parameters(self):
        return self.model.parameters()
