import os
from torchflow.tool import parser, logger
from torchflow.environment import distributed


class Exporter:
    def __init__(self, proxy, args, modules):
        self.proxy = proxy
        
        self.args = args
        self._preprocess_args()
        logger.info('Exporter arguments:\n')
        parser.print_args(self.args)

        logger.info('Build exporter...\n')
        self.modules = modules

    def _is_master(self):
        return distributed.rank == 0

    def _preprocess_args(self):
        self.args.module = parser.fetch_arg(self.args.module, [])

    def export(self):
        for name in self.modules:
            if name in self.args.module:
                logger.log('Export module: {0}...'.format(name))

                output_dir = os.path.join(self.args.output_dir, name, 'onnx')
                os.makedirs(output_dir, exist_ok=True)
                try:
                    self.modules[name].export2onnx(output_dir)
                except NotImplementedError:
                    logger.warn('ONNX Exporter of module `{0}` is not implemented\n'.format(name))
                
                output_dir = os.path.join(self.args.output_dir, name, 'coreml')
                os.makedirs(output_dir, exist_ok=True)
                try:
                    self.modules[name].export2coreml(output_dir)
                except NotImplementedError:
                    logger.warn('COreML Exporter of module `{0}` is not implemented\n'.format(name))
