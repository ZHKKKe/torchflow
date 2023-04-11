import os
import time
import torch

from torchflow.environment import distributed
from torchflow.tool import parser, logger, meter


class Tester:
    def __init__(self, proxy, args, datasets, dataloaders, modules, flow_dict):
        self.proxy = proxy

        # arguments
        self.args = args
        self._preprocess_args()
        logger.info('Tester arguments:\n')
        parser.print_args(self.args)

        # components
        logger.info('Build tester...\n')
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.modules = modules

        # flows
        self.flow_dict = flow_dict
        self.flow = None
        self._build_flow()

        # tools
        self.meter = meter.MeanMeter()

    def _is_master(self):
        return distributed.rank == 0

    def _preprocess_args(self):
        self.args.visualization_dir = os.path.join(self.args.output_dir, 'visualization')
        os.makedirs(self.args.visualization_dir, exist_ok=True)

        self.args.logging_iter = parser.fetch_arg(self.args.logging_iter, None)
        self.args.visualization_iter = parser.fetch_arg(self.args.visualization_iter, None)

    def _build_flow(self):
        assert len(vars(self.args.flow)) == 1

        _fname = list(vars(self.args.flow).keys())[0]
        _flow = vars(self.args.flow)[_fname]

        logger.log('Build tester flow: {0}...'.format(_fname))
        datasets = {}
        dataloaders = {}
        _dataset_args = vars(_flow.args.dataset)
        for _dname in _dataset_args:
            _dataset = _dataset_args[_dname]
            datasets[_dname] = self.datasets[_dataset.name]
            dataloaders[_dname] = self.dataloaders[_dataset.name]
        
        modules = {}
        _module_args = vars(_flow.args.module)
        for _mname in _module_args:
            _module = _module_args[_mname]
            modules[_mname] = self.modules[_module.name]

        self.flow = self.flow_dict[_flow.type](
            _flow.args, datasets, dataloaders, modules)

    def test(self, uid=''):
        logger.log('\n')
        logger.info('Start testing...\n')

        total_iter = 0

        for _dname in self.flow.datasets:
            _dataset = self.flow.datasets[_dname]
            total_iter += len(_dataset)

        for i in range(0, total_iter):
            _time = time.time()

            self.flow()

            if self._is_master():
                # visualization
                if self.args.visualization_iter is not None and i % self.args.visualization_iter == 0:
                    self.flow.visualization(
                        os.path.join(self.args.visualization_dir),
                        '{0}_{1}'.format(uid, i)
                    )

                # logging
                self.meter['time'] = time.time() - _time
                if self.args.logging_iter is not None and i % self.args.logging_iter == 0:
                    logger.log('\n')
                    logger.log('Iteration: [{0}/{1}]'.format(i, total_iter))
                    logger.log('Batch Time (s): {meter[time]:.3f}'.format(meter=self.meter))
                    logger.log('--------------------------------')
                    self.flow.logging()

        logger.log('\n')
        logger.info('Finish testing.\n')
