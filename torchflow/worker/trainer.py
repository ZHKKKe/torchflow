import os
import time
import torch

from torchflow.nnmodule import parallel
from torchflow.environment import distributed
from torchflow.tool import helper, parser, logger, meter


class Trainer:
    def __init__(self, proxy, args, datasets, dataloaders, modules, optimizers, lrers, flow_dict):
        self.proxy = proxy

        # arguments
        self.args = args
        self._preprocess_args()
        logger.info('Trainer arguments:\n')
        parser.print_args(self.args)

        # components
        logger.info('Build trainer...\n')
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.modules = modules
        self.optimizers = optimizers
        self.lrers = lrers

        # flows
        self.flow_dict = flow_dict
        self.flows = {}
        self._build_flows()

        # tools
        self.meter = meter.MeanMeter()

    def _is_master(self):
        return distributed.rank == 0

    def _preprocess_args(self):
        self.args.uid = helper.datetime_str()
        self.args.checkpoint_dir = os.path.join(self.args.output_dir, 'checkpoint')
        self.args.visualization_dir = os.path.join(self.args.output_dir, 'visualization')

        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        os.makedirs(self.args.visualization_dir, exist_ok=True)

        self.args.cur_iter = -1
        self.args.max_iter = parser.fetch_arg(self.args.max_iter, None)
        self.args.logging_iter = parser.fetch_arg(self.args.logging_iter, None)
        self.args.visualization_iter = parser.fetch_arg(self.args.visualization_iter, None)
        self.args.test_iter = parser.fetch_arg(self.args.test_iter, None)
        self.args.checkpoint_iter = parser.fetch_arg(self.args.checkpoint_iter, None)

    def _build_flows(self):
        _flow_args = vars(self.args.flow)
        for _fname in _flow_args:
            _flow = _flow_args[_fname]
            logger.log('Build trainer flow: {0}...'.format(_fname))
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

            self.flows[_fname] = self.flow_dict[_flow.type](
                _flow.args, datasets, dataloaders, modules)
            
            if distributed.world_size > 1:
                try:
                    self.flows[_fname] = parallel.DistributedDataParallel(
                        self.flows[_fname], device_ids=[distributed.rank])
                except:
                    # TODO: add warn
                    pass

    def train(self):
        logger.info('Start training...\n')
        for i in range(self.args.cur_iter + 1, self.args.max_iter):
            # batch variables
            self.args.cur_iter = i
            _time = time.time()
            
            # reset optimizers
            for name in self.optimizers:
                optimizer = self.optimizers[name]
                if optimizer is not None:
                    optimizer.zero_grad()

            # run flows
            for name in self.flows:
                flow = self.flows[name]
                flow()
            
            # run optimizers
            for name in self.optimizers:
                optimizer = self.optimizers[name]
                if optimizer is not None:
                    optimizer.step()

            # run lrers
            for name in self.lrers:
                lrer = self.lrers[name]
                if lrer is not None:
                    if i != 0 and i % lrer.step_interval_iters == 0:
                        logger.info('Lrer of module {0}: run a step.'.format(name))
                        lrer.step()
            
            # only for the master process
            if self._is_master():
                # visualization
                if self.args.visualization_iter is not None and i % self.args.visualization_iter == 0:
                    for name in self.flows:
                        flow = self.flows[name]
                        flow.visualization(
                            os.path.join(self.args.visualization_dir),
                            '{0}_{1}'.format(i, name)
                        )

                # logging
                self.meter['time'] = time.time() - _time
                if self.args.logging_iter is not None and i % self.args.logging_iter == 0:
                    logger.log('\n')
                    logger.log('Iteration: [{0}/{1}]'.format(i, self.args.max_iter))
                    logger.log('Batch Time (s): {meter[time]:.3f}'.format(meter=self.meter))
                    logger.log('--------------------------------')
                    for name in self.flows:
                        flow = self.flows[name]
                        logger.log('Flow: {0}'.format(name))
                        flow.logging()

            # save checkpoints
            if self.args.checkpoint_iter is not None and i != 0 and i % self.args.checkpoint_iter == 0:
                distributed.barrier()
                if self._is_master():
                    self.save_checkpoint()
                distributed.barrier()

            yield i

    def save_checkpoint(self):
        logger.log('\n')
        logger.info('Trainer checkpoint...\n')
        
        state = {
            'trainer': {
                'cur_iter': self.args.cur_iter
            }
        }
        state.update(self.proxy.state_dict())
        checkpoint_path = os.path.join(self.args.checkpoint_dir, '{0}.ckpt'.format(self.args.cur_iter))
        torch.save(state, checkpoint_path)

        logger.log('Save trainer checkpoint to:\n  {0}'.format(checkpoint_path))
