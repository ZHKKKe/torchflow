import os
import time
import torch

from torchflow.nnmodule import parallel, optimizer, lrer
from torchflow.environment import distributed
from torchflow.tool import helper, parser, logger, meter


class Trainer:
    def __init__(self, proxy, args, datasets, dataloaders, modules, module_optimizers, module_lrers, flow_dict):
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
        self.module_optimizers = module_optimizers
        self.module_lrers = module_lrers

        # flow-wise optimizers and lrers
        self.flow_optimizers = {}
        self.flow_lrers = {}

        # flows
        self.flow_dict = flow_dict
        self.flows = {}
        self._build_flows()

        # tools
        self.meter = meter.MeanMeter()

        self.status = {
            'cur_iter': -1
        }

    def _is_master(self):
        return distributed.rank == 0

    def _preprocess_args(self):
        self.args.uid = helper.datetime_str()
        self.args.checkpoint_dir = os.path.join(self.args.output_dir, 'checkpoint')
        self.args.visualization_dir = os.path.join(self.args.output_dir, 'visualization')

        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        os.makedirs(self.args.visualization_dir, exist_ok=True)

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
            
            _flow_datasets = {}
            _flow_dataloaders = {}
            _flow_dataset_args = vars(_flow.args.dataset)
            for _dname in _flow_dataset_args:
                _dataset = _flow_dataset_args[_dname]
                _flow_datasets[_dname] = self.datasets[_dataset.name]
                _flow_dataloaders[_dname] = self.dataloaders[_dataset.name]

            _flow_modules = {}
            _flow_module_optimizers = {}
            _flow_module_args = vars(_flow.args.module)
            for _mname in _flow_module_args:
                _module = _flow_module_args[_mname]
                _flow_modules[_mname] = self.modules[_module.name]
                if _module.optimization in [True, None]:
                    _flow_module_optimizers[_mname] = self.module_optimizers[_module.name]
                else:
                    _flow_module_optimizers[_mname] = None

            _flow_optimizer = None
            if parser.fetch_arg(_flow.args.optimizer, False):
                _flow_optimizer_args = _flow.args.optimizer
                _flow_optimizable_parameters = []

                if parser.fetch_arg(_flow_optimizer_args.module, False):
                    if len(_flow_optimizer_args.module) == 0:
                        logger.error('No module is defined for the flow `{0}`\'s optimizer.'.format(_fname))
                    else:
                        for _mname in _flow_optimizer_args.module:
                            _flow_optimizable_parameters.append({ 'params': _flow_modules[_mname].optimizable_parameters() })
                else:
                    logger.error('No module is defined for the flow `{0}`\'s optimizer.'.format(_fname))

                _flow_optimizer = optimizer.__dict__[_flow_optimizer_args.type](_flow_optimizer_args.args)(_flow_optimizable_parameters)
            self.flow_optimizers[_fname] = _flow_optimizer

            _flow_lrer = None
            if parser.fetch_arg(_flow.args.lrer, False) and _flow_optimizer is not None:
                _flow_lrer_args = _flow.args.lrer
                _flow_lrer = lrer.__dict__[_flow_lrer_args.type](_flow_lrer_args.args)(_flow_optimizer)
            self.flow_lrers[_fname] = _flow_lrer

            self.flows[_fname] = self.flow_dict[_flow.type](
                _flow.args, _flow_datasets, _flow_dataloaders, _flow_modules)

            has_flow_module_optimizers = False
            for _oname in _flow_module_optimizers:
                if _flow_module_optimizers[_oname] is not None:
                    has_flow_module_optimizers = True
                    break

            has_flow_optimizer = _flow_optimizer is not None

            if has_flow_module_optimizers and has_flow_optimizer:
                logger.warn(
                    'Both flow_optimizer in the Flow and module_optimizers in the Flow\'s Modules are set.\n'
                    'flow_optimizer in the Flow will be used.\n'
                    'NOTE: flow_optimizer in Flows is recommended.'
                )
            elif not has_flow_module_optimizers and not has_flow_optimizer:
                logger.error(
                    'Neither flow_optimizer in the Flow and module_optimizers in the Flow\'s Modules are set.\n'
                    'Please check the configuration file and set one.\n'
                    'NOTE: flow_optimizer in Flows is recommended.'
                )

            self.flows[_fname].register_module_optimizers(_flow_module_optimizers)

            self.flows[_fname].register_flow_optimizer(_flow_optimizer)
            self.flows[_fname].register_flow_lrer(_flow_lrer)
            
            # TODO: fail to warp flow by `parallel.DistributedDataParallel`
            if distributed.world_size > 1:
                try:
                    self.flows[_fname] = parallel.DistributedDataParallel(
                        self.flows[_fname], device_ids=[distributed.rank])
                except:
                    logger.warn('Failt to set rank - {0} - for flow - {1}.'.format(distributed.rank, _fname))
                    pass
            else:
                self.flows[_fname] = self.flows[_fname].to(distributed.rank)

    def train(self):
        logger.info('Start training...\n')
        for i in range(self.status['cur_iter'] + 1, self.args.max_iter):
            # batch variables
            self.status['cur_iter'] = i
            _time = time.time()
            
            # reset optimizers
            # for name in self.module_optimizers:
            #     optimizer = self.module_optimizers[name]
            #     if optimizer is not None:
            #         optimizer.zero_grad()

            # run flows
            for name in self.flows:
                flow = self.flows[name]
                if self.status['cur_iter'] % flow.args.interval == 0:
                    flow.clear_optimizers()
                    
                    flow.set_cur_iter(self.status['cur_iter'])
                    flow.prepare()
                    flow.forward()
                    flow.postprocess()
                    
                    flow.run_optimizers()
            
            # run lrers
            for name in self.module_lrers:
                lrer = self.module_lrers[name]
                if lrer is not None:
                    if i != 0 and i % lrer.step_interval_iters == 0:
                        logger.info('Lrer of module {0}: run a step.\n'.format(name))
                        lrer.step()

            for name in self.flow_lrers:
                lrer = self.flow_lrers[name]
                if lrer is not None:
                    if i != 0 and i % lrer.step_interval_iters == 0:
                        logger.info('Lrer of flow {0}: run a step.\n'.format(name))
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
                'cur_iter': self.status['cur_iter'],
                'flows': {},
            }
        }

        for _fname in self.flows:
            _flow_optimizer = self.flows[_fname].flow_optimizer
            _flow_lrer = self.flows[_fname].flow_lrer
            state['trainer']['flows'][_fname] = {
                'optimizer': _flow_optimizer.state_dict() if _flow_optimizer is not None else None,
                'lrer': _flow_lrer.state_dict() if _flow_lrer is not None else None,
            }

        state.update(self.proxy.state_dict())
        checkpoint_path = os.path.join(self.args.checkpoint_dir, '{0}.ckpt'.format(self.status['cur_iter']))
        torch.save(state, checkpoint_path)

        logger.log('Save trainer checkpoint to:\n  {0}'.format(checkpoint_path))

    def load_checkpoint(self, state):
        if 'cur_iter' in state.keys():
            self.status['cur_iter'] = state['cur_iter']
            logger.log('Resume trainer argument `cur_iter` to {0}'.format(self.status['cur_iter']))

        for _fname in self.flows.keys():
            if self.flows[_fname] is not None and _fname in state['flows'].keys():
                flow_state_dict = state['flows'][_fname]

                if flow_state_dict['optimizer'] is not None and self.flows[_fname].flow_optimizer is not None:
                    self.flows[_fname].flow_optimizer.load_state_dict(flow_state_dict['optimizer'])
                if flow_state_dict['lrer'] is not None and self.flows[_fname].flow_lrer is not None:
                    self.flows[_fname].flow_lrer.load_state_dict(flow_state_dict['lrer'])
    