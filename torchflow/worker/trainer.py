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

        # global optimizer and global lrer
        self.global_optimizers = {}
        self.global_lrers = {}
        self._build_global_optimizers_and_lrers()

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

    def _build_global_optimizers_and_lrers(self):
        if parser.fetch_arg(self.args.global_optimizer, False):
            _optimizers = vars(self.args.global_optimizer)

            for _oname in _optimizers:
                _optimizer = _optimizers[_oname]
            
                _parameters = []
                _module_lr = vars(_optimizer.module_lr)
                for _mname in _optimizer.module:
                    _lr = _optimizer.args.lr
                    if _mname in _module_lr.keys():
                        _lr = _module_lr[_mname]
                    
                    _parameters.append({
                        'params': self.modules[_mname].optimizable_parameters(),
                        'lr': _lr
                    })

                self.global_optimizers[_oname] = optimizer.__dict__[_optimizer.type](_optimizer.args)(_parameters)

                if parser.fetch_arg(_optimizer.lrer, False):
                    _lrer = _optimizer.lrer
                    self.global_lrers[_oname] = lrer.__dict__[_lrer.type](_lrer.args)(self.global_optimizers[_oname])

    def _build_flows(self):
        _flow_args = vars(self.args.flow)
        for _fname in _flow_args:
            _flow = _flow_args[_fname]
            logger.log('\nBuild trainer flow: {0}...'.format(_fname))
            datasets = {}
            dataloaders = {}

            _dataset_args = vars(_flow.args.dataset)
            for _dname in _dataset_args:
                _dataset = _dataset_args[_dname]
                datasets[_dname] = self.datasets[_dataset.name]
                dataloaders[_dname] = self.dataloaders[_dataset.name]

            modules = {}
            optimizers = {}
            optimization_info = {}
            _module_args = vars(_flow.args.module)
            for _mname in _module_args:
                _module = _module_args[_mname]
                modules[_mname] = self.modules[_module.name]
                if _module.optimization in [True, None]:
                    optimizers[_mname] = self.module_optimizers[_module.name]
                else:
                    optimizers[_mname] = None
                
                if optimizers[_mname] is not None:
                    if _module.name in optimization_info.keys():
                        optimization_info[_module.name].append(_module.name)
                    else: 
                        optimization_info[_module.name] = [_module.name]

                if parser.fetch_arg(self.args.global_optimizer, False):
                    _global_optimizer = vars(self.args.global_optimizer)
                    for _goname in _global_optimizer:
                        if _module.name in _global_optimizer[_goname].module:

                            if _module.name in optimization_info.keys():
                                optimization_info[_module.name].append('GlobalOptimizer_' + _goname)
                            else: 
                                optimization_info[_module.name] = ['GlobalOptimizer_' + _goname]

                            if 'GlobalOptimizer_' + _goname not in optimizers.keys():
                                optimizers['GlobalOptimizer_' + _goname] = self.global_optimizers[_goname]

            self.flows[_fname] = self.flow_dict[_flow.type](
                _flow.args, datasets, dataloaders, modules)
            
            # NOTE: to compatible with old code, set optimizers by 'register_optimizers'
            self.flows[_fname].register_optimizers(optimizers)

            logger.log('  The following modules will be optimized by this flow:')
            for _mname, _onames in optimization_info.items():
                logger.log('    {0} - Optimizers: {1}'.format(_mname, _onames))

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
            for name in self.module_optimizers:
                optimizer = self.module_optimizers[name]
                if optimizer is not None:
                    optimizer.zero_grad()

            # run flows
            for name in self.flows:
                flow = self.flows[name]
                if self.status['cur_iter'] % flow.args.interval == 0:
                    flow.clear_optimizers()
                    
                    flow.set_cur_iter(self.status['cur_iter'])
                    for adx in range(0, flow.args.accumulate_steps):
                        flow.prepare()
                        flow.forward()
                        flow.postprocess()
                    
                    flow.run_optimizers()
            
            # run lrers
            for name in self.module_lrers:
                lrer = self.module_lrers[name]
                if lrer is not None:
                    if i != 0 and i % lrer.step_interval_iters == 0:
                        logger.info('Module Lrer {0}: run a step.\n'.format(name))
                        lrer.step()

            for name in self.global_lrers:
                lrer = self.global_lrers[name]
                if lrer is not None:
                    if i != 0 and i % lrer.step_interval_iters == 0:
                        logger.info('Global Lrer {0}: run a step.\n'.format(name))
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
                'global_optimizer': {},
                'global_lrer': {}
            }
        }

        for _goname in self.global_optimizers:
            state['trainer']['global_optimizer'][_goname] = self.global_optimizers[_goname].state_dict()
        for _glname in self.global_lrers:
            state['trainer']['global_lrer'][_glname] = self.global_lrers[_glname].state_dict()

        state.update(self.proxy.state_dict())
        checkpoint_path = os.path.join(self.args.checkpoint_dir, '{0}.ckpt'.format(self.status['cur_iter']))
        torch.save(state, checkpoint_path)

        logger.log('Save trainer checkpoint to:\n  {0}'.format(checkpoint_path))

    def load_checkpoint(self, trainer_state):
        if 'cur_iter' in trainer_state.keys():
            self.status['cur_iter'] = trainer_state['cur_iter']
            logger.log('Resume trainer argument `cur_iter` to {0}'.format(self.status['cur_iter']))

        if 'global_optimizer' in trainer_state.keys():
            for _goname in self.global_optimizers:
                if _goname in trainer_state['global_optimizer'].keys():
                    logger.log('Load global optimizer: {0}'.format(_goname))
                    self.global_optimizers[_goname].load_state_dict(trainer_state['global_optimizer'][_goname])
                if _goname in trainer_state['global_lrer'].keys():
                    logger.log('Load global lrer: {0}'.format(_goname))
                    self.global_lrers[_goname].load_state_dict(trainer_state['global_lrer'][_goname])
