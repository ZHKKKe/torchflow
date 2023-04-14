import os
import torch

from torchflow.environment import distributed
from torchflow.tool import helper, parser, logger
from torchflow.nnmodule import data, optimizer, lrer, parallel
from .trainer import Trainer
from .tester import Tester
from .exporter import Exporter


class Proxy:
    def __init__(self, args, dataset_dict, module_dict, flow_dict):
        self.args = args
        self._preprocess_args()
        
        logger.info('Log to file:\n  {0}\n'.format(self.args.env.log_file))
        logger.tofile(self.args.env.log_file)

        logger.info('Environment arguments:\n')
        parser.print_args(self.args.env)

        self.dataset_dict = dataset_dict
        self.module_dict = module_dict
        self.flow_dict = flow_dict

        self.datasets = {}
        self.modules = {}
        self.optimizers = {}
        self.lrers = {}

        self.trainer = None
        self.tester = None
        self.exporter = None

        self._build_datasets()
        self._build_modules()
        self._build_trainer()
        self._build_tester()
        self._build_exporter()

        # TODO: check if at least one operator [trainer, tester, exporter] is defined

        if parser.fetch_arg(self.args.env.resume.file, False):
            self.resume()
    
    def execute(self):
        if self.trainer is not None:
            for i in self.trainer.train():
                # TODO: any better way to access `test_iter`?
                if self.args.trainer.test_iter is not None:
                    if i != 0 and i % self.args.trainer.test_iter == 0:
                        with torch.cuda.device(distributed.rank):
                            torch.cuda.empty_cache()

                        distributed.barrier()
                        if self._is_master():
                            self.tester.test(i)
                        distributed.barrier()

        elif self.tester is not None:
            distributed.barrier()
            if self._is_master():
                self.tester.test(0)
            distributed.barrier()

        elif self.exporter is not None:
            distributed.barrier()
            if self._is_master():
                self.exporter.export()
            distributed.barrier()

        # TODO: release multiprocessing resources

    def _is_master(self):
        return distributed.rank == 0

    def _preprocess_args(self):
        # NOTE: preprocess all `env`` arguments here

        self.args.env.output_dir = '{root}/{uid}'.format(
            root=parser.fetch_arg(self.args.env.output_dir, './'),
            uid=helper.datetime_str()
        )
        os.makedirs(self.args.env.output_dir, exist_ok=True)
        self.args.env.log_file = os.path.join(self.args.env.output_dir, 'log')

    def _build_datasets(self):
        logger.info('Dataset arguments:\n')
        parser.print_args(self.args.dataset)

        logger.info('Build datasets:\n')
        _dataset_args = vars(self.args.dataset)
        for _dname in _dataset_args:
            _dataset = _dataset_args[_dname]
            logger.log('Build dataset: {0}...'.format(_dname))
            self.datasets[_dname] = self.dataset_dict[_dataset.type](_dataset.args)
            logger.log('Total items in the dataset {0}: {1}'.format(_dname, len(self.datasets[_dname])))

    def _build_modules(self):
        logger.info('Module arguments:\n')
        parser.print_args(self.args.module)

        logger.info('Build modules...\n')
        _module_args = vars(self.args.module)
        for _mname in _module_args:
            _module = _module_args[_mname]
            logger.log('Build module: {0}...'.format(_mname))

            # build module
            self.modules[_mname] = self.module_dict[_module.type](_module.args).to(distributed.rank)

            if parser.fetch_arg(_module.args.print_params, False):
                logger.log(helper.module_str(self.modules[_mname]))

            if distributed.world_size > 1:
                try:
                    self.modules[_mname] = parallel.DistributedDataParallel(
                        self.modules[_mname], device_ids=[distributed.rank])
                except:
                    pass

            # build module optimizer
            if parser.fetch_arg(_module.optimizer, False):
                _parameters = self.modules[_mname].parameters()
                self.optimizers[_mname] = \
                    optimizer.__dict__[_module.optimizer.type](_module.optimizer.args)(_parameters)
            else:
                self.optimizers[_mname] = None
            
            # build module lrer
            if parser.fetch_arg(_module.lrer, False) and self.optimizers[_mname] is not None:
                self.lrers[_mname] = \
                    lrer.__dict__[_module.lrer.type](_module.lrer.args)(self.optimizers[_mname])
            else:
                self.lrers[_mname] = None

    def _build_trainer(self):
        if parser.fetch_arg(self.args.trainer, False):
            logger.info('Build trainer...\n')

            # build trainer dataloaders
            dataloaders = {}
            _dataset_args = vars(self.args.dataset)
            for _dname in _dataset_args:
                _dataset = _dataset_args[_dname]
                batch_size = parser.fetch_arg(_dataset.loader.args.batch_size, 1)
                num_workers = parser.fetch_arg(_dataset.loader.args.num_workers, 4)
                
                if distributed.world_size > 1:
                    if num_workers != 0:
                        num_workers = 0
                        logger.warn(
                            '{0} trainer loader: automatically set `num_workers=0`'
                            'as multiprocessing is used.\n'.format(_dname))
                    
                    sampler = data.DistributedInfiniteSampler(
                        self.datasets[_dname], 
                        num_replicas=distributed.world_size, 
                        rank=distributed.rank,
                        drop_last=True
                    )
                    dataloaders[_dname] = torch.utils.data.DataLoader(
                        self.datasets[_dname],
                        batch_size=batch_size,
                        sampler=sampler, 
                        num_workers=num_workers,      # NOTE: num_workers should be 0 if using multiprocessing
                        pin_memory=True
                    )
                else:
                    batch_sampler = data.InfiniteBatchSampler(
                        self.datasets[_dname].indexes, batch_size, True)

                    dataloaders[_dname] = torch.utils.data.DataLoader(
                        self.datasets[_dname],
                        batch_sampler=batch_sampler,
                        num_workers=num_workers,
                        pin_memory=True
                    )

            self.args.trainer.output_dir = os.path.join(self.args.env.output_dir, 'trainer')
            self.trainer = Trainer(self, self.args.trainer, self.datasets, dataloaders,
                self.modules, self.optimizers, self.lrers, self.flow_dict)
        else:
            logger.info('No trainer is defined in the config.\n')

    def _build_tester(self):
        if parser.fetch_arg(self.args.tester, False):
            num_workers = 0 if distributed.world_size > 1 else 1

            # NOTE: only build tester on the device with rank = 0
            if self._is_master():
                logger.info('Build tester on the master process...\n')
                dataloaders = {}

                # build tester dataloaders
                _dataset_args = vars(self.args.dataset)
                for _dname in _dataset_args:
                    batch_sampler = data.InfiniteBatchSampler(self.datasets[_dname].indexes, 1, False)
                    dataloaders[_dname] = torch.utils.data.DataLoader(
                        self.datasets[_dname],
                        batch_sampler=batch_sampler,
                        num_workers=num_workers,
                        pin_memory=True
                    )

                self.args.tester.output_dir = os.path.join(self.args.env.output_dir, 'tester')
                self.tester = Tester(
                    self, self.args.tester, self.datasets, dataloaders, self.modules, self.flow_dict)
        else:
            logger.info('No tester is defined in the config.\n')

    def _build_exporter(self):
        if parser.fetch_arg(self.args.exporter, False):

            if self._is_master():
                logger.info('Build exporter on the master process...\n')

                self.args.exporter.output_dir = os.path.join(self.args.env.output_dir, 'exporter')
                self.exporter = Exporter(self, self.args.exporter, self.modules)
        else:
            logger.info('No exporter is defined in the config.\n')

    # TODO: support to resume 'flow' data?
    def resume(self):
        logger.info('Resume checkpoint from:\n  {0}\n'.format(self.args.env.resume.file))

        state = torch.load(self.args.env.resume.file)
        strict = parser.fetch_arg(self.args.env.resume.strict, True)
        restart = parser.fetch_arg(self.args.env.resume.restart, False)

        if parser.fetch_arg(self.args.env.resume.load, False):
            for _mname in parser.fetch_arg(self.args.env.resume.load.module, []):
                if _mname in state['module']:
                    self.modules[_mname].load_state_dict(state['module'][_mname], strict=strict)
            
            if not restart:
                for _oname in parser.fetch_arg(self.args.env.resume.load.optimizer, []):
                    if self.optimizers[_oname] is not None and _oname in state['optimizer']:
                        self.optimizers[_oname].load_state_dict(state['optimizer'][_oname])
                for _lname in parser.fetch_arg(self.args.env.resume.load.lrer, []):
                    if self.lrers[_lname] is not None and _lname in state['lrer']:
                        self.lrers[_lname].load_state_dict(state['lrer'][_lname])
        else:
            for _mname in self.modules.keys():
                if _mname in state['module'].keys():
                    self.modules[_mname].load_state_dict(state['module'][_mname], strict=strict)
            
            if not restart:
                for _oname in self.optimizers.keys():
                    if _oname in state['optimizer'].keys() and self.optimizers[_oname] is not None:
                        self.optimizers[_oname].load_state_dict(state['optimizer'][_oname])
                for _lname in self.lrers.keys():
                    if _lname in state['lrer'].keys() and self.lrers[_lname] is not None:
                        self.lrers[_lname].load_state_dict(state['lrer'][_lname])

        # for trainer
        if parser.fetch_arg(self.args.trainer, False) and not restart:
            if 'trainer' in state.keys() and self.trainer is not None:
                # if not 'restart', reset 'cur_iter' for trainner
                if 'cur_iter' in state['trainer'].keys():
                    self.args.trainer.cur_iter = state['trainer']['cur_iter']
                    logger.log('Resume trainer argument `cur_iter` to {0}'.format(self.args.trainer.cur_iter))

    def state_dict(self):
        state = {
            'module': {},
            'optimizer': {},
            'lrer': {}
        }

        if parser.fetch_arg(self.args.env.resume.save, False):
            _mnames = parser.fetch_arg(self.args.env.resume.save.module, {})
            _onames = parser.fetch_arg(self.args.env.resume.save.optimizer, {})
            _lnames = parser.fetch_arg(self.args.env.resume.save.lrer, {})
        else:
            _mnames = self.modules.keys()
            _onames = self.optimizers.keys()
            _lnames = self.lrers.keys()

        for _mname in _mnames:
            state['module'][_mname] = self.modules[_mname].state_dict()
        for _oname in _onames:
            if self.optimizers[_oname] is not None:
                state['optimizer'][_oname] = self.optimizers[_oname].state_dict()
        for _lname in _lnames:
            if self.lrers[_lname] is not None:
                state['lrer'][_lname] = self.lrers[_lname].state_dict()

        return state
