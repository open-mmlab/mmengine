# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import os.path as osp
import random
import shutil
import tempfile
from functools import partial
from unittest import TestCase, skipIf

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset

from mmengine.config import Config
from mmengine.dataset import DefaultSampler, pseudo_collate
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, Hook,
                            IterTimerHook, LoggerHook, ParamSchedulerHook,
                            RuntimeInfoHook)
from mmengine.logging import HistoryBuffer, MessageHub, MMLogger
from mmengine.model import BaseDataPreprocessor, BaseModel, ImgDataPreprocessor
from mmengine.optim import (DefaultOptimWrapperConstructor, MultiStepLR,
                            OptimWrapper, OptimWrapperDict, StepLR)
from mmengine.registry import (DATASETS, EVALUATOR, FUNCTIONS, HOOKS,
                               LOG_PROCESSORS, LOOPS, METRICS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPER_CONSTRUCTORS,
                               OPTIM_WRAPPERS, PARAM_SCHEDULERS, RUNNERS,
                               Registry)
from mmengine.runner import (BaseLoop, EpochBasedTrainLoop, IterBasedTrainLoop,
                             LogProcessor, Runner, TestLoop, ValLoop)
from mmengine.runner.loops import _InfiniteDataloaderIterator
from mmengine.runner.priority import Priority, get_priority
from mmengine.utils import digit_version, is_list_of
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.visualization import Visualizer


def skip_test_comile():
    if digit_version(torch.__version__) < digit_version('2.0.0'):
        return True
    # The default compiling backend for PyTorch 2.0, inductor, does not support
    # Nvidia graphics cards older than Volta architecture.
    # As PyTorch does not provide a public function to confirm the availability
    # of the inductor, we check its availability by attempting compilation.
    if not torch.cuda.is_available():
        return True
    try:
        model = nn.Sequential(nn.Conv2d(1, 1, 1), nn.BatchNorm2d(1)).cuda()
        compiled_model = torch.compile(model)
        compiled_model(torch.ones(3, 1, 1, 1).cuda())
    except Exception:
        return True
    else:
        return False


SKIP_TEST_COMPILE = skip_test_comile()


class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs, data_sample, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if isinstance(data_sample, list):
            data_sample = torch.stack(data_sample)
        outputs = self.linear1(inputs)
        outputs = self.linear2(outputs)

        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (data_sample - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            return outputs


class ToyModel1(ToyModel):

    def __init__(self):
        super().__init__()


class ToySyncBNModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 2)
        self.bn = nn.SyncBatchNorm(8)

    def forward(self, inputs, data_sample, mode='tensor'):
        data_sample = torch.stack(data_sample)
        inputs = torch.stack(inputs)
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)

        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (data_sample - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        elif mode == 'predict':
            outputs = dict(log_vars=dict(a=1, b=0.5))
            return outputs


class ToyGANModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 1)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs, data_sample, mode='tensor'):
        data_sample = torch.stack(data_sample)
        inputs = torch.stack(inputs)
        output1 = self.linear1(inputs)
        output2 = self.linear2(inputs)

        if mode == 'tensor':
            return output1, output2
        elif mode == 'loss':
            loss1 = (data_sample - output1).sum()
            loss2 = (data_sample - output2).sum()
            outputs = dict(linear1=loss1, linear2=loss2)
            return outputs
        elif mode == 'predict':
            return output1, output2

    def train_step(self, data, optim_wrapper):
        data = self.data_preprocessor(data)
        loss = self(**data, mode='loss')
        optim_wrapper['linear1'].update_params(loss['linear1'])
        optim_wrapper['linear2'].update_params(loss['linear2'])
        return loss


class CustomModelWrapper(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.model = module


class ToyMultipleOptimizerConstructor:

    def __init__(self, optim_wrapper_cfg, paramwise_cfg=None):
        if not isinstance(optim_wrapper_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optim_wrapper_cfg)}')
        assert paramwise_cfg is None, (
            'parawise_cfg should be set in each optimizer separately')
        self.optim_wrapper_cfg = optim_wrapper_cfg
        self.constructors = {}
        for key, cfg in self.optim_wrapper_cfg.items():
            _cfg = cfg.copy()
            paramwise_cfg_ = _cfg.pop('paramwise_cfg', None)
            self.constructors[key] = DefaultOptimWrapperConstructor(
                _cfg, paramwise_cfg_)

    def __call__(self, model: nn.Module) -> OptimWrapperDict:
        optimizers = {}
        while hasattr(model, 'module'):
            model = model.module

        for key, constructor in self.constructors.items():
            module = getattr(model, key)
            optimizers[key] = constructor(module)
        return OptimWrapperDict(**optimizers)


class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class ToyDatasetNoMeta(Dataset):
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class ToyMetric1(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class ToyMetric2(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class ToyOptimWrapper(OptimWrapper):
    ...


class ToyHook(Hook):
    priority = 'Lowest'

    def before_train_epoch(self, runner):
        pass


class ToyHook2(Hook):
    priority = 'Lowest'

    def after_train_epoch(self, runner):
        pass


class CustomTrainLoop(BaseLoop):

    def __init__(self, runner, dataloader, max_epochs):
        super().__init__(runner, dataloader)
        self._max_epochs = max_epochs

    def run(self) -> None:
        pass


class CustomValLoop(BaseLoop):

    def __init__(self, runner, dataloader, evaluator):
        super().__init__(runner, dataloader)
        self._runner = runner

        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator

    def run(self) -> None:
        pass


class CustomTestLoop(BaseLoop):

    def __init__(self, runner, dataloader, evaluator):
        super().__init__(runner, dataloader)
        self._runner = runner

        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator

    def run(self) -> None:
        pass


class CustomLogProcessor(LogProcessor):

    def __init__(self, window_size=10, by_epoch=True, custom_cfg=None):
        self.window_size = window_size
        self.by_epoch = by_epoch
        self.custom_cfg = custom_cfg if custom_cfg else []
        self._check_custom_cfg()


class CustomRunner(Runner):

    def __init__(self,
                 model,
                 work_dir,
                 train_dataloader=None,
                 val_dataloader=None,
                 test_dataloader=None,
                 train_cfg=None,
                 val_cfg=None,
                 test_cfg=None,
                 auto_scale_lr=None,
                 optim_wrapper=None,
                 param_scheduler=None,
                 val_evaluator=None,
                 test_evaluator=None,
                 default_hooks=None,
                 custom_hooks=None,
                 data_preprocessor=None,
                 load_from=None,
                 resume=False,
                 launcher='none',
                 env_cfg=dict(dist_cfg=dict(backend='nccl')),
                 log_processor=None,
                 log_level='INFO',
                 visualizer=None,
                 default_scope=None,
                 randomness=dict(seed=None),
                 experiment_name=None,
                 cfg=None):
        pass

    def setup_env(self, env_cfg):
        pass


class ToyEvaluator(Evaluator):

    def __init__(self, metrics):
        super().__init__(metrics)


def collate_fn(data_batch):
    return pseudo_collate(data_batch)


def custom_collate(data_batch, pad_value):
    return pseudo_collate(data_batch)


def custom_worker_init(worker_id):
    np.random.seed(worker_id)
    random.seed(worker_id)


class TestRunner(TestCase):

    def setUp(self):
        MODELS.register_module(module=ToyModel, force=True)
        MODELS.register_module(module=ToyModel1, force=True)
        MODELS.register_module(module=ToySyncBNModel, force=True)
        MODELS.register_module(module=ToyGANModel, force=True)
        MODEL_WRAPPERS.register_module(module=CustomModelWrapper, force=True)
        OPTIM_WRAPPER_CONSTRUCTORS.register_module(
            module=ToyMultipleOptimizerConstructor, force=True)
        DATASETS.register_module(module=ToyDataset, force=True)
        DATASETS.register_module(module=ToyDatasetNoMeta, force=True)
        METRICS.register_module(module=ToyMetric1, force=True)
        METRICS.register_module(module=ToyMetric2, force=True)
        OPTIM_WRAPPERS.register_module(module=ToyOptimWrapper, force=True)
        HOOKS.register_module(module=ToyHook, force=True)
        HOOKS.register_module(module=ToyHook2, force=True)
        LOOPS.register_module(module=CustomTrainLoop, force=True)
        LOOPS.register_module(module=CustomValLoop, force=True)
        LOOPS.register_module(module=CustomTestLoop, force=True)
        LOG_PROCESSORS.register_module(module=CustomLogProcessor, force=True)
        RUNNERS.register_module(module=CustomRunner, force=True)
        EVALUATOR.register_module(module=ToyEvaluator, force=True)
        FUNCTIONS.register_module(module=custom_collate, force=True)
        FUNCTIONS.register_module(module=custom_worker_init, force=True)

        self.temp_dir = tempfile.mkdtemp()
        epoch_based_cfg = dict(
            model=dict(type='ToyModel'),
            work_dir=self.temp_dir,
            train_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            test_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            auto_scale_lr=dict(base_batch_size=16, enable=False),
            optim_wrapper=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
            val_evaluator=dict(type='ToyMetric1'),
            test_evaluator=dict(type='ToyMetric1'),
            train_cfg=dict(
                by_epoch=True, max_epochs=3, val_interval=1, val_begin=1),
            val_cfg=dict(),
            test_cfg=dict(),
            custom_hooks=[],
            default_hooks=dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(
                    type='CheckpointHook', interval=1, by_epoch=True),
                sampler_seed=dict(type='DistSamplerSeedHook')),
            data_preprocessor=None,
            launcher='none',
            env_cfg=dict(dist_cfg=dict(backend='nccl')),
        )
        self.epoch_based_cfg = Config(epoch_based_cfg)
        self.iter_based_cfg = copy.deepcopy(self.epoch_based_cfg)
        self.iter_based_cfg.train_dataloader = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='InfiniteSampler', shuffle=True),
            batch_size=3,
            num_workers=0)
        self.iter_based_cfg.train_cfg = dict(by_epoch=False, max_iters=12)
        self.iter_based_cfg.default_hooks = dict(
            runtime_info=dict(type='RuntimeInfoHook'),
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook'),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=False),
            sampler_seed=dict(type='DistSamplerSeedHook'))

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        MODELS.module_dict.pop('ToyModel')
        MODELS.module_dict.pop('ToyModel1')
        MODELS.module_dict.pop('ToySyncBNModel')
        MODELS.module_dict.pop('ToyGANModel')
        MODEL_WRAPPERS.module_dict.pop('CustomModelWrapper')
        OPTIM_WRAPPER_CONSTRUCTORS.module_dict.pop(
            'ToyMultipleOptimizerConstructor')
        OPTIM_WRAPPERS.module_dict.pop('ToyOptimWrapper')
        DATASETS.module_dict.pop('ToyDataset')
        DATASETS.module_dict.pop('ToyDatasetNoMeta')
        METRICS.module_dict.pop('ToyMetric1')
        METRICS.module_dict.pop('ToyMetric2')
        HOOKS.module_dict.pop('ToyHook')
        HOOKS.module_dict.pop('ToyHook2')
        LOOPS.module_dict.pop('CustomTrainLoop')
        LOOPS.module_dict.pop('CustomValLoop')
        LOOPS.module_dict.pop('CustomTestLoop')
        LOG_PROCESSORS.module_dict.pop('CustomLogProcessor')
        RUNNERS.module_dict.pop('CustomRunner')
        EVALUATOR.module_dict.pop('ToyEvaluator')
        FUNCTIONS.module_dict.pop('custom_collate')
        FUNCTIONS.module_dict.pop('custom_worker_init')

        logging.shutdown()
        MMLogger._instance_dict.clear()
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # 1. test arguments
        # 1.1 train_dataloader, train_cfg, optimizer and param_scheduler
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init1'
        cfg.pop('train_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        # all of training related configs are None and param_scheduler should
        # also be None
        cfg.experiment_name = 'test_init2'
        cfg.pop('train_dataloader')
        cfg.pop('optim_wrapper')
        cfg.pop('param_scheduler')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # all of training related configs are not None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init3'
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # all of training related configs are not None and param_scheduler
        # can be None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init4'
        cfg.pop('param_scheduler')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)
        self.assertEqual(runner.param_schedulers, None)

        # param_scheduler should be None when optimizer is None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init5'
        cfg.pop('train_cfg')
        cfg.pop('train_dataloader')
        cfg.pop('optim_wrapper')
        with self.assertRaisesRegex(ValueError, 'should be None'):
            runner = Runner(**cfg)

        # 1.2 val_dataloader, val_evaluator, val_cfg
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init6'
        cfg.pop('val_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        cfg.experiment_name = 'test_init7'
        cfg.pop('val_dataloader')
        cfg.pop('val_evaluator')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init8'
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # 1.3 test_dataloader, test_evaluator and test_cfg
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init9'
        cfg.pop('test_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            runner = Runner(**cfg)

        cfg.experiment_name = 'test_init10'
        cfg.pop('test_dataloader')
        cfg.pop('test_evaluator')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # 1.4 test env params
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init11'
        runner = Runner(**cfg)
        self.assertFalse(runner.distributed)
        self.assertFalse(runner.deterministic)

        # 1.5 message_hub, logger and visualizer
        # they are all not specified
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init12'
        runner = Runner(**cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertIsInstance(runner.visualizer, Visualizer)

        # they are all specified
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init13'
        cfg.log_level = 'INFO'
        cfg.visualizer = None
        runner = Runner(**cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertIsInstance(runner.visualizer, Visualizer)

        assert runner.distributed is False
        assert runner.seed is not None
        assert runner.work_dir == self.temp_dir

        # 2 model should be initialized
        self.assertIsInstance(runner.model,
                              (nn.Module, DistributedDataParallel))
        self.assertEqual(runner.model_name, 'ToyModel')

        # 3. test lazy initialization
        self.assertIsInstance(runner._train_dataloader, dict)
        self.assertIsInstance(runner._val_dataloader, dict)
        self.assertIsInstance(runner._test_dataloader, dict)
        self.assertIsInstance(runner.optim_wrapper, dict)
        self.assertIsInstance(runner.param_schedulers, dict)

        # After calling runner.train(),
        # train_dataloader and val_loader should be initialized but
        # test_dataloader should also be dict
        runner.train()

        self.assertIsInstance(runner._train_loop, BaseLoop)
        self.assertIsInstance(runner.train_dataloader, DataLoader)
        self.assertIsInstance(runner.optim_wrapper, OptimWrapper)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)
        self.assertIsInstance(runner._val_loop, BaseLoop)
        self.assertIsInstance(runner._val_loop.dataloader, DataLoader)
        self.assertIsInstance(runner._val_loop.evaluator, Evaluator)

        # After calling runner.test(), test_dataloader should be initialized
        self.assertIsInstance(runner._test_loop, dict)
        runner.test()
        self.assertIsInstance(runner._test_loop, BaseLoop)
        self.assertIsInstance(runner._test_loop.dataloader, DataLoader)
        self.assertIsInstance(runner._test_loop.evaluator, Evaluator)

        # 4. initialize runner with objects rather than config
        model = ToyModel()
        optim_wrapper = OptimWrapper(SGD(
            model.parameters(),
            lr=0.01,
        ))
        toy_hook = ToyHook()
        toy_hook2 = ToyHook2()

        train_dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        val_dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        test_dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        runner = Runner(
            model=model,
            work_dir=self.temp_dir,
            train_cfg=dict(
                by_epoch=True, max_epochs=3, val_interval=1, val_begin=1),
            train_dataloader=train_dataloader,
            optim_wrapper=optim_wrapper,
            param_scheduler=MultiStepLR(optim_wrapper, milestones=[1, 2]),
            val_cfg=dict(),
            val_dataloader=val_dataloader,
            val_evaluator=[ToyMetric1()],
            test_cfg=dict(),
            test_dataloader=test_dataloader,
            test_evaluator=[ToyMetric1()],
            default_hooks=dict(param_scheduler=toy_hook),
            custom_hooks=[toy_hook2],
            experiment_name='test_init14')
        runner.train()
        runner.test()

        # 5. Test building multiple runners. In Windows, nccl could not be
        # available, and this test will be skipped.
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            cfg = copy.deepcopy(self.epoch_based_cfg)
            cfg.experiment_name = 'test_init15'
            cfg.launcher = 'pytorch'
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29600'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['LOCAL_RANK'] = '0'
            Runner(**cfg)
            cfg.experiment_name = 'test_init16'
            Runner(**cfg)

        # 6.1 Test initializing with empty scheduler.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init17'
        cfg.param_scheduler = None
        runner = Runner(**cfg)
        self.assertIsNone(runner.param_schedulers)

        # 6.2 Test initializing single scheduler.
        cfg.experiment_name = 'test_init18'
        cfg.param_scheduler = dict(type='MultiStepLR', milestones=[1, 2])
        Runner(**cfg)

        # 6.3 Test initializing list of scheduler.
        cfg.param_scheduler = [
            dict(type='MultiStepLR', milestones=[1, 2]),
            dict(type='MultiStepLR', milestones=[2, 3])
        ]
        cfg.experiment_name = 'test_init19'
        Runner(**cfg)

        # 6.4 Test initializing 2 schedulers for 2 optimizers.
        cfg.param_scheduler = dict(
            linear1=dict(type='MultiStepLR', milestones=[1, 2]),
            linear2=dict(type='MultiStepLR', milestones=[1, 2]),
        )
        cfg.experiment_name = 'test_init20'
        Runner(**cfg)

        # 6.5 Test initializing 2 schedulers for 2 optimizers.
        cfg.param_scheduler = dict(
            linear1=[dict(type='MultiStepLR', milestones=[1, 2])],
            linear2=[dict(type='MultiStepLR', milestones=[1, 2])],
        )
        cfg.experiment_name = 'test_init21'
        Runner(**cfg)

        # 6.6 Test initializing with `_ParameterScheduler`.
        optimizer = SGD(nn.Linear(1, 1).parameters(), lr=0.1)
        cfg.param_scheduler = MultiStepLR(
            milestones=[1, 2], optimizer=optimizer)
        cfg.experiment_name = 'test_init22'
        Runner(**cfg)

        # 6.7 Test initializing with list of `_ParameterScheduler`.
        cfg.param_scheduler = [
            MultiStepLR(milestones=[1, 2], optimizer=optimizer)
        ]
        cfg.experiment_name = 'test_init23'
        Runner(**cfg)

        # 6.8 Test initializing with 2 `_ParameterScheduler` for 2 optimizers.
        cfg.param_scheduler = dict(
            linear1=MultiStepLR(milestones=[1, 2], optimizer=optimizer),
            linear2=MultiStepLR(milestones=[1, 2], optimizer=optimizer))
        cfg.experiment_name = 'test_init24'
        Runner(**cfg)

        # 6.9 Test initializing with 2 list of `_ParameterScheduler` for 2
        # optimizers.
        cfg.param_scheduler = dict(
            linear1=[MultiStepLR(milestones=[1, 2], optimizer=optimizer)],
            linear2=[MultiStepLR(milestones=[1, 2], optimizer=optimizer)])
        cfg.experiment_name = 'test_init25'
        Runner(**cfg)

        # 6.10 Test initializing with error type scheduler.
        cfg.param_scheduler = dict(linear1='error_type')
        cfg.experiment_name = 'test_init26'
        with self.assertRaisesRegex(AssertionError, 'Each value of'):
            Runner(**cfg)

        cfg.param_scheduler = 'error_type'
        cfg.experiment_name = 'test_init27'
        with self.assertRaisesRegex(TypeError,
                                    '`param_scheduler` should be a'):
            Runner(**cfg)

    def test_dump_config(self):
        # dump config from dict.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        for idx, cfg in enumerate((cfg, cfg._cfg_dict)):
            cfg.experiment_name = f'test_dump{idx}'
            runner = Runner.from_cfg(cfg=cfg)
            assert osp.exists(
                osp.join(runner.work_dir, f'{runner.timestamp}.py'))
            # dump config from file.
            with tempfile.TemporaryDirectory() as temp_config_dir:
                # Set `delete=Flase` and close the file to make it
                # work in Windows.
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix='.py', delete=False)
                temp_config_file.close()
                file_cfg = Config(
                    self.epoch_based_cfg._cfg_dict,
                    filename=temp_config_file.name)
                file_cfg.experiment_name = f'test_dump2{idx}'
                runner = Runner.from_cfg(cfg=file_cfg)
                assert osp.exists(
                    osp.join(runner.work_dir,
                             osp.basename(temp_config_file.name)))

    def test_from_cfg(self):
        runner = Runner.from_cfg(cfg=self.epoch_based_cfg)
        self.assertIsInstance(runner, Runner)

    def test_setup_env(self):
        # TODO
        pass

    def test_build_logger(self):
        self.epoch_based_cfg.experiment_name = 'test_build_logger1'
        runner = Runner.from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertEqual(runner.experiment_name, runner.logger.instance_name)

        # input is a dict
        logger = runner.build_logger(name='test_build_logger2')
        self.assertIsInstance(logger, MMLogger)
        self.assertEqual(logger.instance_name, 'test_build_logger2')

        # input is a dict but does not contain name key
        runner._experiment_name = 'test_build_logger3'
        logger = runner.build_logger()
        self.assertIsInstance(logger, MMLogger)
        self.assertEqual(logger.instance_name, 'test_build_logger3')

    def test_build_message_hub(self):
        self.epoch_based_cfg.experiment_name = 'test_build_message_hub1'
        runner = Runner.from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertEqual(runner.message_hub.instance_name,
                         runner.experiment_name)

        # input is a dict
        message_hub_cfg = dict(name='test_build_message_hub2')
        message_hub = runner.build_message_hub(message_hub_cfg)
        self.assertIsInstance(message_hub, MessageHub)
        self.assertEqual(message_hub.instance_name, 'test_build_message_hub2')

        # input is a dict but does not contain name key
        runner._experiment_name = 'test_build_message_hub3'
        message_hub_cfg = dict()
        message_hub = runner.build_message_hub(message_hub_cfg)
        self.assertIsInstance(message_hub, MessageHub)
        self.assertEqual(message_hub.instance_name, 'test_build_message_hub3')

        # input is not a valid type
        with self.assertRaisesRegex(TypeError, 'message_hub should be'):
            runner.build_message_hub('invalid-type')

    def test_build_visualizer(self):
        self.epoch_based_cfg.experiment_name = 'test_build_visualizer1'
        runner = Runner.from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.visualizer, Visualizer)
        self.assertEqual(runner.experiment_name,
                         runner.visualizer.instance_name)

        # input is a Visualizer object
        self.assertEqual(
            id(runner.build_visualizer(runner.visualizer)),
            id(runner.visualizer))

        # input is a dict
        visualizer_cfg = dict(type='Visualizer', name='test_build_visualizer2')
        visualizer = runner.build_visualizer(visualizer_cfg)
        self.assertIsInstance(visualizer, Visualizer)
        self.assertEqual(visualizer.instance_name, 'test_build_visualizer2')

        # input is a dict but does not contain name key
        runner._experiment_name = 'test_build_visualizer3'
        visualizer_cfg = None
        visualizer = runner.build_visualizer(visualizer_cfg)
        self.assertIsInstance(visualizer, Visualizer)
        self.assertEqual(visualizer.instance_name, 'test_build_visualizer3')

        # input is not a valid type
        with self.assertRaisesRegex(TypeError, 'visualizer should be'):
            runner.build_visualizer('invalid-type')

    def test_default_scope(self):
        TOY_SCHEDULERS = Registry(
            'parameter scheduler', parent=PARAM_SCHEDULERS, scope='toy')

        @TOY_SCHEDULERS.register_module(force=True)
        class ToyScheduler(MultiStepLR):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        self.epoch_based_cfg.param_scheduler = dict(
            type='ToyScheduler', milestones=[1, 2])
        self.epoch_based_cfg.default_scope = 'toy'

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_default_scope'
        runner = Runner.from_cfg(cfg)
        runner.train()
        self.assertIsInstance(runner.param_schedulers[0], ToyScheduler)

    def test_build_model(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_model1'
        runner = Runner.from_cfg(cfg)
        self.assertIsInstance(runner.model, ToyModel)
        self.assertIsInstance(runner.model.data_preprocessor,
                              BaseDataPreprocessor)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_data_preprocessor'
        cfg.data_preprocessor = dict(type='ImgDataPreprocessor')
        runner = Runner.from_cfg(cfg)
        # data_preprocessor is passed to used if no `data_preprocessor`
        # in model config.
        self.assertIsInstance(runner.model.data_preprocessor,
                              ImgDataPreprocessor)

        # input should be a nn.Module object or dict
        with self.assertRaisesRegex(TypeError, 'model should be'):
            runner.build_model('invalid-type')

        # input is a nn.Module object
        _model = ToyModel1()
        model = runner.build_model(_model)
        self.assertEqual(id(model), id(_model))

        # input is a dict
        model = runner.build_model(dict(type='ToyModel1'))
        self.assertIsInstance(model, ToyModel1)

    def test_wrap_model(self):
        # revert sync batchnorm
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_revert_syncbn'
        cfg.model = dict(type='ToySyncBNModel')
        runner = Runner.from_cfg(cfg)
        self.assertIsInstance(runner.model, BaseModel)
        assert not isinstance(runner.model.bn, nn.SyncBatchNorm)

        # custom model wrapper
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_wrap_model'
        cfg.model_wrapper_cfg = dict(type='CustomModelWrapper')
        runner = Runner.from_cfg(cfg)
        self.assertIsInstance(runner.model, BaseModel)

        # Test with ddp wrapper
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29515'
            os.environ['RANK'] = str(0)
            os.environ['WORLD_SIZE'] = str(1)
            cfg.launcher = 'pytorch'
            cfg.experiment_name = 'test_wrap_model1'
            runner = Runner.from_cfg(cfg)
            self.assertIsInstance(runner.model, CustomModelWrapper)

            # Test cfg.sync_bn = 'torch', when model does not have BN layer
            cfg = copy.deepcopy(self.epoch_based_cfg)
            cfg.launcher = 'pytorch'
            cfg.experiment_name = 'test_wrap_model2'
            cfg.sync_bn = 'torch'
            cfg.model_wrapper_cfg = dict(type='CustomModelWrapper')
            runner.from_cfg(cfg)

            @MODELS.register_module(force=True)
            class ToyBN(BaseModel):

                def __init__(self):
                    super().__init__()
                    self.bn = nn.BatchNorm2d(2)

                def forward(self, *args, **kwargs):
                    pass

            cfg.model = dict(type='ToyBN')
            cfg.experiment_name = 'test_data_preprocessor2'
            runner = Runner.from_cfg(cfg)
            self.assertIsInstance(runner.model.model.bn,
                                  torch.nn.SyncBatchNorm)

            cfg.sync_bn = 'unknown'
            cfg.experiment_name = 'test_data_preprocessor3'
            with self.assertRaises(ValueError):
                Runner.from_cfg(cfg)

    def test_scale_lr(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_scale_lr'
        runner = Runner.from_cfg(cfg)

        # When no base_batch_size in auto_scale_lr, an
        # assertion error will raise.
        auto_scale_lr = dict(enable=True)
        optim_wrapper = OptimWrapper(SGD(runner.model.parameters(), lr=0.01))
        with self.assertRaises(AssertionError):
            runner.scale_lr(optim_wrapper, auto_scale_lr)

        # When auto_scale_lr is None or enable is False, the lr will
        # not be linearly scaled.
        auto_scale_lr = dict(base_batch_size=16, enable=False)
        optim_wrapper = OptimWrapper(SGD(runner.model.parameters(), lr=0.01))
        runner.scale_lr(optim_wrapper)
        self.assertEqual(optim_wrapper.optimizer.param_groups[0]['lr'], 0.01)
        runner.scale_lr(optim_wrapper, auto_scale_lr)
        self.assertEqual(optim_wrapper.optimizer.param_groups[0]['lr'], 0.01)

        # When auto_scale_lr is correct and enable is True, the lr will
        # be linearly scaled.
        auto_scale_lr = dict(base_batch_size=16, enable=True)
        real_bs = runner.world_size * cfg.train_dataloader['batch_size']
        optim_wrapper = OptimWrapper(SGD(runner.model.parameters(), lr=0.01))
        runner.scale_lr(optim_wrapper, auto_scale_lr)
        self.assertEqual(optim_wrapper.optimizer.param_groups[0]['lr'],
                         0.01 * (real_bs / 16))

        # Test when optim_wrapper is an OptimWrapperDict
        optim_wrapper = OptimWrapper(SGD(runner.model.parameters(), lr=0.01))
        wrapper_dict = OptimWrapperDict(wrapper=optim_wrapper)
        runner.scale_lr(wrapper_dict, auto_scale_lr)
        scaled_lr = wrapper_dict['wrapper'].optimizer.param_groups[0]['lr']
        self.assertEqual(scaled_lr, 0.01 * (real_bs / 16))

    def test_build_optim_wrapper(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_optim_wrapper'
        runner = Runner.from_cfg(cfg)

        # input should be an Optimizer object or dict
        with self.assertRaisesRegex(TypeError, 'optimizer wrapper should be'):
            runner.build_optim_wrapper('invalid-type')

        # 1. test one optimizer
        # 1.1 input is an Optimizer object
        optimizer = SGD(runner.model.parameters(), lr=0.01)
        optim_wrapper = OptimWrapper(optimizer)
        optim_wrapper = runner.build_optim_wrapper(optim_wrapper)
        self.assertEqual(id(optimizer), id(optim_wrapper.optimizer))

        # 1.2 input is a dict
        optim_wrapper = runner.build_optim_wrapper(
            dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)))
        self.assertIsInstance(optim_wrapper, OptimWrapper)

        # 1.3 use default OptimWrapper type.
        optim_wrapper = runner.build_optim_wrapper(
            dict(optimizer=dict(type='SGD', lr=0.01)))
        self.assertIsInstance(optim_wrapper, OptimWrapper)

        # 2. test multiple optmizers
        # 2.1 input is a dict which contains multiple optimizer objects
        optimizer1 = SGD(runner.model.linear1.parameters(), lr=0.01)
        optim_wrapper1 = OptimWrapper(optimizer1)
        optimizer2 = Adam(runner.model.linear2.parameters(), lr=0.02)
        optim_wrapper2 = OptimWrapper(optimizer2)
        optim_wrapper_cfg = dict(key1=optim_wrapper1, key2=optim_wrapper2)
        optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
        self.assertIsInstance(optim_wrapper, OptimWrapperDict)
        self.assertIsInstance(optim_wrapper['key1'].optimizer, SGD)
        self.assertIsInstance(optim_wrapper['key2'].optimizer, Adam)

        # 2.2 each item mush be an optimizer object when "type" and
        # "constructor" are not in optimizer
        optimizer1 = SGD(runner.model.linear1.parameters(), lr=0.01)
        optim_wrapper1 = OptimWrapper(optimizer1)
        optim_wrapper2 = dict(
            type='OptimWrapper', optimizer=dict(type='Adam', lr=0.01))
        optim_cfg = dict(key1=optim_wrapper1, key2=optim_wrapper2)
        with self.assertRaisesRegex(ValueError,
                                    'each item mush be an optimizer object'):
            runner.build_optim_wrapper(optim_cfg)

        # 2.3 input is a dict which contains multiple configs
        optim_wrapper_cfg = dict(
            linear1=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            linear2=dict(
                type='OptimWrapper', optimizer=dict(type='Adam', lr=0.02)),
            constructor='ToyMultipleOptimizerConstructor')
        optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
        self.assertIsInstance(optim_wrapper, OptimWrapperDict)
        self.assertIsInstance(optim_wrapper['linear1'].optimizer, SGD)
        self.assertIsInstance(optim_wrapper['linear2'].optimizer, Adam)

        # 2.4 input is a dict which contains optimizer instance.
        model = nn.Linear(1, 1)
        optimizer = SGD(model.parameters(), lr=0.1)
        optim_wrapper_cfg = dict(optimizer=optimizer)
        optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
        self.assertIsInstance(optim_wrapper, OptimWrapper)
        self.assertIs(optim_wrapper.optimizer, optimizer)

        # Specify the type of optimizer wrapper
        model = nn.Linear(1, 1)
        optimizer = SGD(model.parameters(), lr=0.1)
        optim_wrapper_cfg = dict(
            optimizer=optimizer, type='ToyOptimWrapper', accumulative_counts=2)
        optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
        self.assertIsInstance(optim_wrapper, ToyOptimWrapper)
        self.assertIs(optim_wrapper.optimizer, optimizer)
        self.assertEqual(optim_wrapper._accumulative_counts, 2)

    def test_build_param_scheduler(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_param_scheduler'
        runner = Runner.from_cfg(cfg)

        # `build_optim_wrapper` should be called before
        # `build_param_scheduler`
        cfg = dict(type='MultiStepLR', milestones=[1, 2])
        runner.optim_wrapper = dict(
            key1=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            key2=dict(
                type='OptimWrapper', optimizer=dict(type='Adam', lr=0.02)),
        )
        with self.assertRaisesRegex(AssertionError, 'should be called before'):
            runner.build_param_scheduler(cfg)

        runner.optim_wrapper = runner.build_optim_wrapper(
            dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)))
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertIsInstance(param_schedulers, list)
        self.assertEqual(len(param_schedulers), 1)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)

        # 1. test one optimizer and one parameter scheduler
        # 1.1 input is a ParamScheduler object
        param_scheduler = MultiStepLR(runner.optim_wrapper, milestones=[1, 2])
        param_schedulers = runner.build_param_scheduler(param_scheduler)
        self.assertEqual(len(param_schedulers), 1)
        self.assertEqual(id(param_schedulers[0]), id(param_scheduler))

        # 1.2 input is a dict
        param_schedulers = runner.build_param_scheduler(param_scheduler)
        self.assertEqual(len(param_schedulers), 1)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)

        # 2. test one optimizer and list of parameter schedulers
        # 2.1 input is a list of dict
        cfg = [
            dict(type='MultiStepLR', milestones=[1, 2]),
            dict(type='StepLR', step_size=1)
        ]
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertEqual(len(param_schedulers), 2)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)
        self.assertIsInstance(param_schedulers[1], StepLR)

        # 2.2 input is a list and some items are ParamScheduler objects
        cfg = [param_scheduler, dict(type='StepLR', step_size=1)]
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertEqual(len(param_schedulers), 2)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)
        self.assertIsInstance(param_schedulers[1], StepLR)

        # 3. test multiple optimizers and list of parameter schedulers
        optimizer1 = SGD(runner.model.linear1.parameters(), lr=0.01)
        optim_wrapper1 = OptimWrapper(optimizer1)
        optimizer2 = Adam(runner.model.linear2.parameters(), lr=0.02)
        optim_wrapper2 = OptimWrapper(optimizer2)
        optim_wrapper_cfg = dict(key1=optim_wrapper1, key2=optim_wrapper2)
        runner.optim_wrapper = runner.build_optim_wrapper(optim_wrapper_cfg)
        cfg = [
            dict(type='MultiStepLR', milestones=[1, 2]),
            dict(type='StepLR', step_size=1)
        ]
        param_schedulers = runner.build_param_scheduler(cfg)
        print(param_schedulers)
        self.assertIsInstance(param_schedulers, dict)
        self.assertEqual(len(param_schedulers), 2)
        self.assertEqual(len(param_schedulers['key1']), 2)
        self.assertEqual(len(param_schedulers['key2']), 2)

        # 4. test multiple optimizers and multiple parameter shceduers
        cfg = dict(
            key1=dict(type='MultiStepLR', milestones=[1, 2]),
            key2=[
                dict(type='MultiStepLR', milestones=[1, 2]),
                dict(type='StepLR', step_size=1)
            ])
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertIsInstance(param_schedulers, dict)
        self.assertEqual(len(param_schedulers), 2)
        self.assertEqual(len(param_schedulers['key1']), 1)
        self.assertEqual(len(param_schedulers['key2']), 2)

        # 5. test converting epoch-based scheduler to iter-based
        runner.optim_wrapper = runner.build_optim_wrapper(
            dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)))

        # 5.1 train loop should be built before converting scheduler
        cfg = dict(
            type='MultiStepLR', milestones=[1, 2], convert_to_iter_based=True)

        # 5.2 convert epoch-based to iter-based scheduler
        cfg = dict(
            type='MultiStepLR',
            milestones=[1, 2],
            begin=1,
            end=7,
            convert_to_iter_based=True)
        runner._train_loop = runner.build_train_loop(runner.train_loop)
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertFalse(param_schedulers[0].by_epoch)
        self.assertEqual(param_schedulers[0].begin, 4)
        self.assertEqual(param_schedulers[0].end, 28)

        # 6. test set default end of schedulers
        cfg = dict(type='MultiStepLR', milestones=[1, 2], begin=1)
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertTrue(param_schedulers[0].by_epoch)
        self.assertEqual(param_schedulers[0].begin, 1)
        # runner.max_epochs = 3
        self.assertEqual(param_schedulers[0].end, 3)

        cfg = dict(
            type='MultiStepLR',
            milestones=[1, 2],
            begin=1,
            convert_to_iter_based=True)
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertFalse(param_schedulers[0].by_epoch)
        self.assertEqual(param_schedulers[0].begin, 4)
        # runner.max_iters = 3*4
        self.assertEqual(param_schedulers[0].end, 12)

    def test_build_evaluator(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_evaluator'
        runner = Runner.from_cfg(cfg)

        # input is a BaseEvaluator or ComposedEvaluator object
        evaluator = Evaluator(ToyMetric1())
        self.assertEqual(id(runner.build_evaluator(evaluator)), id(evaluator))

        evaluator = Evaluator([ToyMetric1(), ToyMetric2()])
        self.assertEqual(id(runner.build_evaluator(evaluator)), id(evaluator))

        # input is a dict
        evaluator = dict(type='ToyMetric1')
        self.assertIsInstance(runner.build_evaluator(evaluator), Evaluator)

        # input is a list of dict
        evaluator = [dict(type='ToyMetric1'), dict(type='ToyMetric2')]
        self.assertIsInstance(runner.build_evaluator(evaluator), Evaluator)

        # input is a list of built metric.
        metric = [ToyMetric1(), ToyMetric2()]
        _evaluator = runner.build_evaluator(metric)
        self.assertIs(_evaluator.metrics[0], metric[0])
        self.assertIs(_evaluator.metrics[1], metric[1])

        # test collect device
        evaluator = [
            dict(type='ToyMetric1', collect_device='cpu'),
            dict(type='ToyMetric2', collect_device='gpu')
        ]
        _evaluator = runner.build_evaluator(evaluator)
        self.assertEqual(_evaluator.metrics[0].collect_device, 'cpu')
        self.assertEqual(_evaluator.metrics[1].collect_device, 'gpu')

        # test build a customize evaluator
        evaluator = dict(
            type='ToyEvaluator',
            metrics=[
                dict(type='ToyMetric1', collect_device='cpu'),
                dict(type='ToyMetric2', collect_device='gpu')
            ])
        _evaluator = runner.build_evaluator(evaluator)
        self.assertIsInstance(runner.build_evaluator(evaluator), ToyEvaluator)
        self.assertEqual(_evaluator.metrics[0].collect_device, 'cpu')
        self.assertEqual(_evaluator.metrics[1].collect_device, 'gpu')

        # test evaluator must be a Evaluator instance
        with self.assertRaisesRegex(TypeError, 'evaluator should be'):
            _evaluator = runner.build_evaluator(ToyMetric1())

    def test_build_dataloader(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_dataloader'
        runner = Runner.from_cfg(cfg)

        cfg = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=1,
            num_workers=0)
        seed = np.random.randint(2**31)
        dataloader = runner.build_dataloader(cfg, seed=seed)
        self.assertIsInstance(dataloader, DataLoader)
        self.assertIsInstance(dataloader.dataset, ToyDataset)
        self.assertIsInstance(dataloader.sampler, DefaultSampler)
        self.assertEqual(dataloader.sampler.seed, seed)

        # diff_rank_seed is True
        dataloader = runner.build_dataloader(
            cfg, seed=seed, diff_rank_seed=True)
        self.assertNotEqual(dataloader.sampler.seed, seed)

        # custom worker_init_fn
        cfg = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            worker_init_fn=dict(type='custom_worker_init'),
            batch_size=1,
            num_workers=2)
        dataloader = runner.build_dataloader(cfg)
        self.assertIs(dataloader.worker_init_fn.func, custom_worker_init)

        # collate_fn is a dict
        cfg = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            worker_init_fn=dict(type='custom_worker_init'),
            batch_size=1,
            num_workers=2,
            collate_fn=dict(type='pseudo_collate'))
        dataloader = runner.build_dataloader(cfg)
        self.assertIsInstance(dataloader.collate_fn, partial)

        # collate_fn is a callable object
        def custom_collate(data_batch):
            return data_batch

        cfg = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            worker_init_fn=dict(type='custom_worker_init'),
            batch_size=1,
            num_workers=2,
            collate_fn=custom_collate)
        dataloader = runner.build_dataloader(cfg)
        self.assertIs(dataloader.collate_fn, custom_collate)
        # collate_fn is a invalid value
        with self.assertRaisesRegex(
                TypeError, 'collate_fn should be a dict or callable object'):
            cfg = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                worker_init_fn=dict(type='custom_worker_init'),
                batch_size=1,
                num_workers=2,
                collate_fn='collate_fn')
            dataloader = runner.build_dataloader(cfg)
            self.assertIsInstance(dataloader.collate_fn, partial)

        # num_batch_per_epoch is not None
        cfg = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            collate_fn=dict(type='default_collate'),
            batch_size=3,
            num_workers=2,
            num_batch_per_epoch=2)
        dataloader = runner.build_dataloader(cfg)
        self.assertEqual(len(dataloader.dataset), 6)

    def test_build_train_loop(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_train_loop'
        runner = Runner.from_cfg(cfg)

        # input should be a Loop object or dict
        with self.assertRaisesRegex(TypeError, 'should be'):
            runner.build_train_loop('invalid-type')

        # Only one of type or by_epoch can exist in cfg
        cfg = dict(type='EpochBasedTrainLoop', by_epoch=True, max_epochs=3)
        with self.assertRaisesRegex(RuntimeError, 'Only one'):
            runner.build_train_loop(cfg)

        # input is a dict and contains type key
        cfg = dict(type='EpochBasedTrainLoop', max_epochs=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, EpochBasedTrainLoop)

        cfg = dict(type='IterBasedTrainLoop', max_iters=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, IterBasedTrainLoop)

        # input is a dict and does not contain type key
        cfg = dict(by_epoch=True, max_epochs=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, EpochBasedTrainLoop)

        cfg = dict(by_epoch=False, max_iters=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, IterBasedTrainLoop)

        # input is a Loop object
        self.assertEqual(id(runner.build_train_loop(loop)), id(loop))

        # param_schedulers can be None
        cfg = dict(type='EpochBasedTrainLoop', max_epochs=3)
        runner.param_schedulers = None
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, EpochBasedTrainLoop)

        # test custom training loop
        cfg = dict(type='CustomTrainLoop', max_epochs=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, CustomTrainLoop)

    def test_build_val_loop(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_val_loop'
        runner = Runner.from_cfg(cfg)

        # input should be a Loop object or dict
        with self.assertRaisesRegex(TypeError, 'should be'):
            runner.build_test_loop('invalid-type')

        # input is a dict and contains type key
        cfg = dict(type='ValLoop')
        loop = runner.build_test_loop(cfg)
        self.assertIsInstance(loop, ValLoop)

        # input is a dict but does not contain type key
        cfg = dict()
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, ValLoop)

        # input is a Loop object
        self.assertEqual(id(runner.build_val_loop(loop)), id(loop))

        # test custom validation loop
        cfg = dict(type='CustomValLoop')
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, CustomValLoop)

    def test_build_test_loop(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_test_loop'
        runner = Runner.from_cfg(cfg)

        # input should be a Loop object or dict
        with self.assertRaisesRegex(TypeError, 'should be'):
            runner.build_test_loop('invalid-type')

        # input is a dict and contains type key
        cfg = dict(type='TestLoop')
        loop = runner.build_test_loop(cfg)
        self.assertIsInstance(loop, TestLoop)

        # input is a dict but does not contain type key
        cfg = dict()
        loop = runner.build_test_loop(cfg)
        self.assertIsInstance(loop, TestLoop)

        # input is a Loop object
        self.assertEqual(id(runner.build_test_loop(loop)), id(loop))

        # test custom validation loop
        cfg = dict(type='CustomTestLoop')
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, CustomTestLoop)

    def test_build_log_processor(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_log_processor'
        runner = Runner.from_cfg(cfg)

        # input should be a LogProcessor object or dict
        with self.assertRaisesRegex(TypeError, 'should be'):
            runner.build_log_processor('invalid-type')

        # input is a dict and contains type key
        cfg = dict(type='LogProcessor')
        log_processor = runner.build_log_processor(cfg)
        self.assertIsInstance(log_processor, LogProcessor)

        # input is a dict but does not contain type key
        cfg = dict()
        log_processor = runner.build_log_processor(cfg)
        self.assertIsInstance(log_processor, LogProcessor)

        # input is a LogProcessor object
        self.assertEqual(
            id(runner.build_log_processor(log_processor)), id(log_processor))

        # test custom validation log_processor
        cfg = dict(type='CustomLogProcessor')
        log_processor = runner.build_log_processor(cfg)
        self.assertIsInstance(log_processor, CustomLogProcessor)

    def test_train(self):
        # 1. test `self.train_loop` is None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train1'
        cfg.pop('train_dataloader')
        cfg.pop('train_cfg')
        cfg.pop('optim_wrapper')
        cfg.pop('param_scheduler')
        runner = Runner.from_cfg(cfg)
        with self.assertRaisesRegex(RuntimeError, 'should not be None'):
            runner.train()

        # 2. test iter and epoch counter of EpochBasedTrainLoop and timing of
        # running ValLoop
        epoch_results = []
        epoch_targets = [i for i in range(3)]
        iter_results = []
        iter_targets = [i for i in range(4 * 3)]
        batch_idx_results = []
        batch_idx_targets = [i for i in range(4)] * 3  # train and val
        val_epoch_results = []
        val_epoch_targets = [i for i in range(2, 4)]

        @HOOKS.register_module(force=True)
        class TestEpochHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner, batch_idx, data_batch=None):
                iter_results.append(runner.iter)
                batch_idx_results.append(batch_idx)

            def before_val_epoch(self, runner):
                val_epoch_results.append(runner.epoch)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train2'
        cfg.custom_hooks = [dict(type='TestEpochHook', priority=50)]
        cfg.train_cfg = dict(by_epoch=True, max_epochs=3, val_begin=2)
        runner = Runner.from_cfg(cfg)
        runner.train()
        self.assertEqual(runner.optim_wrapper._inner_count, 12)
        self.assertEqual(runner.optim_wrapper._max_counts, 12)

        assert isinstance(runner.train_loop, EpochBasedTrainLoop)

        for result, target, in zip(epoch_results, epoch_targets):
            self.assertEqual(result, target)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(batch_idx_results, batch_idx_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_epoch_results, val_epoch_targets):
            self.assertEqual(result, target)

        # 3. test iter and epoch counter of IterBasedTrainLoop and timing of
        # running ValLoop
        epoch_results = []
        iter_results = []
        batch_idx_results = []
        val_iter_results = []
        val_batch_idx_results = []
        iter_targets = [i for i in range(12)]
        batch_idx_targets = [i for i in range(12)]
        val_iter_targets = [i for i in range(4, 12)]
        val_batch_idx_targets = [i for i in range(4)] * 2

        @HOOKS.register_module(force=True)
        class TestIterHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner, batch_idx, data_batch=None):
                iter_results.append(runner.iter)
                batch_idx_results.append(batch_idx)

            def before_val_iter(self, runner, batch_idx, data_batch=None):
                val_epoch_results.append(runner.iter)
                val_batch_idx_results.append(batch_idx)

        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_train3'
        cfg.custom_hooks = [dict(type='TestIterHook', priority=50)]
        cfg.train_cfg = dict(
            by_epoch=False, max_iters=12, val_interval=4, val_begin=4)
        runner = Runner.from_cfg(cfg)
        runner.train()

        self.assertEqual(runner.optim_wrapper._inner_count, 12)
        self.assertEqual(runner.optim_wrapper._max_counts, 12)
        assert isinstance(runner.train_loop, IterBasedTrainLoop)

        self.assertEqual(len(epoch_results), 1)
        self.assertEqual(epoch_results[0], 0)
        self.assertEqual(runner.val_interval, 4)
        self.assertEqual(runner.val_begin, 4)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(batch_idx_results, batch_idx_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_iter_results, val_iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_batch_idx_results,
                                   val_batch_idx_targets):
            self.assertEqual(result, target)

        # 4. test iter and epoch counter of IterBasedTrainLoop and timing of
        # running ValLoop without InfiniteSampler
        epoch_results = []
        iter_results = []
        batch_idx_results = []
        val_iter_results = []
        val_batch_idx_results = []
        iter_targets = [i for i in range(12)]
        batch_idx_targets = [i for i in range(12)]
        val_iter_targets = [i for i in range(4, 12)]
        val_batch_idx_targets = [i for i in range(4)] * 2

        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_train4'
        cfg.train_dataloader.sampler = dict(
            type='DefaultSampler', shuffle=True)
        cfg.custom_hooks = [dict(type='TestIterHook', priority=50)]
        cfg.train_cfg = dict(
            by_epoch=False, max_iters=12, val_interval=4, val_begin=4)
        runner = Runner.from_cfg(cfg)
        # Warning should be raised since the sampler is not InfiniteSampler.
        with self.assertLogs(MMLogger.get_current_instance(), level='WARNING'):
            runner.train()

        assert isinstance(runner.train_loop, IterBasedTrainLoop)
        assert isinstance(runner.train_loop.dataloader_iterator,
                          _InfiniteDataloaderIterator)

        self.assertEqual(len(epoch_results), 1)
        self.assertEqual(epoch_results[0], 0)
        self.assertEqual(runner.val_interval, 4)
        self.assertEqual(runner.val_begin, 4)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(batch_idx_results, batch_idx_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_iter_results, val_iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_batch_idx_results,
                                   val_batch_idx_targets):
            self.assertEqual(result, target)

        # 5.1 test dynamic interval in IterBasedTrainLoop
        max_iters = 12
        interval = 5
        dynamic_intervals = [(11, 2)]
        iter_results = []
        iter_targets = [5, 10, 12]
        val_interval_results = []
        val_interval_targets = [5] * 10 + [2] * 2

        @HOOKS.register_module(force=True)
        class TestIterDynamicIntervalHook(Hook):

            def before_val(self, runner):
                iter_results.append(runner.iter)

            def before_train_iter(self, runner, batch_idx, data_batch=None):
                val_interval_results.append(runner.train_loop.val_interval)

        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_train5'
        cfg.train_dataloader.sampler = dict(
            type='DefaultSampler', shuffle=True)
        cfg.custom_hooks = [
            dict(type='TestIterDynamicIntervalHook', priority=50)
        ]
        cfg.train_cfg = dict(
            by_epoch=False,
            max_iters=max_iters,
            val_interval=interval,
            dynamic_intervals=dynamic_intervals)
        runner = Runner.from_cfg(cfg)
        runner.train()
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_interval_results, val_interval_targets):
            self.assertEqual(result, target)

        # 5.2 test dynamic interval in EpochBasedTrainLoop
        max_epochs = 12
        interval = 5
        dynamic_intervals = [(11, 2)]
        epoch_results = []
        epoch_targets = [5, 10, 12]
        val_interval_results = []
        val_interval_targets = [5] * 10 + [2] * 2

        @HOOKS.register_module(force=True)
        class TestEpochDynamicIntervalHook(Hook):

            def before_val_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_epoch(self, runner):
                val_interval_results.append(runner.train_loop.val_interval)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train6'
        cfg.train_dataloader.sampler = dict(
            type='DefaultSampler', shuffle=True)
        cfg.custom_hooks = [
            dict(type='TestEpochDynamicIntervalHook', priority=50)
        ]
        cfg.train_cfg = dict(
            by_epoch=True,
            max_epochs=max_epochs,
            val_interval=interval,
            dynamic_intervals=dynamic_intervals)
        runner = Runner.from_cfg(cfg)
        runner.train()
        for result, target, in zip(epoch_results, epoch_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_interval_results, val_interval_targets):
            self.assertEqual(result, target)

        # 7. test init weights
        @MODELS.register_module(force=True)
        class ToyModel2(ToyModel):

            def __init__(self):
                super().__init__()
                self.initiailzed = False

            def init_weights(self):
                self.initiailzed = True

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train7'
        runner = Runner.from_cfg(cfg)
        model = ToyModel2()
        runner.model = model
        runner.train()
        self.assertTrue(model.initiailzed)

        # 8.1 test train with multiple optimizer and single list of schedulers.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train8'
        cfg.param_scheduler = dict(type='MultiStepLR', milestones=[1, 2])
        cfg.optim_wrapper = dict(
            linear1=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            linear2=dict(
                type='OptimWrapper', optimizer=dict(type='Adam', lr=0.02)),
            constructor='ToyMultipleOptimizerConstructor')
        cfg.model = dict(type='ToyGANModel')
        runner = runner.from_cfg(cfg)
        runner.train()

        # 8.1 Test train with multiple optimizer and single schedulers.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train8.1.1'
        cfg.param_scheduler = dict(type='MultiStepLR', milestones=[1, 2])
        cfg.optim_wrapper = dict(
            linear1=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            linear2=dict(
                type='OptimWrapper', optimizer=dict(type='Adam', lr=0.02)),
            constructor='ToyMultipleOptimizerConstructor')
        cfg.model = dict(type='ToyGANModel')
        runner = runner.from_cfg(cfg)
        runner.train()

        # Test list like single scheduler.
        cfg.experiment_name = 'test_train8.1.2'
        cfg.param_scheduler = [dict(type='MultiStepLR', milestones=[1, 2])]
        runner = runner.from_cfg(cfg)
        runner.train()

        # 8.2 Test train with multiple optimizer and multiple schedulers.
        cfg.experiment_name = 'test_train8.2.1'
        cfg.param_scheduler = dict(
            linear1=dict(type='MultiStepLR', milestones=[1, 2]),
            linear2=dict(type='MultiStepLR', milestones=[1, 2]),
        )
        runner = runner.from_cfg(cfg)
        runner.train()

        cfg.experiment_name = 'test_train8.2.2'
        cfg.param_scheduler = dict(
            linear1=[dict(type='MultiStepLR', milestones=[1, 2])],
            linear2=[dict(type='MultiStepLR', milestones=[1, 2])],
        )
        runner = runner.from_cfg(cfg)
        runner.train()

        # 9 Test training with a dataset without metainfo
        cfg.experiment_name = 'test_train9'
        cfg = copy.deepcopy(cfg)
        cfg.train_dataloader.dataset = dict(type='ToyDatasetNoMeta')
        runner = runner.from_cfg(cfg)
        runner.train()

        # 10.1 Test build dataloader with default collate function
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_train10.1'
        cfg.train_dataloader.update(collate_fn=dict(type='default_collate'))
        runner = Runner.from_cfg(cfg)
        runner.train()

        # 10.2 Test build dataloader with custom collate function
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_train10.2'
        cfg.train_dataloader.update(
            collate_fn=dict(type='custom_collate', pad_value=100))
        runner = Runner.from_cfg(cfg)
        runner.train()

        # 10.3 Test build dataloader with custom worker_init function
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_train10.3'
        cfg.train_dataloader.update(
            worker_init_fn=dict(type='custom_worker_init'))
        runner = Runner.from_cfg(cfg)
        runner.train()

        # 11 test build dataloader without default arguments of collate
        # function.
        with self.assertRaises(TypeError):
            cfg = copy.deepcopy(self.iter_based_cfg)
            cfg.experiment_name = 'test_train11'
            cfg.train_dataloader.update(collate_fn=dict(type='custom_collate'))
            runner = Runner.from_cfg(cfg)
            runner.train()

        # 12.1 Test train with model, which does not inherit from BaseModel
        @MODELS.register_module(force=True)
        class ToyModel3(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1, 1)

            def train_step(self, *args, **kwargs):
                return dict(loss=torch.tensor(1))

        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.pop('val_cfg')
        cfg.pop('val_dataloader')
        cfg.pop('val_evaluator')
        cfg.model = dict(type='ToyModel3')
        cfg.experiment_name = 'test_train12.1'
        runner = Runner.from_cfg(cfg)
        runner.train()

        # 12.2 Test val_step should be implemented if val_cfg is not None
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.model = dict(type='ToyModel3')
        cfg.experiment_name = 'test_train12.2'
        runner = Runner.from_cfg(cfg)

        with self.assertRaisesRegex(AssertionError, 'If you want to validate'):
            runner.train()

        # 13 Test the logs will be printed when the length of
        # train_dataloader is smaller than the interval set in LoggerHook
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train13'
        cfg.default_hooks = dict(logger=dict(type='LoggerHook', interval=5))
        runner = Runner.from_cfg(cfg)
        runner.train()
        with open(runner.logger._log_file) as f:
            log = f.read()
        self.assertIn('Epoch(train) [1][4/4]', log)

        # 14. test_loop will not be built
        for cfg in (self.epoch_based_cfg, self.iter_based_cfg):
            cfg = copy.deepcopy(cfg)
            cfg.experiment_name = 'test_train14'
            runner = Runner.from_cfg(cfg)
            runner.train()
            self.assertIsInstance(runner._train_loop, BaseLoop)
            self.assertIsInstance(runner._val_loop, BaseLoop)
            self.assertIsInstance(runner._test_loop, dict)

        # 15. test num_batch_per_epoch
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train15'
        cfg.train_dataloader['num_batch_per_epoch'] = 2
        cfg.train_cfg = dict(
            by_epoch=True,
            max_epochs=3,
        )
        runner = Runner.from_cfg(cfg)
        runner.train()
        self.assertEqual(runner.iter, 3 * 2)

    @skipIf(
        SKIP_TEST_COMPILE,
        reason='torch.compile is not valid, please install PyTorch>=2.0.0')
    def test_train_with_compile(self):
        # 1. test with simple configuration
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train_compile_simple'
        cfg.compile = True
        runner = Runner.from_cfg(cfg)
        runner.train()

        # 2. test with advanced configuration
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train_compile_advanced'
        cfg.compile = dict(backend='inductor', mode='default')
        runner = Runner.from_cfg(cfg)
        runner.train()

        runner._maybe_compile('train_step')
        # PyTorch 2.0.0 could close the FileHandler after calling of
        # ``torch.compile``. So we need to test our file handler still works.
        with open(osp.join(f'{runner.log_dir}',
                           f'{runner.timestamp}.log')) as f:
            last_line = f.readlines()[-1]
            self.assertTrue(last_line.endswith('please be patient.\n'))

    def test_val(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_val1'
        cfg.pop('val_dataloader')
        cfg.pop('val_cfg')
        cfg.pop('val_evaluator')
        runner = Runner.from_cfg(cfg)
        with self.assertRaisesRegex(RuntimeError, 'should not be None'):
            runner.val()

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_val2'
        runner = Runner.from_cfg(cfg)
        runner.val()

        # test run val without train and test components
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_individually_val'
        cfg.pop('train_dataloader')
        cfg.pop('train_cfg')
        cfg.pop('optim_wrapper')
        cfg.pop('param_scheduler')
        cfg.pop('test_dataloader')
        cfg.pop('test_cfg')
        cfg.pop('test_evaluator')
        runner = Runner.from_cfg(cfg)

        # Test default fp32 `autocast` context.
        predictions = []

        def get_outputs_callback(module, inputs, outputs):
            predictions.append(outputs)

        runner.model.register_forward_hook(get_outputs_callback)
        runner.val()
        self.assertEqual(predictions[0].dtype, torch.float32)
        predictions.clear()

        # Test fp16 `autocast` context.
        cfg.experiment_name = 'test_val3'
        cfg.val_cfg = dict(fp16=True)
        runner = Runner.from_cfg(cfg)
        runner.model.register_forward_hook(get_outputs_callback)
        if (digit_version(TORCH_VERSION) < digit_version('1.10.0')
                and not torch.cuda.is_available()):
            with self.assertRaisesRegex(RuntimeError, 'If pytorch versions'):
                runner.val()
        else:
            runner.val()
            self.assertIn(predictions[0].dtype,
                          (torch.float16, torch.bfloat16))

        # train_loop and test_loop will not be built
        for cfg in (self.epoch_based_cfg, self.iter_based_cfg):
            cfg = copy.deepcopy(cfg)
            cfg.experiment_name = 'test_val4'
            runner = Runner.from_cfg(cfg)
            runner.val()
            self.assertIsInstance(runner._train_loop, dict)
            self.assertIsInstance(runner._test_loop, dict)

        # test num_batch_per_epoch
        val_result = 0

        @HOOKS.register_module(force=True)
        class TestIterHook(Hook):

            def __init__(self):
                self.val_iter = 0

            def after_val_iter(self,
                               runner,
                               batch_idx,
                               data_batch=None,
                               outputs=None):
                self.val_iter += 1
                nonlocal val_result
                val_result = self.val_iter

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks = [dict(type='TestIterHook', priority=50)]
        cfg.val_dataloader['num_batch_per_epoch'] = 2
        runner = Runner.from_cfg(cfg)
        runner.val()
        self.assertEqual(val_result, 2)

    @skipIf(
        SKIP_TEST_COMPILE,
        reason='torch.compile is not valid, please install PyTorch>=2.0.0')
    def test_val_with_compile(self):
        # 1. test with simple configuration
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_val_compile_simple'
        cfg.compile = True
        runner = Runner.from_cfg(cfg)
        runner.val()

        # 2. test with advanced configuration
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_val_compile_advanced'
        cfg.compile = dict(backend='inductor', mode='default')
        runner = Runner.from_cfg(cfg)
        runner.val()

    def test_test(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_test1'
        cfg.pop('test_dataloader')
        cfg.pop('test_cfg')
        cfg.pop('test_evaluator')
        runner = Runner.from_cfg(cfg)
        with self.assertRaisesRegex(RuntimeError, 'should not be None'):
            runner.test()

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_test2'
        runner = Runner.from_cfg(cfg)
        runner.test()
        # Test run test without building train loop.
        self.assertIsInstance(runner._train_loop, dict)

        # test run test without train and test components
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_individually_test'
        cfg.pop('train_dataloader')
        cfg.pop('train_cfg')
        cfg.pop('optim_wrapper')
        cfg.pop('param_scheduler')
        cfg.pop('val_dataloader')
        cfg.pop('val_cfg')
        cfg.pop('val_evaluator')
        runner = Runner.from_cfg(cfg)

        # Test default fp32 `autocast` context.
        predictions = []

        def get_outputs_callback(module, inputs, outputs):
            predictions.append(outputs)

        runner.model.register_forward_hook(get_outputs_callback)
        runner.test()
        self.assertEqual(predictions[0].dtype, torch.float32)
        predictions.clear()

        # Test fp16 `autocast` context.
        cfg.experiment_name = 'test_test3'
        cfg.test_cfg = dict(fp16=True)
        runner = Runner.from_cfg(cfg)
        runner.model.register_forward_hook(get_outputs_callback)
        if (digit_version(TORCH_VERSION) < digit_version('1.10.0')
                and not torch.cuda.is_available()):
            with self.assertRaisesRegex(RuntimeError, 'If pytorch versions'):
                runner.test()
        else:
            runner.test()
            self.assertIn(predictions[0].dtype,
                          (torch.float16, torch.bfloat16))
        # train_loop and val_loop will not be built
        for cfg in (self.epoch_based_cfg, self.iter_based_cfg):
            cfg = copy.deepcopy(cfg)
            cfg.experiment_name = 'test_test4'
            runner = Runner.from_cfg(cfg)
            runner.test()
            self.assertIsInstance(runner._train_loop, dict)
            self.assertIsInstance(runner._val_loop, dict)

        # test num_batch_per_epoch
        test_result = 0

        @HOOKS.register_module(force=True)
        class TestIterHook(Hook):

            def __init__(self):
                self.test_iter = 0

            def after_test_iter(self,
                                runner,
                                batch_idx,
                                data_batch=None,
                                outputs=None):
                self.test_iter += 1
                nonlocal test_result
                test_result = self.test_iter

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks = [dict(type='TestIterHook', priority=50)]
        cfg.test_dataloader['num_batch_per_epoch'] = 2
        runner = Runner.from_cfg(cfg)
        runner.test()
        self.assertEqual(test_result, 2)

    @skipIf(
        SKIP_TEST_COMPILE,
        reason='torch.compile is not valid, please install PyTorch>=2.0.0')
    def test_test_with_compile(self):
        # 1. test with simple configuration
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_test_compile_simple'
        cfg.compile = True
        runner = Runner.from_cfg(cfg)
        runner.test()

        # 2. test with advanced configuration
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_test_compile_advanced'
        cfg.compile = dict(backend='inductor', mode='default')
        runner = Runner.from_cfg(cfg)
        runner.test()

    def test_register_hook(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_register_hook'
        runner = Runner.from_cfg(cfg)
        runner._hooks = []

        # 1. test `hook` parameter
        # 1.1 `hook` should be either a Hook object or dict
        with self.assertRaisesRegex(
                TypeError, 'hook should be an instance of Hook or dict'):
            runner.register_hook(['string'])

        # 1.2 `hook` is a dict
        timer_cfg = dict(type='IterTimerHook')
        runner.register_hook(timer_cfg)
        self.assertEqual(len(runner._hooks), 1)
        self.assertTrue(isinstance(runner._hooks[0], IterTimerHook))
        # default priority of `IterTimerHook` is 'NORMAL'
        self.assertEqual(
            get_priority(runner._hooks[0].priority), get_priority('NORMAL'))

        runner._hooks = []
        # 1.2.1 `hook` is a dict and contains `priority` field
        # set the priority of `IterTimerHook` as 'BELOW_NORMAL'
        timer_cfg = dict(type='IterTimerHook', priority='BELOW_NORMAL')
        runner.register_hook(timer_cfg)
        self.assertEqual(len(runner._hooks), 1)
        self.assertTrue(isinstance(runner._hooks[0], IterTimerHook))
        self.assertEqual(
            get_priority(runner._hooks[0].priority),
            get_priority('BELOW_NORMAL'))

        # 1.3 `hook` is a hook object
        runtime_info_hook = RuntimeInfoHook()
        runner.register_hook(runtime_info_hook)
        self.assertEqual(len(runner._hooks), 2)
        # The priority of `runtime_info_hook` is `HIGH` which is greater than
        # `IterTimerHook`, so the first item of `_hooks` should be
        # `runtime_info_hook`
        self.assertTrue(isinstance(runner._hooks[0], RuntimeInfoHook))
        self.assertEqual(
            get_priority(runner._hooks[0].priority), get_priority('VERY_HIGH'))

        # 2. test `priority` parameter
        # `priority` argument is not None and it will be set as priority of
        # hook
        param_scheduler_cfg = dict(type='ParamSchedulerHook', priority='LOW')
        runner.register_hook(param_scheduler_cfg, priority='VERY_LOW')
        self.assertEqual(len(runner._hooks), 3)
        self.assertTrue(isinstance(runner._hooks[2], ParamSchedulerHook))
        self.assertEqual(
            get_priority(runner._hooks[2].priority), get_priority('VERY_LOW'))

        # `priority` is Priority
        logger_cfg = dict(type='LoggerHook', priority='BELOW_NORMAL')
        runner.register_hook(logger_cfg, priority=Priority.VERY_LOW)
        self.assertEqual(len(runner._hooks), 4)
        self.assertTrue(isinstance(runner._hooks[3], LoggerHook))
        self.assertEqual(
            get_priority(runner._hooks[3].priority), get_priority('VERY_LOW'))

    def test_default_hooks(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_default_hooks'
        runner = Runner.from_cfg(cfg)
        runner._hooks = []

        # register 7 hooks by default
        runner.register_default_hooks()
        self.assertEqual(len(runner._hooks), 6)
        # the third registered hook should be `DistSamplerSeedHook`
        self.assertTrue(isinstance(runner._hooks[2], DistSamplerSeedHook))
        # the fifth registered hook should be `ParamSchedulerHook`
        self.assertTrue(isinstance(runner._hooks[4], ParamSchedulerHook))

        runner._hooks = []
        # remove `ParamSchedulerHook` from default hooks
        runner.register_default_hooks(hooks=dict(timer=None))
        self.assertEqual(len(runner._hooks), 5)
        # `ParamSchedulerHook` was popped so the fifth is `CheckpointHook`
        self.assertTrue(isinstance(runner._hooks[4], CheckpointHook))

        # add a new default hook
        runner._hooks = []
        runner.register_default_hooks(hooks=dict(ToyHook=dict(type='ToyHook')))
        self.assertEqual(len(runner._hooks), 7)
        self.assertTrue(isinstance(runner._hooks[6], ToyHook))

    def test_custom_hooks(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_custom_hooks'
        runner = Runner.from_cfg(cfg)

        self.assertEqual(len(runner._hooks), 6)
        custom_hooks = [dict(type='ToyHook')]
        runner.register_custom_hooks(custom_hooks)
        self.assertEqual(len(runner._hooks), 7)
        self.assertTrue(isinstance(runner._hooks[6], ToyHook))

    def test_register_hooks(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_register_hooks'
        runner = Runner.from_cfg(cfg)

        runner._hooks = []
        custom_hooks = [dict(type='ToyHook')]
        runner.register_hooks(custom_hooks=custom_hooks)
        # six default hooks + custom hook (ToyHook)
        self.assertEqual(len(runner._hooks), 7)
        self.assertTrue(isinstance(runner._hooks[6], ToyHook))

    def test_custom_loop(self):
        # test custom loop with additional hook
        @LOOPS.register_module(force=True)
        class CustomTrainLoop2(IterBasedTrainLoop):
            """Custom train loop with additional warmup stage."""

            def __init__(self, runner, dataloader, max_iters, warmup_loader,
                         max_warmup_iters):
                super().__init__(
                    runner=runner, dataloader=dataloader, max_iters=max_iters)
                self.warmup_loader = self.runner.build_dataloader(
                    warmup_loader)
                self.max_warmup_iters = max_warmup_iters

            def run(self):
                self.runner.call_hook('before_train')
                self.runner.cur_dataloader = self.warmup_loader
                for idx, data_batch in enumerate(self.warmup_loader, 1):
                    self.warmup_iter(data_batch)
                    if idx == self.max_warmup_iters:
                        break

                self.runner.cur_dataloader = self.warmup_loader
                self.runner.call_hook('before_train_epoch')
                while self.runner.iter < self._max_iters:
                    data_batch = next(self.dataloader_iterator)
                    self.run_iter(data_batch)
                self.runner.call_hook('after_train_epoch')

                self.runner.call_hook('after_train')

            def warmup_iter(self, data_batch):
                self.runner.call_hook(
                    'before_warmup_iter', data_batch=data_batch)
                train_logs = self.runner.model.train_step(
                    data_batch, self.runner.optim_wrapper)
                self.runner.message_hub.update_info('train_logs', train_logs)
                self.runner.call_hook(
                    'after_warmup_iter', data_batch=data_batch)

        before_warmup_iter_results = []
        after_warmup_iter_results = []

        @HOOKS.register_module(force=True)
        class TestWarmupHook(Hook):
            """test custom train loop."""

            def before_warmup_iter(self, runner, data_batch=None):
                before_warmup_iter_results.append('before')

            def after_warmup_iter(self, runner, data_batch=None, outputs=None):
                after_warmup_iter_results.append('after')

        self.iter_based_cfg.train_cfg = dict(
            type='CustomTrainLoop2',
            max_iters=10,
            warmup_loader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='InfiniteSampler', shuffle=True),
                batch_size=1,
                num_workers=0),
            max_warmup_iters=5)
        self.iter_based_cfg.custom_hooks = [
            dict(type='TestWarmupHook', priority=50)
        ]
        self.iter_based_cfg.experiment_name = 'test_custom_loop'
        runner = Runner.from_cfg(self.iter_based_cfg)
        runner.train()

        self.assertIsInstance(runner.train_loop, CustomTrainLoop2)

        # test custom hook triggered as expected
        self.assertEqual(len(before_warmup_iter_results), 5)
        self.assertEqual(len(after_warmup_iter_results), 5)
        for before, after in zip(before_warmup_iter_results,
                                 after_warmup_iter_results):
            self.assertEqual(before, 'before')
            self.assertEqual(after, 'after')

    def test_checkpoint(self):
        # 1. test epoch based
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint1'
        runner = Runner.from_cfg(cfg)
        runner.train()

        # 1.1 test `save_checkpoint` which is called by `CheckpointHook`
        path = osp.join(self.temp_dir, 'epoch_3.pth')
        self.assertTrue(osp.exists(path))
        self.assertFalse(osp.exists(osp.join(self.temp_dir, 'epoch_4.pth')))

        ckpt = torch.load(path)
        self.assertEqual(ckpt['meta']['epoch'], 3)
        self.assertEqual(ckpt['meta']['iter'], 12)
        self.assertEqual(ckpt['meta']['experiment_name'],
                         runner.experiment_name)
        self.assertEqual(ckpt['meta']['seed'], runner.seed)
        assert isinstance(ckpt['optimizer'], dict)
        assert isinstance(ckpt['param_schedulers'], list)
        self.assertIsInstance(ckpt['message_hub'], dict)
        message_hub = MessageHub.get_instance('test_ckpt')
        message_hub.load_state_dict(ckpt['message_hub'])
        self.assertEqual(message_hub.get_info('epoch'), 2)
        self.assertEqual(message_hub.get_info('iter'), 11)

        # 1.2 test `load_checkpoint`
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint2'
        cfg.optim_wrapper = dict(type='SGD', lr=0.2)
        cfg.param_scheduler = dict(type='MultiStepLR', milestones=[1, 2, 3])
        runner = Runner.from_cfg(cfg)
        runner.load_checkpoint(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 0)
        self.assertTrue(runner._has_loaded)
        # load checkpoint will not initialize optimizer and param_schedulers
        # objects
        self.assertIsInstance(runner.optim_wrapper, dict)
        self.assertIsInstance(runner.param_schedulers, dict)

        # 1.3.1 test `resume`
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint3'
        cfg.optim_wrapper = dict(
            type='OptimWrapper', optimizer=dict(type='SGD', lr=0.2))
        cfg.param_scheduler = dict(type='MultiStepLR', milestones=[1, 2, 3])
        runner = Runner.from_cfg(cfg)
        runner.resume(path)
        self.assertEqual(runner.epoch, 3)
        self.assertEqual(runner.iter, 12)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optim_wrapper.optimizer, SGD)
        self.assertIsInstance(runner.optim_wrapper.optimizer, SGD)
        self.assertEqual(runner.optim_wrapper.param_groups[0]['lr'], 0.0001)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)
        self.assertEqual(runner.param_schedulers[0].milestones, {1: 1, 2: 1})
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertEqual(runner.message_hub.get_info('epoch'), 2)
        self.assertEqual(runner.message_hub.get_info('iter'), 11)
        self.assertEqual(MessageHub.get_current_instance().get_info('epoch'),
                         2)
        self.assertEqual(MessageHub.get_current_instance().get_info('iter'),
                         11)

        # 1.3.2 test resume with unmatched dataset_meta
        ckpt_modified = copy.deepcopy(ckpt)
        ckpt_modified['meta']['dataset_meta'] = {'CLASSES': ['cat', 'dog']}
        # ckpt_modified['meta']['seed'] = 123
        path_modified = osp.join(self.temp_dir, 'modified.pth')
        torch.save(ckpt_modified, path_modified)
        # Warning should be raised since dataset_meta is not matched
        with self.assertLogs(MMLogger.get_current_instance(), level='WARNING'):
            runner.resume(path_modified)

        # 1.3.3 test resume with unmatched seed
        ckpt_modified = copy.deepcopy(ckpt)
        ckpt_modified['meta']['seed'] = 123
        path_modified = osp.join(self.temp_dir, 'modified.pth')
        torch.save(ckpt_modified, path_modified)
        # Warning should be raised since seed is not matched
        with self.assertLogs(MMLogger.get_current_instance(), level='WARNING'):
            runner.resume(path_modified)

        # 1.3.3 test resume with no seed and dataset meta
        ckpt_modified = copy.deepcopy(ckpt)
        ckpt_modified['meta'].pop('seed')
        ckpt_modified['meta'].pop('dataset_meta')
        path_modified = osp.join(self.temp_dir, 'modified.pth')
        torch.save(ckpt_modified, path_modified)
        runner.resume(path_modified)

        # 1.4 test auto resume
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint4'
        cfg.resume = True
        runner = Runner.from_cfg(cfg)
        runner.load_or_resume()
        self.assertEqual(runner.epoch, 3)
        self.assertEqual(runner.iter, 12)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optim_wrapper.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)

        # 1.5 test resume from a specified checkpoint
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint5'
        cfg.resume = True
        cfg.load_from = osp.join(self.temp_dir, 'epoch_1.pth')
        runner = Runner.from_cfg(cfg)
        runner.load_or_resume()
        self.assertEqual(runner.epoch, 1)
        self.assertEqual(runner.iter, 4)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optim_wrapper.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)

        # 1.6 multiple optimizers
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint6'
        cfg.optim_wrapper = dict(
            linear1=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            linear2=dict(
                type='OptimWrapper', optimizer=dict(type='Adam', lr=0.02)),
            constructor='ToyMultipleOptimizerConstructor')
        cfg.model = dict(type='ToyGANModel')
        # disable OptimizerHook because it only works with one optimizer
        runner = Runner.from_cfg(cfg)
        runner.train()
        path = osp.join(self.temp_dir, 'epoch_3.pth')
        self.assertTrue(osp.exists(path))
        self.assertEqual(runner.optim_wrapper['linear1'].param_groups[0]['lr'],
                         0.0001)
        self.assertIsInstance(runner.optim_wrapper['linear2'].optimizer, Adam)
        self.assertEqual(runner.optim_wrapper['linear2'].param_groups[0]['lr'],
                         0.0002)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint7'
        cfg.optim_wrapper = dict(
            linear1=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.2)),
            linear2=dict(
                type='OptimWrapper', optimizer=dict(type='Adam', lr=0.03)),
            constructor='ToyMultipleOptimizerConstructor')
        cfg.model = dict(type='ToyGANModel')
        cfg.param_scheduler = dict(type='MultiStepLR', milestones=[1, 2, 3])
        runner = Runner.from_cfg(cfg)
        runner.resume(path)
        self.assertIsInstance(runner.optim_wrapper, OptimWrapperDict)
        self.assertIsInstance(runner.optim_wrapper['linear1'].optimizer, SGD)
        self.assertEqual(runner.optim_wrapper['linear1'].param_groups[0]['lr'],
                         0.0001)
        self.assertIsInstance(runner.optim_wrapper['linear2'].optimizer, Adam)
        self.assertEqual(runner.optim_wrapper['linear2'].param_groups[0]['lr'],
                         0.0002)
        self.assertIsInstance(runner.param_schedulers, dict)
        self.assertEqual(len(runner.param_schedulers['linear1']), 1)
        self.assertIsInstance(runner.param_schedulers['linear1'][0],
                              MultiStepLR)
        self.assertEqual(runner.param_schedulers['linear1'][0].milestones, {
            1: 1,
            2: 1
        })
        self.assertEqual(len(runner.param_schedulers['linear2']), 1)
        self.assertIsInstance(runner.param_schedulers['linear2'][0],
                              MultiStepLR)
        self.assertEqual(runner.param_schedulers['linear2'][0].milestones, {
            1: 1,
            2: 1
        })

        # 2. test iter based
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_checkpoint8'
        runner = Runner.from_cfg(cfg)
        runner.train()

        # 2.1.1 test `save_checkpoint` which is called by `CheckpointHook`
        path = osp.join(self.temp_dir, 'iter_12.pth')
        self.assertTrue(osp.exists(path))
        self.assertFalse(osp.exists(osp.join(self.temp_dir, 'epoch_13.pth')))

        ckpt = torch.load(path)
        self.assertEqual(ckpt['meta']['epoch'], 0)
        self.assertEqual(ckpt['meta']['iter'], 12)
        assert isinstance(ckpt['optimizer'], dict)
        assert isinstance(ckpt['param_schedulers'], list)
        self.assertIsInstance(ckpt['message_hub'], dict)
        message_hub.load_state_dict(ckpt['message_hub'])
        self.assertEqual(message_hub.get_info('epoch'), 0)
        self.assertEqual(message_hub.get_info('iter'), 11)
        # 2.1.2 check class attribute _statistic_methods can be saved
        HistoryBuffer._statistics_methods.clear()
        ckpt = torch.load(path)
        self.assertIn('min', HistoryBuffer._statistics_methods)

        # 2.2 test `load_checkpoint`
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_checkpoint9'
        runner = Runner.from_cfg(cfg)
        runner.load_checkpoint(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 0)
        self.assertTrue(runner._has_loaded)

        # 2.3 test `resume`
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_checkpoint10'
        runner = Runner.from_cfg(cfg)
        runner.resume(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 12)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optim_wrapper.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)
        self.assertEqual(runner.message_hub.get_info('epoch'), 0)
        self.assertEqual(runner.message_hub.get_info('iter'), 11)

        # 2.4 test auto resume
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_checkpoint11'
        cfg.resume = True
        runner = Runner.from_cfg(cfg)
        runner.load_or_resume()
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 12)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optim_wrapper.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)

        # 2.5 test resume from a specified checkpoint
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_checkpoint12'
        cfg.resume = True
        cfg.load_from = osp.join(self.temp_dir, 'iter_3.pth')
        runner = Runner.from_cfg(cfg)
        runner.load_or_resume()
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 3)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optim_wrapper.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)

        # 2.6 test resumed message_hub has the history value.
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_checkpoint13'
        cfg.resume = True
        cfg.load_from = osp.join(self.temp_dir, 'iter_3.pth')
        runner = Runner.from_cfg(cfg)
        runner.load_or_resume()
        assert len(runner.message_hub.log_scalars['train/lr'].data[1]) == 3
        assert len(MessageHub.get_current_instance().log_scalars['train/lr'].
                   data[1]) == 3

        # 2.7.1 test `resume` 2 optimizers and 1 scheduler list.
        path = osp.join(self.temp_dir, 'epoch_3.pth')
        optim_cfg = dict(
            linear1=dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01)),
            linear2=dict(
                type='OptimWrapper', optimizer=dict(type='Adam', lr=0.02)),
            constructor='ToyMultipleOptimizerConstructor')
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint14'
        cfg.optim_wrapper = optim_cfg
        cfg.param_scheduler = dict(type='MultiStepLR', milestones=[1, 2, 3])
        cfg.model = dict(type='ToyGANModel')
        resumed_cfg = copy.deepcopy(cfg)
        runner = Runner.from_cfg(cfg)
        runner.train()
        resumed_cfg.experiment_name = 'test_checkpoint15'
        runner = Runner.from_cfg(resumed_cfg)
        runner.resume(path)
        self.assertEqual(len(runner.param_schedulers['linear1']), 1)
        self.assertEqual(len(runner.param_schedulers['linear2']), 1)
        self.assertIsInstance(runner.param_schedulers['linear1'][0],
                              MultiStepLR)
        self.assertIsInstance(runner.param_schedulers['linear2'][0],
                              MultiStepLR)

        # 2.7.2 test `resume` 2 optimizers and 2 scheduler list.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint16'
        cfg.optim_wrapper = optim_cfg
        cfg.param_scheduler = dict(
            linear1=dict(type='MultiStepLR', milestones=[1, 2, 3]),
            linear2=dict(type='StepLR', gamma=0.1, step_size=3))
        cfg.model = dict(type='ToyGANModel')
        resumed_cfg = copy.deepcopy(cfg)
        runner = Runner.from_cfg(cfg)
        runner.train()
        resumed_cfg.experiment_name = 'test_checkpoint17'
        runner = Runner.from_cfg(resumed_cfg)
        runner.resume(path)
        self.assertEqual(len(runner.param_schedulers['linear1']), 1)
        self.assertEqual(len(runner.param_schedulers['linear2']), 1)
        self.assertIsInstance(runner.param_schedulers['linear1'][0],
                              MultiStepLR)
        self.assertIsInstance(runner.param_schedulers['linear2'][0], StepLR)

        # 2.7.3 test `resume` 2 optimizers and 0 scheduler list.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint18'
        cfg.optim_wrapper = optim_cfg
        cfg.model = dict(type='ToyGANModel')
        cfg.param_scheduler = None
        resumed_cfg = copy.deepcopy(cfg)
        runner = Runner.from_cfg(cfg)
        runner.train()
        resumed_cfg.experiment_name = 'test_checkpoint19'
        runner = Runner.from_cfg(resumed_cfg)
        runner.resume(path)
        self.assertIsNone(runner.param_schedulers)

    def test_build_runner(self):
        # No need to test other cases which have been tested in
        # `test_build_from_cfg`
        # test custom runner
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_runner1'
        cfg.runner_type = 'CustomRunner'
        assert isinstance(RUNNERS.build(cfg), CustomRunner)

        # test default runner
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_runner2'
        assert isinstance(RUNNERS.build(cfg), Runner)

    def test_get_hooks_info(self):
        # test get_hooks_info() function
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_get_hooks_info_from_test_runner_py'
        cfg.runner_type = 'Runner'
        runner = RUNNERS.build(cfg)
        self.assertIsInstance(runner, Runner)
        target_str = ('after_train_iter:\n'
                      '(VERY_HIGH   ) RuntimeInfoHook                    \n'
                      '(NORMAL      ) IterTimerHook                      \n'
                      '(BELOW_NORMAL) LoggerHook                         \n'
                      '(LOW         ) ParamSchedulerHook                 \n'
                      '(VERY_LOW    ) CheckpointHook                     \n')
        self.assertIn(target_str, runner.get_hooks_info(),
                      'target string is not in logged hooks information.')
