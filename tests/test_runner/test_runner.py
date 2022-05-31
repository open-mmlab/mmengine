# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import shutil
import tempfile
from unittest import TestCase

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset

from mmengine.config import Config
from mmengine.data import DefaultSampler
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, Hook,
                            IterTimerHook, LoggerHook, ParamSchedulerHook,
                            RuntimeInfoHook)
from mmengine.logging import LogProcessor, MessageHub, MMLogger
from mmengine.optim import (AmpOptimWrapper, DefaultOptimWrapperConstructor,
                            MultiStepLR, OptimWrapper, OptimWrapperDict,
                            StepLR)
from mmengine.registry import (DATASETS, HOOKS, LOG_PROCESSORS, LOOPS, METRICS,
                               MODEL_WRAPPERS, MODELS,
                               OPTIM_WRAPPER_CONSTRUCTORS, PARAM_SCHEDULERS,
                               Registry)
from mmengine.runner import (BaseLoop, EpochBasedTrainLoop, IterBasedTrainLoop,
                             Runner, TestLoop, ValLoop)
from mmengine.runner.priority import Priority, get_priority
from mmengine.utils import is_list_of
from mmengine.visualization import Visualizer


@MODELS.register_module()
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, data_batch, return_loss=False):
        inputs, labels = [], []
        for x in data_batch:
            inputs.append(x['inputs'])
            labels.append(x['data_sample'])

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = torch.stack(inputs).to(device)
        labels = torch.stack(labels).to(device)
        outputs = self.linear1(inputs)
        outputs = self.linear2(outputs)
        if return_loss:
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss, log_vars=dict(loss=loss.item()))
            return outputs
        else:
            outputs = dict(log_vars=dict(a=1, b=0.5))
            return outputs


@MODELS.register_module()
class ToyModel1(ToyModel):

    def __init__(self):
        super().__init__()


@MODEL_WRAPPERS.register_module()
class CustomModelWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class ToyMultipleOptimizerConstructor:

    def __init__(self, optimizer_cfg, paramwise_cfg=None):
        if not isinstance(optimizer_cfg, dict):
            raise TypeError('optimizer_cfg should be a dict',
                            f'but got {type(optimizer_cfg)}')
        assert paramwise_cfg is None, (
            'parawise_cfg should be set in each optimizer separately')
        self.optimizer_cfg = optimizer_cfg
        self.constructors = {}
        for key, cfg in self.optimizer_cfg.items():
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


@DATASETS.register_module()
class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


@METRICS.register_module()
class ToyMetric1(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


@METRICS.register_module()
class ToyMetric2(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


@HOOKS.register_module()
class ToyHook(Hook):
    priority = 'Lowest'

    def before_train_epoch(self, runner):
        pass


@HOOKS.register_module()
class ToyHook2(Hook):
    priority = 'Lowest'

    def after_train_epoch(self, runner):
        pass


@LOOPS.register_module()
class CustomTrainLoop(BaseLoop):

    def __init__(self, runner, dataloader, max_epochs):
        super().__init__(runner, dataloader)
        self._max_epochs = max_epochs

    def run(self) -> None:
        pass


@LOOPS.register_module()
class CustomValLoop(BaseLoop):

    def __init__(self, runner, dataloader, evaluator, interval=1):
        super().__init__(runner, dataloader)
        self._runner = runner

        if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator

    def run(self) -> None:
        pass


@LOOPS.register_module()
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


@LOG_PROCESSORS.register_module()
class CustomLogProcessor(LogProcessor):

    def __init__(self, window_size=10, by_epoch=True, custom_cfg=None):
        self.window_size = window_size
        self.by_epoch = by_epoch
        self.custom_cfg = custom_cfg if custom_cfg else []
        self._check_custom_cfg()


def collate_fn(data_batch):
    return data_batch


class TestRunner(TestCase):

    def setUp(self):
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
            optimizer=dict(type='SGD', lr=0.01),
            param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
            val_evaluator=dict(type='ToyMetric1'),
            test_evaluator=dict(type='ToyMetric1'),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            val_cfg=dict(interval=1, begin=1),
            test_cfg=dict(),
            custom_hooks=[],
            default_hooks=dict(
                runtime_info=dict(type='RuntimeInfoHook'),
                optimizer=dict(type='OptimizerHook', grad_clip=None),
                timer=dict(type='IterTimerHook'),
                logger=dict(type='LoggerHook'),
                param_scheduler=dict(type='ParamSchedulerHook'),
                checkpoint=dict(
                    type='CheckpointHook', interval=1, by_epoch=True),
                sampler_seed=dict(type='DistSamplerSeedHook')),
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
            optimizer=dict(type='OptimizerHook', grad_clip=None),
            timer=dict(type='IterTimerHook'),
            logger=dict(type='LoggerHook'),
            param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=False),
            sampler_seed=dict(type='DistSamplerSeedHook'))

    def tearDown(self):
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
        cfg.pop('optimizer')
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
        self.assertEqual(runner.param_schedulers, [])

        # param_scheduler should be None when optimizer is None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init5'
        cfg.pop('train_cfg')
        cfg.pop('train_dataloader')
        cfg.pop('optimizer')
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
        self.assertIsInstance(runner._optimizer, dict)
        self.assertIsInstance(runner.param_schedulers[0], dict)

        # After calling runner.train(),
        # train_dataloader and val_loader should be initialized but
        # test_dataloader should also be dict
        runner.train()

        self.assertIsInstance(runner._train_loop, BaseLoop)
        self.assertIsInstance(runner.train_dataloader, DataLoader)
        self.assertIsInstance(runner.optim_wrapper, OptimWrapper)
        self.assertIsInstance(runner._optimizer, dict)
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
        optimizer = SGD(
            model.parameters(),
            lr=0.01,
        )
        toy_hook = ToyHook()
        toy_hook2 = ToyHook2()

        train_dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        val_dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        test_dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        runner = Runner(
            model=model,
            work_dir=self.temp_dir,
            train_cfg=dict(by_epoch=True, max_epochs=3),
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
            val_cfg=dict(interval=1, begin=1),
            val_dataloader=val_dataloader,
            val_evaluator=ToyMetric1(),
            test_cfg=dict(),
            test_dataloader=test_dataloader,
            test_evaluator=ToyMetric1(),
            default_hooks=dict(param_scheduler=toy_hook),
            custom_hooks=[toy_hook2],
            experiment_name='test_init14')
        runner.train()
        runner.test()

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
                temp_config_file = tempfile.NamedTemporaryFile(
                    dir=temp_config_dir, suffix='.py')
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

        @TOY_SCHEDULERS.register_module()
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
        cfg.experiment_name = 'test_build_model'
        runner = Runner.from_cfg(cfg)
        self.assertIsInstance(runner.model, ToyModel)

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

        # test init weights
        @MODELS.register_module()
        class ToyModel2(ToyModel):

            def __init__(self):
                super().__init__()
                self.initiailzed = False

            def init_weights(self):
                self.initiailzed = True

        model = runner.build_model(dict(type='ToyModel2'))
        self.assertTrue(model.initiailzed)

        # test init weights with model object
        _model = ToyModel2()
        model = runner.build_model(_model)
        self.assertFalse(model.initiailzed)

    def test_wrap_model(self):
        # TODO: test on distributed environment
        # custom model wrapper
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_wrap_model'
        cfg.model_wrapper_cfg = dict(type='CustomModelWrapper')
        runner = Runner.from_cfg(cfg)
        self.assertIsInstance(runner.model, CustomModelWrapper)

    def test_build_optim_wrapper(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_optim_wrapper'
        runner = Runner.from_cfg(cfg)

        # input should be an Optimizer object or dict
        with self.assertRaisesRegex(TypeError, 'optimizer should be'):
            runner.build_optim_wrapper('invalid-type')

        # 1. test one optimizer
        # 1.1 input is an Optimizer object
        _optimizer = SGD(runner.model.parameters(), lr=0.01)
        optim_wrapper = runner.build_optim_wrapper(_optimizer)
        self.assertEqual(id(_optimizer), id(optim_wrapper.optimizer))

        # 1.2 input is a dict
        optim_wrapper = runner.build_optim_wrapper(dict(type='SGD', lr=0.01))
        self.assertIsInstance(optim_wrapper, OptimWrapper)

        # 1.3 build custom optimizer wrapper
        optim_wrapper = runner.build_optim_wrapper(
            dict(
                type='SGD',
                lr=0.01,
                optim_wrapper=dict(type='AmpOptimWrapper')))
        self.assertIsInstance(optim_wrapper, AmpOptimWrapper)

        # 2. test multiple optmizers
        # 2.1 input is a dict which contains multiple optimizer objects
        optimizer1 = SGD(runner.model.linear1.parameters(), lr=0.01)
        optimizer2 = Adam(runner.model.linear2.parameters(), lr=0.02)
        optim_cfg = dict(key1=optimizer1, key2=optimizer2)
        optim_wrapper = runner.build_optim_wrapper(optim_cfg)
        self.assertIsInstance(optim_wrapper, OptimWrapperDict)
        self.assertIsInstance(optim_wrapper['key1'].optimizer, SGD)
        self.assertIsInstance(optim_wrapper['key2'].optimizer, Adam)

        # 2.2 each item mush be an optimizer object when "type" and
        # "constructor" are not in optimizer
        optimizer1 = SGD(runner.model.linear1.parameters(), lr=0.01)
        optimizer2 = dict(type='Adam', lr=0.02)
        optim_cfg = dict(key1=optimizer1, key2=optimizer2)
        with self.assertRaisesRegex(ValueError,
                                    'each item mush be an optimizer object'):
            runner.build_optim_wrapper(optim_cfg)

        # 2.3 input is a dict which contains multiple configs
        optim_cfg = dict(
            linear1=dict(type='SGD', lr=0.01),
            linear2=dict(type='Adam', lr=0.02),
            constructor='ToyMultipleOptimizerConstructor')
        optim_wrapper = runner.build_optim_wrapper(optim_cfg)
        self.assertIsInstance(optim_wrapper, OptimWrapperDict)
        self.assertIsInstance(optim_wrapper['linear1'].optimizer, SGD)
        self.assertIsInstance(optim_wrapper['linear2'].optimizer, Adam)

    def test_build_param_scheduler(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_param_scheduler'
        runner = Runner.from_cfg(cfg)

        # `build_optim_wrapper` should be called before
        # `build_param_scheduler`
        cfg = dict(type='MultiStepLR', milestones=[1, 2])
        runner.optim_wrapper = dict(
            key1=dict(type='SGD', lr=0.01),
            key2=dict(type='Adam', lr=0.02),
        )
        with self.assertRaisesRegex(AssertionError, 'should be called before'):
            runner.build_param_scheduler(cfg)

        runner.optim_wrapper = runner.build_optim_wrapper(
            dict(type='SGD', lr=0.01))
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
        optimizer2 = Adam(runner.model.linear2.parameters(), lr=0.02)
        optim_cfg = dict(key1=optimizer1, key2=optimizer2)
        runner.optim_wrapper = runner.build_optim_wrapper(optim_cfg)
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
            dict(type='SGD', lr=0.01))

        # 5.1 train loop should be built before converting scheduler
        cfg = dict(
            type='MultiStepLR', milestones=[1, 2], convert_to_iter_based=True)
        with self.assertRaisesRegex(
                AssertionError,
                'Scheduler can only be converted to iter-based when '
                'train loop is built.'):
            runner.build_param_scheduler(cfg)

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

        # test collect device
        evaluator = [
            dict(type='ToyMetric1', collect_device='cpu'),
            dict(type='ToyMetric2', collect_device='gpu')
        ]
        _evaluator = runner.build_evaluator(evaluator)
        self.assertEqual(_evaluator.metrics[0].collect_device, 'cpu')
        self.assertEqual(_evaluator.metrics[1].collect_device, 'gpu')

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

        # param_schedulers can be []
        cfg = dict(type='EpochBasedTrainLoop', max_epochs=3)
        runner.param_schedulers = []
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
        cfg = dict(type='ValLoop', interval=1, begin=1)
        loop = runner.build_test_loop(cfg)
        self.assertIsInstance(loop, ValLoop)

        # input is a dict but does not contain type key
        cfg = dict(interval=1, begin=1)
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, ValLoop)

        # input is a Loop object
        self.assertEqual(id(runner.build_val_loop(loop)), id(loop))

        # test custom validation loop
        cfg = dict(type='CustomValLoop', interval=1)
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
        cfg.pop('optimizer')
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

        @HOOKS.register_module()
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
        cfg.val_cfg = dict(begin=2)
        runner = Runner.from_cfg(cfg)

        runner.train()

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

        @HOOKS.register_module()
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
        cfg.val_cfg = dict(interval=4, begin=4)
        runner = Runner.from_cfg(cfg)
        runner.train()

        assert isinstance(runner.train_loop, IterBasedTrainLoop)

        self.assertEqual(len(epoch_results), 1)
        self.assertEqual(epoch_results[0], 0)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(batch_idx_results, batch_idx_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_iter_results, val_iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(val_batch_idx_results,
                                   val_batch_idx_targets):
            self.assertEqual(result, target)

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
        cfg.pop('optimizer')
        cfg.pop('param_scheduler')
        cfg.pop('test_dataloader')
        cfg.pop('test_cfg')
        cfg.pop('test_evaluator')
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

        # test run test without train and test components
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_individually_test'
        cfg.pop('train_dataloader')
        cfg.pop('train_cfg')
        cfg.pop('optimizer')
        cfg.pop('param_scheduler')
        cfg.pop('val_dataloader')
        cfg.pop('val_cfg')
        cfg.pop('val_evaluator')
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
        self.assertEqual(len(runner._hooks), 7)
        # the third registered hook should be `DistSamplerSeedHook`
        self.assertTrue(isinstance(runner._hooks[3], DistSamplerSeedHook))
        # the fifth registered hook should be `ParamSchedulerHook`
        self.assertTrue(isinstance(runner._hooks[5], ParamSchedulerHook))

        runner._hooks = []
        # remove `ParamSchedulerHook` from default hooks
        runner.register_default_hooks(hooks=dict(timer=None))
        self.assertEqual(len(runner._hooks), 6)
        # `ParamSchedulerHook` was popped so the fifth is `CheckpointHook`
        self.assertTrue(isinstance(runner._hooks[5], CheckpointHook))

        # add a new default hook
        runner._hooks = []
        runner.register_default_hooks(hooks=dict(ToyHook=dict(type='ToyHook')))
        self.assertEqual(len(runner._hooks), 8)
        self.assertTrue(isinstance(runner._hooks[7], ToyHook))

    def test_custom_hooks(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_custom_hooks'
        runner = Runner.from_cfg(cfg)

        self.assertEqual(len(runner._hooks), 7)
        custom_hooks = [dict(type='ToyHook')]
        runner.register_custom_hooks(custom_hooks)
        self.assertEqual(len(runner._hooks), 8)
        self.assertTrue(isinstance(runner._hooks[7], ToyHook))

    def test_register_hooks(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_register_hooks'
        runner = Runner.from_cfg(cfg)

        runner._hooks = []
        custom_hooks = [dict(type='ToyHook')]
        runner.register_hooks(custom_hooks=custom_hooks)
        # six default hooks + custom hook (ToyHook)
        self.assertEqual(len(runner._hooks), 8)
        self.assertTrue(isinstance(runner._hooks[7], ToyHook))

    def test_custom_loop(self):
        # test custom loop with additional hook
        @LOOPS.register_module()
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
                    data_batch = next(self.dataloader)
                    self.run_iter(data_batch)
                self.runner.call_hook('after_train_epoch')

                self.runner.call_hook('after_train')

            def warmup_iter(self, data_batch):
                self.runner.call_hook(
                    'before_warmup_iter', data_batch=data_batch)
                self.runner.outputs = self.runner.model(
                    data_batch, return_loss=True)
                self.runner.call_hook(
                    'after_warmup_iter',
                    data_batch=data_batch,
                    outputs=self.runner.outputs)

        before_warmup_iter_results = []
        after_warmup_iter_results = []

        @HOOKS.register_module()
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
        self.assertTrue(osp.exists(osp.join(self.temp_dir, 'latest.pth')))
        self.assertFalse(osp.exists(osp.join(self.temp_dir, 'epoch_4.pth')))

        ckpt = torch.load(path)
        self.assertEqual(ckpt['meta']['epoch'], 3)
        self.assertEqual(ckpt['meta']['iter'], 12)
        # self.assertEqual(ckpt['meta']['hook_msgs']['last_ckpt'], path)
        assert isinstance(ckpt['optimizer'], dict)
        assert isinstance(ckpt['param_schedulers'], list)

        # 1.2 test `load_checkpoint`
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint2'
        cfg.optimizer = dict(type='SGD', lr=0.2)
        cfg.param_scheduler = dict(type='MultiStepLR', milestones=[1, 2, 3])
        runner = Runner.from_cfg(cfg)
        runner.load_checkpoint(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 0)
        self.assertTrue(runner._has_loaded)
        # load checkpoint will not initialize optimizer and param_schedulers
        # objects
        self.assertIsInstance(runner._optimizer, dict)
        self.assertIsInstance(runner.param_schedulers, list)
        self.assertIsInstance(runner.param_schedulers[0], dict)

        # 1.3 test `resume`
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint3'
        cfg.optimizer = dict(type='SGD', lr=0.2)
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

        # 1.4 test auto resume
        cfg = copy.deepcopy(self.iter_based_cfg)
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
        cfg = copy.deepcopy(self.iter_based_cfg)
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
        cfg.optimizer = dict(
            linear1=dict(type='SGD', lr=0.01),
            linear2=dict(type='Adam', lr=0.02),
            constructor='ToyMultipleOptimizerConstructor',
        )
        # disable OptimizerHook because it only works with one optimizer
        cfg.default_hooks = dict(optimizer=None)
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
        cfg.optimizer = dict(
            linear1=dict(type='SGD', lr=0.2),
            linear2=dict(type='Adam', lr=0.03),
            constructor='ToyMultipleOptimizerConstructor',
        )
        cfg.param_scheduler = dict(type='MultiStepLR', milestones=[1, 2, 3])
        cfg.default_hooks = dict(optimizer=None)
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

        # 2.1 test `save_checkpoint` which is called by `CheckpointHook`
        path = osp.join(self.temp_dir, 'iter_12.pth')
        self.assertTrue(osp.exists(path))
        self.assertTrue(osp.exists(osp.join(self.temp_dir, 'latest.pth')))
        self.assertFalse(osp.exists(osp.join(self.temp_dir, 'epoch_13.pth')))

        ckpt = torch.load(path)
        self.assertEqual(ckpt['meta']['epoch'], 0)
        self.assertEqual(ckpt['meta']['iter'], 12)
        # self.assertEqual(ckpt['meta']['hook_msgs']['last_ckpt'], path)
        assert isinstance(ckpt['optimizer'], dict)
        assert isinstance(ckpt['param_schedulers'], list)

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
