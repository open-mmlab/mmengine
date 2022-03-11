# Copyright (c) OpenMMLab. All rights reserved.
import copy
import multiprocessing as mp
import os
import os.path as osp
import platform
import shutil
import tempfile
import time
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from mmengine.config import Config
from mmengine.data import DefaultSampler
from mmengine.evaluator import BaseEvaluator, build_evaluator
from mmengine.hooks import (Hook, IterTimerHook, LoggerHook, OptimizerHook,
                            ParamSchedulerHook)
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.logging import MessageHub, MMLogger
from mmengine.model.wrappers import MMDataParallel, MMDistributedDataParallel
from mmengine.optim.scheduler import MultiStepLR, StepLR
from mmengine.registry import (DATASETS, EVALUATORS, HOOKS, LOOPS,
                               MODEL_WRAPPERS, MODELS, PARAM_SCHEDULERS,
                               Registry)
from mmengine.runner import (BaseLoop, EpochBasedTrainLoop, IterBasedTrainLoop,
                             Runner, TestLoop, ValLoop)
from mmengine.runner.priority import Priority, get_priority
from mmengine.utils import is_list_of
from mmengine.visualization.writer import ComposedWriter


@MODELS.register_module()
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, data_batch, return_loss=False):
        input, label = zip(*data_batch)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        input = torch.stack(input).to(device)
        label = torch.stack(label).to(device)
        output = self.linear(input)
        if return_loss:
            loss = (label - output).sum()
            outputs = dict(loss=loss, log_vars=dict(loss=loss.item()))
            return outputs
        else:
            outputs = dict(log_vars=dict(a=1, b=0.5))
            return outputs


@MODELS.register_module()
class ToyModel1(nn.Module):

    def __init__(self):
        super().__init__()


@DATASETS.register_module()
class ToyDataset(Dataset):
    META = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


@EVALUATORS.register_module()
class ToyEvaluator(BaseEvaluator):

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
            val_evaluator=dict(type='ToyEvaluator'),
            test_evaluator=dict(type='ToyEvaluator'),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            val_cfg=dict(interval=1),
            test_cfg=dict(),
            custom_hooks=[],
            default_hooks=dict(
                timer=dict(type='IterTimerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1),
                logger=dict(type='LoggerHook'),
                optimizer=dict(type='OptimizerHook', grad_clip=None),
                param_scheduler=dict(type='ParamSchedulerHook')),
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
            timer=dict(type='IterTimerHook'),
            checkpoint=dict(type='CheckpointHook', interval=2, by_epoch=False),
            logger=dict(type='LoggerHook', by_epoch=False),
            optimizer=dict(type='OptimizerHook', grad_clip=None),
            param_scheduler=dict(type='ParamSchedulerHook'))

        time.sleep(1)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # 1. test arguments
        # 1.1 train_dataloader, train_cfg, optimizer and param_scheduler
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.pop('train_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        # all of training related configs are None
        cfg.pop('train_dataloader')
        cfg.pop('optimizer')
        cfg.pop('param_scheduler')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # avoid different runners having same timestamp
        time.sleep(1)

        # all of training related configs are not None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # 1.2 val_dataloader, val_evaluator, val_cfg
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.pop('val_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        time.sleep(1)

        cfg.pop('val_dataloader')
        cfg.pop('val_evaluator')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        time.sleep(1)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # 1.3 test_dataloader, test_evaluator and test_cfg
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.pop('test_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            runner = Runner(**cfg)

        time.sleep(1)

        cfg.pop('test_dataloader')
        cfg.pop('test_evaluator')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        time.sleep(1)

        # 1.4 test env params
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = Runner(**cfg)
        self.assertFalse(runner.distributed)
        self.assertFalse(runner.deterministic)

        time.sleep(1)

        # 1.5 message_hub, logger and writer
        # they are all not specified
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = Runner(**cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertIsInstance(runner.writer, ComposedWriter)

        time.sleep(1)

        # they are all specified
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.logger = dict(name='test_logger')
        cfg.message_hub = dict(name='test_message_hub')
        cfg.writer = dict(name='test_writer')
        runner = Runner(**cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertEqual(runner.logger.instance_name, 'test_logger')
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertEqual(runner.message_hub.instance_name, 'test_message_hub')
        self.assertIsInstance(runner.writer, ComposedWriter)
        self.assertEqual(runner.writer.instance_name, 'test_writer')

        assert runner.distributed is False
        assert runner.seed is not None
        assert runner.work_dir == self.temp_dir

        # 2 model should be initialized
        self.assertIsInstance(runner.model,
                              (nn.Module, DistributedDataParallel))
        self.assertEqual(runner.model_name, 'ToyModel')

        # 3. test lazy initialization
        self.assertIsInstance(runner.train_dataloader, dict)
        self.assertIsInstance(runner.val_dataloader, dict)
        self.assertIsInstance(runner.test_dataloader, dict)
        self.assertIsInstance(runner.optimizer, dict)
        self.assertIsInstance(runner.param_schedulers[0], dict)

        # After calling runner.train(),
        # train_dataloader and val_loader should be initialized but
        # test_dataloader should also be dict
        runner.train()

        self.assertIsInstance(runner.train_loop, BaseLoop)
        self.assertIsInstance(runner.train_loop.dataloader, DataLoader)
        self.assertIsInstance(runner.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)
        self.assertIsInstance(runner.val_loop, BaseLoop)
        self.assertIsInstance(runner.val_loop.dataloader, DataLoader)
        self.assertIsInstance(runner.val_loop.evaluator, ToyEvaluator)

        # After calling runner.test(), test_dataloader should be initialized
        self.assertIsInstance(runner.test_loop, dict)
        runner.test()
        self.assertIsInstance(runner.test_loop, BaseLoop)
        self.assertIsInstance(runner.test_loop.dataloader, DataLoader)
        self.assertIsInstance(runner.test_loop.evaluator, ToyEvaluator)

        time.sleep(1)
        # 4. initialize runner with objects rather than config
        model = ToyModel()
        optimizer = SGD(
            model.parameters(),
            lr=0.01,
        )
        toy_hook = ToyHook()
        toy_hook2 = ToyHook2()
        runner = Runner(
            model=model,
            work_dir=self.temp_dir,
            train_cfg=dict(by_epoch=True, max_epochs=3),
            train_dataloader=DataLoader(dataset=ToyDataset()),
            optimizer=optimizer,
            param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
            val_cfg=dict(interval=1),
            val_dataloader=DataLoader(dataset=ToyDataset()),
            val_evaluator=ToyEvaluator(),
            test_cfg=dict(),
            test_dataloader=DataLoader(dataset=ToyDataset()),
            test_evaluator=ToyEvaluator(),
            default_hooks=dict(param_scheduler=toy_hook),
            custom_hooks=[toy_hook2])
        runner.train()
        runner.test()

    def test_build_from_cfg(self):
        runner = Runner.build_from_cfg(cfg=self.epoch_based_cfg)
        self.assertIsInstance(runner, Runner)

    def test_setup_env(self):
        # temporarily store system setting
        sys_start_method = mp.get_start_method(allow_none=True)
        # pop and temp save system env vars
        sys_omp_threads = os.environ.pop('OMP_NUM_THREADS', None)
        sys_mkl_threads = os.environ.pop('MKL_NUM_THREADS', None)

        # test default multi-processing setting when workers > 1
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.train_dataloader.num_workers = 4
        cfg.test_dataloader.num_workers = 4
        cfg.val_dataloader.num_workers = 4
        Runner.build_from_cfg(cfg)
        assert os.getenv('OMP_NUM_THREADS') == '1'
        assert os.getenv('MKL_NUM_THREADS') == '1'
        if platform.system() != 'Windows':
            assert mp.get_start_method() == 'fork'

        # test default multi-processing setting when workers <= 1
        os.environ.pop('OMP_NUM_THREADS')
        os.environ.pop('MKL_NUM_THREADS')
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.train_dataloader.num_workers = 0
        cfg.test_dataloader.num_workers = 0
        cfg.val_dataloader.num_workers = 0
        Runner.build_from_cfg(cfg)
        assert 'OMP_NUM_THREADS' not in os.environ
        assert 'MKL_NUM_THREADS' not in os.environ

        # test manually set env var
        os.environ['OMP_NUM_THREADS'] = '3'
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.train_dataloader.num_workers = 2
        cfg.test_dataloader.num_workers = 2
        cfg.val_dataloader.num_workers = 2
        Runner.build_from_cfg(cfg)
        assert os.getenv('OMP_NUM_THREADS') == '3'

        # test manually set mp start method
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.env_cfg.mp_cfg = dict(mp_start_method='spawn')
        Runner.build_from_cfg(cfg)
        assert mp.get_start_method() == 'spawn'

        # revert setting to avoid affecting other programs
        if sys_start_method:
            mp.set_start_method(sys_start_method, force=True)
        if sys_omp_threads:
            os.environ['OMP_NUM_THREADS'] = sys_omp_threads
        else:
            os.environ.pop('OMP_NUM_THREADS')
        if sys_mkl_threads:
            os.environ['MKL_NUM_THREADS'] = sys_mkl_threads
        else:
            os.environ.pop('MKL_NUM_THREADS')

    def test_logger(self):
        # TODO
        # runner = Runner.build_from_cfg(self.epoch_based_cfg)
        # assert isinstance(runner.logger, MMLogger)
        # # test latest logger and runner logger are the same
        # assert runner.logger.level == logging.INFO
        # assert MMLogger.get_instance(
        # ).instance_name == runner.logger.instance_name
        # # test latest message hub and runner message hub are the same
        # assert isinstance(runner.message_hub, MessageHub)
        # assert MessageHub.get_instance(
        # ).instance_name == runner.message_hub.instance_name

        # # test set log level in cfg
        # self.epoch_based_cfg.log_cfg.log_level = 'DEBUG'
        # runner = Runner.build_from_cfg(self.epoch_based_cfg)
        # assert runner.logger.level == logging.DEBUG
        pass

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

        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.train()
        assert isinstance(runner.scheduler[0], ToyScheduler)

    def test_build_model(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
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

    @patch('torch.distributed.get_rank', lambda: 0)
    @patch('torch.distributed.is_initialized', lambda: True)
    @patch('torch.distributed.is_available', lambda: True)
    def test_model_wrapper(self):
        # non-distributed model build from config
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        assert isinstance(runner.model, MMDataParallel)

        # non-distributed model build manually
        model = ToyModel()
        runner = Runner(
            model=model, train_cfg=dict(by_epoch=True, max_epochs=3))
        assert isinstance(runner.model, MMDataParallel)

        # distributed model build from config
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.launcher = 'pytorch'
        runner = Runner.build_from_cfg(cfg)
        assert isinstance(runner.model, MMDistributedDataParallel)

        # distributed model build manually
        model = ToyModel()
        runner = Runner(
            model=model,
            train_cfg=dict(by_epoch=True, max_epochs=3),
            env_cfg=dict(dist_params=dict(backend='nccl')),
            launcher='pytorch')
        assert isinstance(runner.model, MMDistributedDataParallel)

        # custom model wrapper
        @MODEL_WRAPPERS.register_module()
        class CustomModelWrapper:

            def train_step(self, *inputs, **kwargs):
                pass

            def val_step(self, *inputs, **kwargs):
                pass

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model_wrapper = dict(type='CustomModelWrapper')
        runner = Runner.build_from_cfg(cfg)
        self.assertIsInstance(runner.model, CustomModelWrapper)

    def test_build_optimizer(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        # input should be an Optimizer object or dict
        with self.assertRaisesRegex(TypeError, 'optimizer should be'):
            runner.build_optimizer('invalid-type')

        # input is an Optimizer object
        _optimizer = SGD(runner.model.parameters(), lr=0.01)
        optimizer = runner.build_optimizer(_optimizer)
        self.assertEqual(id(_optimizer), id(optimizer))

        # input is a dict
        optimizer = runner.build_optimizer(dict(type='SGD', lr=0.01))
        self.assertIsInstance(optimizer, SGD)

    def test_build_param_scheduler(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        # `build_optimizer` should be called before `build_param_scheduler`
        cfg = dict(type='MultiStepLR', milestones=[1, 2])
        runner.optimizer = None
        with self.assertRaisesRegex(RuntimeError, 'should be called before'):
            runner.build_param_scheduler(cfg)

        runner.optimizer = runner.build_optimizer(dict(type='SGD', lr=0.01))
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertIsInstance(param_schedulers, list)
        self.assertEqual(len(param_schedulers), 1)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)

        # input is a ParamScheduler object
        param_scheduler = MultiStepLR(runner.optimizer, milestones=[1, 2])
        param_schedulers = runner.build_param_scheduler(param_scheduler)
        self.assertEqual(id(param_schedulers[0]), id(param_scheduler))

        # input is a list of dict
        cfg = [
            dict(type='MultiStepLR', milestones=[1, 2]),
            dict(type='StepLR', step_size=1)
        ]
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertEqual(len(param_schedulers), 2)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)
        self.assertIsInstance(param_schedulers[1], StepLR)

        # input is a list and some items are ParamScheduler objects
        cfg = [param_scheduler, dict(type='StepLR', step_size=1)]
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertEqual(len(param_schedulers), 2)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)
        self.assertIsInstance(param_schedulers[1], StepLR)

    def test_build_dataloader(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        cfg = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=1,
            num_workers=0)
        dataloader = runner.build_dataloader(cfg)
        self.assertIsInstance(dataloader, DataLoader)
        self.assertIsInstance(dataloader.dataset, ToyDataset)
        self.assertIsInstance(dataloader.sampler, DefaultSampler)

    def test_build_train_loop(self):
        # input should be a Loop object or dict
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
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

        # test custom training loop
        @LOOPS.register_module()
        class CustomTrainLoop(BaseLoop):

            def __init__(self, runner, dataloader, max_epochs):
                super().__init__(runner, dataloader)
                self._max_epochs = max_epochs

            def run(self) -> None:
                pass

        cfg = dict(type='CustomTrainLoop', max_epochs=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, CustomTrainLoop)

    def test_build_val_loop(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        # input should be a Loop object or dict
        with self.assertRaisesRegex(TypeError, 'should be'):
            runner.build_test_loop('invalid-type')

        # input is a dict and contains type key
        cfg = dict(type='ValLoop', interval=1)
        loop = runner.build_test_loop(cfg)
        self.assertIsInstance(loop, EpochBasedTrainLoop)

        # input is a dict but does not contain type key
        cfg = dict(interval=1)
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, ValLoop)

        # input is a Loop object
        self.assertEqual(id(runner.build_val_loop(loop)), id(loop))

        @LOOPS.register_module()
        class CustomValLoop(BaseLoop):

            def __init__(self, runner, dataloader, evaluator, interval=1):
                super().__init__(runner, dataloader)
                self._runner = runner

                if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
                    self.evaluator = build_evaluator(evaluator)  # type: ignore
                else:
                    self.evaluator = evaluator

            def run(self) -> None:
                pass

        # test custom validation loop
        cfg = dict(type='CustomValLoop', interval=1)
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, CustomValLoop)

    def test_build_test_loop(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

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

        @LOOPS.register_module()
        class CustomTestLoop(BaseLoop):

            def __init__(self, runner, dataloader, evaluator):
                super().__init__(runner, dataloader)
                self._runner = runner

                if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
                    self.evaluator = build_evaluator(evaluator)  # type: ignore
                else:
                    self.evaluator = evaluator

            def run(self) -> None:
                pass

        # test custom validation loop
        cfg = dict(type='CustomTestLoop')
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, CustomTestLoop)

    def test_train(self):
        pass

    def test_val(self):
        pass

    def test_test(self):
        pass

    def test_register_hook(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
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
        optimizer_hook = OptimizerHook()
        runner.register_hook(optimizer_hook)
        self.assertEqual(len(runner._hooks), 2)
        # The priority of `OptimizerHook` is `HIGH` which is greater than
        # `IterTimerHook`, so the first item of `_hooks` should be
        # `OptimizerHook`
        self.assertTrue(isinstance(runner._hooks[0], OptimizerHook))
        self.assertEqual(
            get_priority(runner._hooks[0].priority), get_priority('HIGH'))

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
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner._hooks = []

        # register five hooks by default
        runner.register_default_hooks()
        self.assertEqual(len(runner._hooks), 5)
        # the forth registered hook should be `ParamSchedulerHook`
        self.assertTrue(isinstance(runner._hooks[3], ParamSchedulerHook))

        runner._hooks = []
        # remove `ParamSchedulerHook` from default hooks
        runner.register_default_hooks(hooks=dict(timer=None))
        self.assertEqual(len(runner._hooks), 4)
        # `ParamSchedulerHook` was popped so the forth is `CheckpointHook`
        self.assertTrue(isinstance(runner._hooks[3], CheckpointHook))

        # add a new default hook
        runner._hooks = []
        runner.register_default_hooks(hooks=dict(ToyHook=dict(type='ToyHook')))
        self.assertEqual(len(runner._hooks), 6)
        self.assertTrue(isinstance(runner._hooks[5], ToyHook))

    def test_custom_hooks(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        self.assertEqual(len(runner._hooks), 5)
        custom_hooks = [dict(type='ToyHook')]
        runner.register_custom_hooks(custom_hooks)
        self.assertEqual(len(runner._hooks), 6)
        self.assertTrue(isinstance(runner._hooks[5], ToyHook))

    def test_register_hooks(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner._hooks = []
        custom_hooks = [dict(type='ToyHook')]
        runner.register_hooks(custom_hooks=custom_hooks)
        # five default hooks + custom hook (ToyHook)
        self.assertEqual(len(runner._hooks), 6)
        self.assertTrue(isinstance(runner._hooks[5], ToyHook))

    def test_iter_based(self):
        self.epoch_based_cfg.train_cfg = dict(by_epoch=False, max_iters=30)

        # test iter and epoch counter of IterBasedTrainLoop
        epoch_results = []
        iter_results = []
        inner_iter_results = []
        iter_targets = [i for i in range(30)]

        @HOOKS.register_module()
        class TestIterHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner):
                iter_results.append(runner.iter)
                inner_iter_results.append(runner.inner_iter)

        self.epoch_based_cfg.custom_hooks = [
            dict(type='TestIterHook', priority=50)
        ]
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        assert isinstance(runner._train_loop, IterBasedTrainLoop)

        runner.train()

        self.assertEqual(len(epoch_results), 1)
        self.assertEqual(epoch_results[0], 0)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(inner_iter_results, iter_targets):
            self.assertEqual(result, target)

    def test_epoch_based(self):
        self.epoch_based_cfg.train_cfg = dict(by_epoch=True, max_epochs=3)

        # test iter and epoch counter of EpochBasedTrainLoop
        epoch_results = []
        epoch_targets = [i for i in range(3)]
        iter_results = []
        iter_targets = [i for i in range(10 * 3)]
        inner_iter_results = []
        inner_iter_targets = [i for i in range(10)] * 3  # train and val

        @HOOKS.register_module()
        class TestEpochHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner, data_batch=None):
                iter_results.append(runner.iter)
                inner_iter_results.append(runner.inner_iter)

        self.epoch_based_cfg.custom_hooks = [
            dict(type='TestEpochHook', priority=50)
        ]
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        assert isinstance(runner._train_loop, EpochBasedTrainLoop)

        runner.train()

        for result, target, in zip(epoch_results, epoch_targets):
            self.assertEqual(result, target)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(inner_iter_results, inner_iter_targets):
            self.assertEqual(result, target)

    def test_custom_loop(self):
        # test custom loop with additional hook
        @LOOPS.register_module()
        class CustomTrainLoop(EpochBasedTrainLoop):
            """custom train loop with additional warmup stage."""

            def __init__(self, runner, loader, max_epochs, warmup_loader,
                         max_warmup_iters):
                super().__init__(
                    runner=runner, loader=loader, max_epochs=max_epochs)
                self.warmup_loader = self.runner.build_dataloader(
                    warmup_loader)
                self.max_warmup_iters = max_warmup_iters

            def run(self):
                self.runner.call_hooks('before_run')
                for idx, data_batch in enumerate(self.warmup_loader):
                    self.warmup_iter(data_batch)
                    if idx >= self.max_warmup_iters:
                        break

                self.runner.call_hooks('before_train_epoch')
                while self.runner.iter < self._max_iter:
                    data_batch = next(self.loader)
                    self.run_iter(data_batch)
                self.runner.call_hooks('after_train_epoch')
                self.runner.call_hooks('after_run')

            def warmup_iter(self, data_batch):
                self.runner.call_hooks(
                    'before_warmup_iter', args=dict(data_batch=data_batch))
                outputs = self.runner.model.train_step(data_batch)
                self.runner.call_hooks(
                    'after_warmup_iter',
                    args=dict(data_batch=data_batch, outputs=outputs))

        before_warmup_iter_results = []
        after_warmup_iter_results = []

        @HOOKS.register_module()
        class TestWarmupHook(Hook):
            """test custom train loop."""

            def before_warmup_iter(self, data_batch=None):
                before_warmup_iter_results.append('before')

            def after_warmup_iter(self, data_batch=None, outputs=None):
                after_warmup_iter_results.append('after')

        self.epoch_based_cfg.train_cfg = dict(
            type='CustomTrainLoop',
            max_epochs=3,
            warmup_loader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=0),
            max_warmup_iters=5)
        self.epoch_based_cfg.custom_hooks = [
            dict(type='TestWarmupHook', priority=50)
        ]
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        assert isinstance(runner._train_loop, CustomTrainLoop)

        runner.train()

        # test custom hook triggered normally
        self.assertEqual(len(before_warmup_iter_results), 5)
        self.assertEqual(len(after_warmup_iter_results), 5)
        for before, after in zip(before_warmup_iter_results,
                                 after_warmup_iter_results):
            self.assertEqual(before, 'before')
            self.assertEqual(after, 'after')

    def test_checkpoint(self):
        # 1. test epoch based
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.train()

        # 1.1 test `save_checkpoint` which called by `CheckpointHook`
        path = osp.join(self.temp_dir, 'epoch_3.pth')
        self.assertTrue(osp.exists(path))
        self.assertTrue(osp.exists(osp.join(self.temp_dir, 'latest.pth')))
        self.assertFalse(osp.exists(osp.join(self.temp_dir, 'epoch_4.pth')))

        ckpt = torch.load(path)
        self.assertEqual(ckpt['meta']['epoch'], 3)
        self.assertEqual(ckpt['meta']['iter'], 12)
        self.assertEqual(ckpt['meta']['inner_iter'], 3)
        # self.assertEqual(ckpt['meta']['hook_msgs']['last_ckpt'], path)
        assert isinstance(ckpt['optimizer'], dict)
        assert isinstance(ckpt['param_schedulers'], list)

        # 1.2 test `load_checkpoint`
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.load_checkpoint(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 0)
        self.assertEqual(runner.inner_iter, 0)
        self.assertTrue(runner._has_loaded)

        time.sleep(1)
        # 1.3 test `resume`
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.resume(path)
        self.assertEqual(runner.epoch, 3)
        self.assertEqual(runner.iter, 12)
        self.assertEqual(runner.inner_iter, 3)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)

        # 2. test iter based
        time.sleep(1)
        runner = Runner.build_from_cfg(self.iter_based_cfg)
        runner.train()

        # 2.1 test `save_checkpoint` which called by `CheckpointHook`
        path = osp.join(self.temp_dir, 'epoch_12.pth')
        self.assertTrue(osp.exists(path))
        self.assertTrue(osp.exists(osp.join(self.temp_dir, 'latest.pth')))
        self.assertFalse(osp.exists(osp.join(self.temp_dir, 'epoch_13.pth')))

        ckpt = torch.load(path)
        self.assertEqual(ckpt['meta']['epoch'], 0)
        self.assertEqual(ckpt['meta']['iter'], 12)
        self.assertEqual(ckpt['meta']['inner_iter'], 0)
        # self.assertEqual(ckpt['meta']['hook_msgs']['last_ckpt'], path)
        assert isinstance(ckpt['optimizer'], dict)
        assert isinstance(ckpt['param_schedulers'], list)

        # 2.2 test `load_checkpoint`
        runner = Runner.build_from_cfg(self.iter_based_cfg)
        runner.load_checkpoint(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 0)
        self.assertEqual(runner.inner_iter, 0)
        self.assertTrue(runner._has_loaded)

        time.sleep(1)
        # 2.3 test `resume`
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.resume(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 12)
        self.assertEqual(runner.inner_iter, 0)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)
