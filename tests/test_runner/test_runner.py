# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import multiprocessing as mp
import os
import os.path as osp
import platform
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from mmengine.config import Config
from mmengine.evaluator import BaseEvaluator
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger
from mmengine.model.wrappers import MMDataParallel, MMDistributedDataParallel
from mmengine.optim.scheduler import MultiStepLR
from mmengine.registry import (DATASETS, EVALUATORS, HOOKS, LOOPS,
                               MODEL_WRAPPERS, MODELS, PARAM_SCHEDULERS,
                               Registry)
from mmengine.runner import Runner
from mmengine.runner.loop import EpochBasedTrainLoop, IterBasedTrainLoop


@MODELS.register_module()
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))

    def train_step(self, *inputs, **kwargs):
        pass

    def val_step(self, *inputs, **kwargs):
        pass


@DATASETS.register_module()
class ToyDataset(Dataset):
    META = dict()  # type: ignore
    data = np.zeros((10, 1, 1, 1))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index])


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


class TestRunner(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.gettempdir()
        full_cfg = dict(
            model=dict(type='ToyModel'),
            train_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=1,
                num_workers=0),
            test_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=1,
                num_workers=0),
            optimizer=dict(type='SGD', lr=0.01),
            param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
            evaluator=dict(type='ToyEvaluator'),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            validation_cfg=dict(interval=1),
            test_cfg=dict(),
            custom_hooks=[],
            default_hooks=dict(
                timer=dict(type='IterTimerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1),
                logger=dict(type='TextLoggerHook'),
                optimizer=dict(type='OptimizerHook', grad_clip=False),
                param_scheduler=dict(type='ParamSchedulerHook')),
            env_cfg=dict(dist_params=dict(backend='nccl'), ),
            log_cfg=dict(log_level='INFO'),
            work_dir=self.temp_dir)
        self.full_cfg = Config(full_cfg)

    def tearDown(self):
        os.removedirs(self.temp_dir)

    def test_build_from_cfg(self):
        runner = Runner.build_from_cfg(cfg=self.full_cfg)
        # test env params
        assert runner.distributed is False
        assert runner.seed is not None
        assert runner.work_dir == self.temp_dir

        # model should be full initialized
        assert isinstance(runner.model, (nn.Module, MMDataParallel))
        # lazy init
        assert isinstance(runner.optimzier, dict)
        assert isinstance(runner.scheduler, list)
        assert isinstance(runner.train_dataloader, dict)
        assert isinstance(runner.val_dataloader, dict)
        assert isinstance(runner.test_dataloader, dict)
        assert isinstance(runner.evaluator, dict)

        # after runner.train(), train and val loader should be initialized
        # test loader should still be config
        runner.train()
        assert isinstance(runner.test_dataloader, dict)
        assert isinstance(runner.train_dataloader, DataLoader)
        assert isinstance(runner.val_dataloader, DataLoader)
        assert isinstance(runner.optimzier, SGD)
        assert isinstance(runner.evaluator, ToyEvaluator)

        runner.test()
        assert isinstance(runner.test_dataloader, DataLoader)

        # cannot run runner.test() without evaluator cfg
        with self.assertRaisesRegex(AssertionError,
                                    'evaluator does not exist'):
            cfg = copy.deepcopy(self.full_cfg)
            cfg.pop('evaluator')
            runner = Runner.build_from_cfg(cfg)
            runner.test()

        # cannot run runner.train() without optimizer cfg
        with self.assertRaisesRegex(AssertionError,
                                    'optimizer does not exist'):
            cfg = copy.deepcopy(self.full_cfg)
            cfg.pop('optimizer')
            runner = Runner.build_from_cfg(cfg)
            runner.train()

        # can run runner.train() without validation
        cfg = copy.deepcopy(self.full_cfg)
        cfg.validation_cfg = None
        cfg.pop('evaluator')
        cfg.pop('val_dataloader')
        runner = Runner.build_from_cfg(cfg)
        runner.train()

    def test_manually_init(self):
        model = ToyModel()
        optimizer = SGD(
            model.parameters(),
            lr=0.01,
        )

        class ToyHook(Hook):

            def before_train_epoch(self, runner):
                pass

        class ToyHook2(Hook):

            def after_train_epoch(self, runner):
                pass

        toy_hook = ToyHook()
        toy_hook2 = ToyHook2()
        runner = Runner(
            model=model,
            train_dataloader=DataLoader(dataset=ToyDataset()),
            val_dataloader=DataLoader(dataset=ToyDataset()),
            optimzier=optimizer,
            param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
            evaluator=ToyEvaluator(),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            validation_cfg=dict(interval=1),
            default_hooks=dict(param_scheduler=toy_hook),
            custom_hooks=[toy_hook2])
        runner.train()
        hook_names = [hook.__class__.__name__ for hook in runner.hooks]
        # test custom hook registered in runner
        assert 'ToyHook2' in hook_names
        # test default hook is replaced
        assert 'ToyHook' in hook_names
        # test other default hooks
        assert 'IterTimerHook' in hook_names

        # cannot run runner.test() when test_dataloader is None
        with self.assertRaisesRegex(AssertionError,
                                    'test dataloader does not exist'):
            runner.test()

        # cannot run runner.train() when optimizer is None
        with self.assertRaisesRegex(AssertionError,
                                    'optimizer does not exist'):
            runner = Runner(
                model=model,
                train_dataloader=DataLoader(dataset=ToyDataset()),
                val_dataloader=DataLoader(dataset=ToyDataset()),
                param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
                evaluator=ToyEvaluator(),
                train_cfg=dict(by_epoch=True, max_epochs=3),
                validation_cfg=dict(interval=1))
            runner.train()

        # cannot run runner.train() when validation_cfg is set
        # but val loader is None
        with self.assertRaisesRegex(AssertionError,
                                    'optimizer does not exist'):
            runner = Runner(
                model=model,
                train_dataloader=DataLoader(dataset=ToyDataset()),
                optimzier=optimizer,
                param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
                train_cfg=dict(by_epoch=True, max_epochs=3),
                validation_cfg=dict(interval=1))
            runner.train()

        # run runner.train() without validation
        runner = Runner(
            model=model,
            train_dataloader=DataLoader(dataset=ToyDataset()),
            optimzier=optimizer,
            param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            validation_cfg=None)
        runner.train()

    def test_setup_env(self):
        # temporarily store system setting
        sys_start_mehod = mp.get_start_method(allow_none=True)
        # pop and temp save system env vars
        sys_omp_threads = os.environ.pop('OMP_NUM_THREADS', default=None)
        sys_mkl_threads = os.environ.pop('MKL_NUM_THREADS', default=None)

        # test default multi-processing setting when workers > 1
        cfg = copy.deepcopy(self.full_cfg)
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
        cfg = copy.deepcopy(self.full_cfg)
        cfg.train_dataloader.num_workers = 0
        cfg.test_dataloader.num_workers = 0
        cfg.val_dataloader.num_workers = 0
        Runner.build_from_cfg(cfg)
        assert 'OMP_NUM_THREADS' not in os.environ
        assert 'MKL_NUM_THREADS' not in os.environ

        # test manually set env var
        os.environ['OMP_NUM_THREADS'] = '3'
        cfg = copy.deepcopy(self.full_cfg)
        cfg.train_dataloader.num_workers = 2
        cfg.test_dataloader.num_workers = 2
        cfg.val_dataloader.num_workers = 2
        Runner.build_from_cfg(cfg)
        assert os.getenv('OMP_NUM_THREADS') == '3'

        # test manually set mp start method
        cfg = copy.deepcopy(self.full_cfg)
        cfg.env_cfg.mp_cfg = dict(mp_start_method='spawn')
        Runner.build_from_cfg(cfg)
        assert mp.get_start_method() == 'spawn'

        # revert setting to avoid affecting other programs
        if sys_start_mehod:
            mp.set_start_method(sys_start_mehod, force=True)
        if sys_omp_threads:
            os.environ['OMP_NUM_THREADS'] = sys_omp_threads
        else:
            os.environ.pop('OMP_NUM_THREADS')
        if sys_mkl_threads:
            os.environ['MKL_NUM_THREADS'] = sys_mkl_threads
        else:
            os.environ.pop('MKL_NUM_THREADS')

    def test_logger(self):
        runner = Runner.build_from_cfg(self.full_cfg)
        assert isinstance(runner.logger, MMLogger)
        # test latest logger and runner logger are the same
        assert runner.logger.level == logging.INFO
        assert MMLogger.get_instance(
        ).instance_name == runner.logger.instance_name
        # test latest message hub and runner message hub are the same
        assert isinstance(runner.message_hub, MessageHub)
        assert MessageHub.get_instance(
        ).instance_name == runner.message_hub.instance_name

        # test set log level in cfg
        self.full_cfg.log_cfg.log_level = 'DEBUG'
        runner = Runner.build_from_cfg(self.full_cfg)
        assert runner.logger.level == logging.DEBUG

    @patch('torch.distributed.get_rank', lambda: 0)
    @patch('torch.distributed.is_initialized', lambda: True)
    @patch('torch.distributed.is_available', lambda: True)
    def test_model_wrapper(self):
        # non-distributed model build from config
        runner = Runner.build_from_cfg(self.full_cfg)
        assert isinstance(runner.model, MMDataParallel)

        # non-distributed model build manually
        model = ToyModel()
        runner = Runner(
            model=model, train_cfg=dict(by_epoch=True, max_epochs=3))
        assert isinstance(runner.model, MMDataParallel)

        # distributed model build from config
        cfg = copy.deepcopy(self.full_cfg)
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

        cfg = copy.deepcopy(self.full_cfg)
        cfg.model_wrapper = dict(type='CustomModelWrapper')
        runner = Runner.build_from_cfg(cfg)
        assert isinstance(runner.model, CustomModelWrapper)

    def test_default_scope(self):
        TOY_SCHEDULERS = Registry(
            'parameter scheduler', parent=PARAM_SCHEDULERS, scope='toy')

        @TOY_SCHEDULERS.register_module()
        class ToyScheduler(MultiStepLR):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        self.full_cfg.param_scheduler = dict(
            type='ToyScheduler', milestones=[1, 2])
        self.full_cfg.default_scope = 'toy'

        runner = Runner.build_from_cfg(self.full_cfg)
        runner.train()
        assert isinstance(runner.scheduler[0], ToyScheduler)

    def test_checkpoint(self):
        runner = Runner.build_from_cfg(self.full_cfg)
        runner.run()
        path = osp.join(self.temp_dir, 'epoch_3.pth')
        runner.save_checkpoint(path)
        assert osp.exists(path)
        ckpt = torch.load(path)
        # scheduler should saved in the checkpoint
        assert isinstance(ckpt['scheduler'], list)

        # load by a new runner but not resume
        runner2 = Runner.build_from_cfg(self.full_cfg)
        runner2.load_checkpoint(path, resume=False)
        self.assertNotEqual(runner2.epoch, runner.epoch)
        self.assertNotEqual(runner2.iter, runner.iter)

        # load by a new runner and resume
        runner3 = Runner.build_from_cfg(self.full_cfg)
        runner3.load_checkpoint(path, resume=True)
        self.assertEqual(runner3.epoch, runner.epoch)
        self.assertEqual(runner3.iter, runner.iter)

    def test_custom_hooks(self):
        results = []
        targets = [0, 1, 2]

        @HOOKS.register_module()
        class ToyHook(Hook):

            def before_train_epoch(self, runner):
                results.append(runner.epoch)

        self.full_cfg.custom_hooks = [dict(type='ToyHook', priority=50)]
        runner = Runner.build_from_cfg(self.full_cfg)

        # test hook registered in runner
        hook_names = [hook.__class__.__name__ for hook in runner.hooks]
        assert 'ToyHook' in hook_names

        # test hook behavior
        runner.train()
        for result, target, in zip(results, targets):
            self.assertEqual(result, target)

    def test_iter_based(self):
        self.full_cfg.train_cfg = dict(by_epoch=False, max_iters=30)

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

        self.full_cfg.custom_hooks = [dict(type='TestIterHook', priority=50)]
        runner = Runner.build_from_cfg(self.full_cfg)

        assert isinstance(runner._train_loop, IterBasedTrainLoop)

        runner.train()

        self.assertEqual(len(epoch_results), 1)
        self.assertEqual(epoch_results[0], 0)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(inner_iter_results, iter_targets):
            self.assertEqual(result, target)

    def test_epoch_based(self):
        self.full_cfg.train_cfg = dict(by_epoch=True, max_epochs=3)

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

        self.full_cfg.custom_hooks = [dict(type='TestEpochHook', priority=50)]
        runner = Runner.build_from_cfg(self.full_cfg)

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

        self.full_cfg.train_cfg = dict(
            type='CustomTrainLoop',
            max_epochs=3,
            warmup_loader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=0),
            max_warmup_iters=5)
        self.full_cfg.custom_hooks = [dict(type='TestWarmupHook', priority=50)]
        runner = Runner.build_from_cfg(self.full_cfg)

        assert isinstance(runner._train_loop, CustomTrainLoop)

        runner.train()

        # test custom hook triggered normally
        self.assertEqual(len(before_warmup_iter_results), 5)
        self.assertEqual(len(after_warmup_iter_results), 5)
        for before, after in zip(before_warmup_iter_results,
                                 after_warmup_iter_results):
            self.assertEqual(before, 'before')
            self.assertEqual(after, 'after')
