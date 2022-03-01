# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from mmengine.config import Config
from mmengine.evaluator import BaseEvaluator
from mmengine.hooks import Hook
from mmengine.model.wrappers import MMDataParallel
from mmengine.optim.scheduler import MultiStepLR
from mmengine.registry import DATASETS, EVALUATORS, HOOKS, LOOPS, MODELS
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
    data = list(range(10))

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]).reshape((1, 1, 1))


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
                optimizer=dict(type='OptimzierHook', grad_clip=False),
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

        # model should full init
        assert isinstance(runner.model, (nn.Module, MMDataParallel))
        # lazy init
        assert isinstance(runner.optimzier, dict)
        assert isinstance(runner.scheduler, list)
        assert isinstance(runner.train_dataloader, dict)
        assert isinstance(runner.val_dataloader, dict)
        assert isinstance(runner.test_dataloader, dict)
        assert isinstance(runner.evaluator, dict)

        # after run train, train and val loader should init
        # test loader should still be config
        runner.train()
        assert isinstance(runner.test_dataloader, dict)
        assert isinstance(runner.train_dataloader, DataLoader)
        assert isinstance(runner.val_dataloader, DataLoader)
        assert isinstance(runner.optimzier, SGD)
        assert isinstance(runner.evaluator, ToyEvaluator)

        runner.test()
        assert isinstance(runner.test_dataloader, DataLoader)

        # cannot run test without evaluator cfg
        with self.assertRaisesRegex(AssertionError,
                                    'evaluator does not exist'):
            cfg = copy.deepcopy(self.full_cfg)
            cfg.pop('evaluator')
            runner = Runner.build_from_cfg(cfg)
            runner.test()

        # cannot run train without optimzier cfg
        with self.assertRaisesRegex(AssertionError, 'optimzer does not exist'):
            cfg = copy.deepcopy(self.full_cfg)
            cfg.pop('optimzier')
            runner = Runner.build_from_cfg(cfg)
            runner.train()

        # can run train without validation
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
        runner = Runner(
            model=model,
            train_dataloader=DataLoader(dataset=ToyDataset()),
            val_dataloader=DataLoader(dataset=ToyDataset()),
            optimzier=optimizer,
            param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
            evaluator=ToyEvaluator(),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            validation_cfg=dict(interval=1))
        runner.train()

        # cannot run test when test_dataloader is None
        with self.assertRaisesRegex(AssertionError,
                                    'test dataloader does not exist'):
            runner.test()

        # cannot run train when optimizer is None
        with self.assertRaisesRegex(AssertionError,
                                    'optimzier does not exist'):
            runner = Runner(
                model=model,
                train_dataloader=DataLoader(dataset=ToyDataset()),
                val_dataloader=DataLoader(dataset=ToyDataset()),
                param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
                evaluator=ToyEvaluator(),
                train_cfg=dict(by_epoch=True, max_epochs=3),
                validation_cfg=dict(interval=1))
            runner.train()

        # cannot run train when validation_cfg is set but val loader is None
        with self.assertRaisesRegex(AssertionError,
                                    'optimzier does not exist'):
            runner = Runner(
                model=model,
                train_dataloader=DataLoader(dataset=ToyDataset()),
                optimzier=optimizer,
                param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
                train_cfg=dict(by_epoch=True, max_epochs=3),
                validation_cfg=dict(interval=1))
            runner.train()

        # run train without validation
        runner = Runner(
            model=model,
            train_dataloader=DataLoader(dataset=ToyDataset()),
            optimzier=optimizer,
            param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            validation_cfg=None)
        runner.train()

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
        self.assertNotEqual(runner2.global_iter, runner.global_iter)

        # load by a new runner and resume
        runner3 = Runner.build_from_cfg(self.full_cfg)
        runner3.load_checkpoint(path, resume=True)
        self.assertEqual(runner3.epoch, runner.epoch)
        self.assertEqual(runner3.global_iter, runner.global_iter)

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
        global_iter_results = []
        inner_iter_results = []
        iter_targets = [i for i in range(30)]

        @HOOKS.register_module()
        class TestIterHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner):
                global_iter_results.append(runner.global_iter)
                inner_iter_results.append(runner.inner_iter)

        self.full_cfg.custom_hooks = [dict(type='TestIterHook', priority=50)]
        runner = Runner.build_from_cfg(self.full_cfg)

        assert isinstance(runner._train_loop, IterBasedTrainLoop)

        runner.train()

        self.assertEqual(len(epoch_results), 1)
        self.assertEqual(epoch_results[0], 0)
        for result, target, in zip(global_iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(inner_iter_results, iter_targets):
            self.assertEqual(result, target)

    def test_epoch_based(self):
        self.full_cfg.train_cfg = dict(by_epoch=True, max_epochs=3)

        # test iter and epoch counter of EpochBasedTrainLoop
        epoch_results = []
        epoch_targets = [i for i in range(3)]
        global_iter_results = []
        global_iter_targets = [i for i in range(10 * 3)]
        inner_iter_results = []
        inner_iter_targets = [i for i in range(10)] * 3  # train and val

        @HOOKS.register_module()
        class TestEpochHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner, data_batch=None):
                global_iter_results.append(runner.global_iter)
                inner_iter_results.append(runner.inner_iter)

        self.full_cfg.custom_hooks = [dict(type='TestEpochHook', priority=50)]
        runner = Runner.build_from_cfg(self.full_cfg)

        assert isinstance(runner._train_loop, EpochBasedTrainLoop)

        runner.train()

        for result, target, in zip(epoch_results, epoch_targets):
            self.assertEqual(result, target)
        for result, target, in zip(global_iter_results, global_iter_targets):
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
                while self.runner.global_iter < self._max_iter:
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
