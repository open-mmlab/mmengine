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
from mmengine.model.wrappers import MMDataParallel
from mmengine.optim.scheduler import MultiStepLR
from mmengine.registry import DATASETS, EVALUATORS, MODELS
from mmengine.runner import Runner


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
    data = list(range(50))

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
        assert runner2.epoch != runner.epoch
        assert runner2.global_iter != runner.global_iter

        # load by a new runner and resume
        runner3 = Runner.build_from_cfg(self.full_cfg)
        runner3.load_checkpoint(path, resume=True)
        assert runner3.epoch == runner.epoch
        assert runner3.global_iter == runner.global_iter

    def test_custom_hooks(self):
        pass

    def test_iter_based(self):
        pass

    def test_epoch_based(self):
        pass

    def test_train(self):
        pass

    def test_val(self):
        pass

    def test_test(self):
        pass
