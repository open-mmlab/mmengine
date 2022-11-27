# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os.path as osp
import tempfile
from unittest import TestCase
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.evaluator import BaseMetric
from mmengine.hooks import EarlyStoppingHook
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, data_sample, mode='tensor'):
        labels = torch.stack(data_sample)
        inputs = torch.stack(inputs)
        outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


class DummyDataset(Dataset):
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


class DummyMetric(BaseMetric):

    default_prefix: str = 'test'

    def __init__(self, length):
        super().__init__()
        self.length = length
        self.best_idx = length // 2
        self.cur_idx = 0
        self.vals = [90, 91, 92, 93, 94, 93] * 2

    def process(self, *args, **kwargs):
        self.results.append(0)

    def compute_metrics(self, *args, **kwargs):
        acc = self.vals[self.cur_idx]
        self.cur_idx += 1
        return dict(acc=acc)


def get_mock_runner():
    runner = Mock()
    runner.train_loop = Mock()
    runner.train_loop.stop_training = False
    return runner


class TestEarlyStoppingHook(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.temp_dir.cleanup()

    def test_init(self):

        hook = EarlyStoppingHook(monitor='acc')
        assert hook.rule == 'greater'
        assert hook.best_score < 0

        hook = EarlyStoppingHook(monitor='loss')
        assert hook.rule == 'less'
        assert hook.best_score > 0

        with pytest.raises(AssertionError):
            EarlyStoppingHook(monitor='accuracy/top1', rule='the world')

    def test_before_run(self):
        runner = Mock()
        runner.train_loop = object()

        # `train_loop` must contain `stop_training` variable.
        with pytest.raises(AssertionError):
            hook = EarlyStoppingHook(monitor='accuracy/top1', rule='greater')
            hook.before_run(runner)

    def test_after_val_epoch(self):
        runner = get_mock_runner()

        # if `monitor` does not match, skip the hook.
        with pytest.warns(UserWarning) as record_warnings:
            metrics = {'accuracy/top1': 0.5, 'loss': 0.23}
            hook = EarlyStoppingHook(monitor='acc', rule='greater')
            hook.after_val_epoch(runner, metrics)

        # Since there will be many warnings thrown, we just need to check
        # if the expected exceptions are thrown
        expected_message = (
            f'Skip early stopping process since the evaluation results '
            f'({metrics.keys()}) do not include `monitor` ({hook.monitor}).')
        for warning in record_warnings:
            if str(warning.message) == expected_message:
                break
        else:
            assert False

        # if `monitor` does not match and strict=True, crash the training.
        with pytest.raises(RuntimeError):
            metrics = {'accuracy/top1': 0.5, 'loss': 0.23}
            hook = EarlyStoppingHook(
                monitor='acc', rule='greater', strict=True)
            hook.after_val_epoch(runner, metrics)

        # Check largest value
        runner = get_mock_runner()
        metrics = [{'accuracy/top1': i / 9.} for i in range(8)]
        hook = EarlyStoppingHook(monitor='accuracy/top1', rule='greater')
        for metric in metrics:
            hook.after_val_epoch(runner, metric)
        assert hook.best_score == 8 / 9

        # Check smalleast value
        runner = get_mock_runner()
        metrics = [{'loss': i / 9.} for i in range(8, 0, -1)]
        hook = EarlyStoppingHook(monitor='loss')
        for metric in metrics:
            hook.after_val_epoch(runner, metric)
        assert hook.best_score == 1 / 9

        # Check stop training
        runner = get_mock_runner()
        metrics = [{'accuracy/top1': i} for i in torch.linspace(98, 99, 8)]
        hook = EarlyStoppingHook(
            monitor='accuracy/top1', rule='greater', min_delta=1)
        for metric in metrics:
            hook.after_val_epoch(runner, metric)
        assert runner.train_loop.stop_training

        # Check patience
        runner = get_mock_runner()
        metrics = [{'accuracy/top1': i} for i in torch.linspace(98, 99, 8)]
        hook = EarlyStoppingHook(
            monitor='accuracy/top1', rule='greater', min_delta=1, patience=5)
        for metric in metrics:
            hook.after_val_epoch(runner, metric)
        assert not runner.train_loop.stop_training

    def test_with_runner(self):
        max_epoch = 10
        work_dir = osp.join(self.temp_dir.name, 'runner_test')
        early_stop_cfg = dict(
            type='EarlyStoppingHook',
            monitor='test/acc',
            rule='greater',
            min_delta=0.4,
        )
        runner = Runner(
            model=ToyModel(),
            work_dir=work_dir,
            train_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=dict(type=DummyMetric, length=max_epoch),
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(
                by_epoch=True, max_epochs=max_epoch, val_interval=1),
            val_cfg=dict(),
            custom_hooks=[early_stop_cfg],
            experiment_name='earlystop_test')
        runner.train()
        assert runner.epoch == 7
