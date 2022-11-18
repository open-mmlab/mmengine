# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.evaluator import BaseMetric
from mmengine.hooks import EarlyStoppingHook
from mmengine.logging import MessageHub
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
    runner.message_hub = MessageHub.get_instance('test_after_val_epoch')
    return runner


class TestEarlyStoppingHook:

    def test_init(self):

        hook = EarlyStoppingHook(metric='acc')
        assert hook.rule == 'greater'

        hook = EarlyStoppingHook(metric='loss')
        assert hook.rule == 'less'

        with pytest.raises(AssertionError):
            EarlyStoppingHook(metric='accuracy/top1', rule='the world')

    def test_before_run(self):
        runner = Mock()
        runner.train_loop = Mock()

        # `train_loop` must contain `stop_training` variable.
        with pytest.raises(AssertionError):
            hook = EarlyStoppingHook(metric='accuracy/top1', rule='greater')
            hook.before_run(runner)

    def test_after_val_epoch(self, tmp_path):
        runner = get_mock_runner()

        # if `metric` does not match, skip the hook.
        with pytest.warns(UserWarning) as record_warnings:
            metrics = {'accuracy/top1': 0.5, 'loss': 0.23}
            hook = EarlyStoppingHook(metric='acc', rule='greater')
            hook.after_val_epoch(runner, metrics)

        # Since there will be many warnings thrown, we just need to check
        # if the expected exceptions are thrown
        expected_message = (
            f'Skip early stopping process since the evaluation results '
            f'({metrics.keys()}) do not include `metric` ({hook.metric}).')
        for warning in record_warnings:
            if str(warning.message) == expected_message:
                break
        else:
            assert False

        # Check largest 5 values
        runner = get_mock_runner()
        metrics = [{'accuracy/top1': i / 10.} for i in range(8)]
        hook = EarlyStoppingHook(metric='accuracy/top1', rule='greater')
        for metric in metrics:
            hook.after_val_epoch(runner, metric)
        assert all([i / 10 in hook.pool_values for i in range(3, 8)])

        # Check smalleast 3 values
        runner = get_mock_runner()
        metrics = [{'loss': i / 10.} for i in range(8)]
        hook = EarlyStoppingHook(metric='loss', pool_size=3)
        for metric in metrics:
            hook.after_val_epoch(runner, metric)
        assert all([i / 10 in hook.pool_values for i in range(3)])

        # Check stop training
        runner = get_mock_runner()
        metrics = [{'accuracy/top1': i} for i in torch.linspace(98, 99, 8)]
        hook = EarlyStoppingHook(
            metric='accuracy/top1', rule='greater', delta=1)
        for metric in metrics:
            hook.after_val_epoch(runner, metric)
        assert runner.train_loop.stop_training

        # Check patience
        runner = get_mock_runner()
        metrics = [{'accuracy/top1': i} for i in torch.linspace(98, 99, 8)]
        hook = EarlyStoppingHook(
            metric='accuracy/top1', rule='greater', delta=1, patience=5)
        for metric in metrics:
            hook.after_val_epoch(runner, metric)
        assert not runner.train_loop.stop_training

    def test_with_runner(self, tmp_path):
        max_epoch = 10
        work_dir = osp.join(str(tmp_path), 'runner_test')
        early_stop_cfg = dict(
            type='EarlyStoppingHook',
            metric='test/acc',
            rule='greater',
            delta=0.4,
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
            default_hooks=dict(early_stop=early_stop_cfg))
        runner.train()
        assert runner.epoch == 7
