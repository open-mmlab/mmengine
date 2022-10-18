# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import tempfile
import time
from unittest import TestCase

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import mmengine.hooks  # noqa F401
import mmengine.optim  # noqa F401
from mmengine.config import Config
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.registry import DATASETS, METRICS, MODELS
from mmengine.runner import Runner


@MODELS.register_module()
class ToyModel(BaseModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.linear1 = nn.Linear(2, 2)
        self.linear2 = nn.Linear(2, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if isinstance(data_samples, list):
            data_sample = torch.stack(data_samples)
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


@DATASETS.register_module()
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
        return dict(inputs=self.data[index], data_samples=self.label[index])


@METRICS.register_module()
class ToyMetric(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class RunnerTestCase(TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        epoch_based_cfg = dict(
            work_dir=self.temp_dir.name,
            model=dict(type='ToyModel'),
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
            val_evaluator=[dict(type='ToyMetric')],
            test_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            test_evaluator=[dict(type='ToyMetric')],
            optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.1)),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            val_cfg=dict(),
            test_cfg=dict(),
            default_hooks=dict(logger=dict(type='LoggerHook', interval=1)),
            custom_hooks=[],
            env_cfg=dict(dist_cfg=dict(backend='nccl')),
            experiment_name='test1')
        self.epoch_based_cfg = Config(epoch_based_cfg)
        self.iter_based_cfg: Config = copy.deepcopy(self.epoch_based_cfg)
        self.iter_based_cfg.train_dataloader = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='InfiniteSampler', shuffle=True),
            batch_size=3,
            num_workers=0)
        self.iter_based_cfg.log_processor = dict(by_epoch=False)

        # prepare iter based cfg.
        self.iter_based_cfg.train_cfg = dict(by_epoch=False, max_iters=12)
        self.iter_based_cfg.default_hooks = dict(
            logger=dict(type='LoggerHook', interval=1),
            checkpoint=dict(
                type='CheckpointHook', interval=12, by_epoch=False))

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.temp_dir.cleanup()

    def build_epoch_based_runner(self, cfg):
        cfg.experiment_name = self.experiment_name
        runner = Runner.from_cfg(cfg)
        return runner

    def build_iter_based_runner(self, cfg):
        cfg.experiment_name = self.experiment_name
        runner = Runner.from_cfg(cfg)
        return runner

    @property
    def experiment_name(self):
        return f'{self._testMethodName}_{time.time_ns()}'

    def setup_dist_env(self):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29600'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
