# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest.mock import MagicMock

import torch

from mmengine.hooks import NaiveVisualizationHook
from mmengine.registry import DATASETS, MODELS
from mmengine.structures import BaseDataElement
from mmengine.testing.runner_test_case import (RunnerTestCase, ToyDataset,
                                               ToyModel)


class DataSampleDataset(ToyDataset):
    data = torch.randn(12, 3, 28, 28)
    label = torch.ones(12)

    def __getitem__(self, index):
        data_samples = BaseDataElement(labels=self.label[index], )
        return dict(inputs=self.data[index], data_samples=data_samples)


class DataSampleModel(ToyModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor)
        self.linear3 = torch.nn.Linear(28 * 28 * 3, 2)

    def forward(self, inputs, data_samples, mode='tensor'):
        data_samples_ = []
        inputs = torch.stack(inputs)
        inputs = inputs.view(-1, (28 * 28 * 3))
        inputs = self.linear3(inputs)
        for data_sample in data_samples:
            data_samples_.append(data_sample.labels)

        data_samples = torch.stack(data_samples_)
        return super().forward(inputs, data_samples, mode)


class TestNaiveVisualizationHook(RunnerTestCase):

    def setUp(self):
        MODELS.register_module(name='DataSampleModel', module=DataSampleModel)
        DATASETS.register_module(
            name='DataSampleDataset', module=DataSampleDataset)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop('DataSampleModel')
        DATASETS.module_dict.pop('DataSampleDataset')

    def test_init(self):
        NaiveVisualizationHook()

        with self.assertRaisesRegex(AssertionError, '`interval` must'):
            NaiveVisualizationHook(interval=-1)

        with self.assertRaisesRegex(AssertionError, '`draw_gt` must be'):
            NaiveVisualizationHook(draw_gt=1)

        with self.assertRaisesRegex(AssertionError, '`draw_pred` must be'):
            NaiveVisualizationHook(draw_pred=1)

    def test_with_runner(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.test_dataloader.dataset = dict(type='DataSampleDataset')
        cfg.custom_hooks = [dict(type='NaiveVisualizationHook')]
        cfg.model = dict(type='DataSampleModel')
        runner = self.build_runner(cfg)
        runner.visualizer.add_datasample = MagicMock()
        runner.test()
        # length of test_dataloader is 12
        self.assertEqual(runner.visualizer.add_datasample.call_count, 12)
