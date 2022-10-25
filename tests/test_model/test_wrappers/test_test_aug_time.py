# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import torch
from torch.utils.data import DataLoader, Dataset

from mmengine.model import BaseModel, BaseTTAModel, build_runner_with_tta
from mmengine.registry import DATASETS, MODELS, TRANSFORMS
from mmengine.testing import RunnerTestCase


def pseudo_pipeline(x):
    return x


class ToyTTAPipeline:

    def __call__(self, result):
        return {key: [value] for key, value in result.items()}


class ToyTestTimeAugModel(BaseTTAModel):

    def merge_preds(self, data_samples_list):
        result = [sum(x) for x in data_samples_list]
        return result


class TTAToyModel(BaseModel):

    def forward(self, inputs, data_samples, mode='tensor'):
        return data_samples


class ToyDatasetTTA(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __init__(self, pipeline=None):
        self.pipeline = pseudo_pipeline if pipeline is None else \
            TRANSFORMS.build(pipeline)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        result = dict(inputs=self.data[index], data_samples=self.label[index])
        result = self.pipeline(result)
        return result


class TestBaseTTAModel(TestCase):

    def setUp(self) -> None:
        dict_dataset = [
            dict(inputs=[1, 2], data_samples=[3, 4]) for _ in range(10)
        ]
        tuple_dataset = [([1, 2], [3, 4]) for _ in range(10)]
        self.model = TTAToyModel()
        self.dict_dataloader = DataLoader(dict_dataset, batch_size=2)
        self.tuple_dataloader = DataLoader(tuple_dataset, batch_size=2)

    def test_test_step(self):
        tta_model = ToyTestTimeAugModel(self.model)

        # Test dict dataset

        for data in self.dict_dataloader:
            # Test step will call forward.
            result = tta_model.test_step(data)
            self.assertEqual(result, [7, 7])

        for data in self.tuple_dataloader:
            result = tta_model.test_step(data)
            self.assertEqual(result, [7, 7])

    def test_init(self):
        tta_model = ToyTestTimeAugModel(self.model)
        self.assertIs(tta_model.module, self.model)
        # Test build from cfg.
        model = dict(type='TTAToyModel')
        tta_model = ToyTestTimeAugModel(model)
        self.assertIsInstance(tta_model.module, TTAToyModel)


class TestBuildRunenrWithTTA(RunnerTestCase):

    def setUp(self) -> None:
        DATASETS.register_module(module=ToyDatasetTTA)
        TRANSFORMS.register_module(module=ToyTTAPipeline)
        MODELS.register_module(module=ToyTestTimeAugModel)
        super().setUp()

    def test_build_runner_with_tta(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.test_dataloader.dataset = dict(type='ToyDatasetTTA')
        cfg.tta_pipeline = dict(type='ToyTTAPipeline')
        cfg.tta_model = dict(type='ToyTestTimeAugModel')
        runner = build_runner_with_tta(cfg)
        runner.test()
        self.assertIsInstance(runner.model, ToyTestTimeAugModel)
