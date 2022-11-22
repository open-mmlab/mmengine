# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from torch.utils.data import DataLoader, Dataset

from mmengine.dataset.utils import pseudo_collate
from mmengine.model import BaseModel, BaseTTAModel
from mmengine.registry import DATASETS, MODELS, TRANSFORMS
from mmengine.testing import RunnerTestCase


class ToyTTAPipeline:

    def __call__(self, result):
        return {key: [value] for key, value in result.items()}


class ToyTestTimeAugModel(BaseTTAModel):

    def merge_preds(self, data_samples_list):
        result = [sum(x) for x in data_samples_list]
        return result


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        # DDPWrapper requires at least one parameter.
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        return data_samples


class ToyDatasetTTA(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __init__(self, pipeline=None):
        self.pipeline = TRANSFORMS.build(pipeline)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        result = dict(inputs=self.data[index], data_samples=self.label[index])
        result = self.pipeline(result)
        return result


class TestBaseTTAModel(RunnerTestCase):

    def setUp(self) -> None:
        super().setUp()
        DATASETS.register_module(module=ToyDatasetTTA, force=True)
        MODELS.register_module(module=ToyTestTimeAugModel, force=True)
        MODELS.register_module(module=ToyModel, force=True)
        TRANSFORMS.register_module(module=ToyTTAPipeline, force=True)

    def tearDown(self):
        super().tearDown()
        DATASETS.module_dict.pop('ToyDatasetTTA', None)
        MODELS.module_dict.pop('ToyTestTimeAugModel', None)
        MODELS.module_dict.pop('ToyModel', None)
        TRANSFORMS.module_dict.pop('ToyTTAPipeline', None)

    def test_test_step(self):
        model = ToyModel()
        tta_model = ToyTestTimeAugModel(model)
        dict_dataset = [
            dict(inputs=[1, 2], data_samples=[3, 4]) for _ in range(10)
        ]
        tuple_dataset = [([1, 2], [3, 4]) for _ in range(10)]

        dict_dataloader = DataLoader(
            dict_dataset, batch_size=2, collate_fn=pseudo_collate)
        tuple_dataloader = DataLoader(
            tuple_dataset, batch_size=2, collate_fn=pseudo_collate)

        for data in dict_dataloader:
            result = tta_model.test_step(data)
            self.assertEqual(result, [7, 7])

        for data in tuple_dataloader:
            result = tta_model.test_step(data)
            self.assertEqual(result, [7, 7])

    def test_init(self):
        model = ToyModel()
        tta_model = ToyTestTimeAugModel(model)
        self.assertIs(tta_model.module, model)
        # Test build from cfg.
        model = dict(type='ToyModel')
        tta_model = ToyTestTimeAugModel(model)
        self.assertIsInstance(tta_model.module, ToyModel)

    def test_with_runner(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = dict(
            type='ToyTestTimeAugModel', module=dict(type='ToyModel'))
        cfg.test_dataloader.dataset = dict(type='ToyDatasetTTA')
        cfg.test_dataloader.dataset['pipeline'] = dict(type='ToyTTAPipeline')
        runner = self.build_runner(cfg)
        runner.test()

        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            cfg.launcher = 'pytorch'
            self.setup_dist_env()
            runner = self.build_runner(cfg)
            runner.test()
