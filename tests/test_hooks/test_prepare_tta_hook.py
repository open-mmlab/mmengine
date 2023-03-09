# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from torch.utils.data import Dataset

from mmengine.hooks import Hook, PrepareTTAHook
from mmengine.hooks.test_time_aug_hook import build_runner_with_tta
from mmengine.model import BaseModel, BaseTTAModel
from mmengine.registry import DATASETS, MODELS, TRANSFORMS
from mmengine.testing import RunnerTestCase


class ToyDatasetTTA(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __init__(self, pipeline):
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


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        # DDPWrapper requires at least one parameter.
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, inputs, data_samples, mode='tensor'):
        return data_samples


class ToyTestTimeAugModel(BaseTTAModel):

    def merge_preds(self, data_samples_list):
        result = [sum(x) for x in data_samples_list]
        return result


class ToyTTAPipeline:

    def __call__(self, result):
        return {key: [value] for key, value in result.items()}


class TestPrepareTTAHook(RunnerTestCase):

    def setUp(self) -> None:
        super().setUp()
        TRANSFORMS.register_module(module=ToyTTAPipeline, force=True)
        MODELS.register_module(module=ToyModel, force=True)
        MODELS.register_module(module=ToyTestTimeAugModel, force=True)
        DATASETS.register_module(module=ToyDatasetTTA, force=True)

    def tearDown(self):
        super().tearDown()
        TRANSFORMS.module_dict.pop('ToyTTAPipeline', None)
        MODELS.module_dict.pop('ToyModel', None)
        MODELS.module_dict.pop('ToyTestTimeAugModel', None)
        DATASETS.module_dict.pop('ToyDatasetTTA', None)

    def test_init(self):
        tta_cfg = dict(type='ToyTTAModel')
        prepare_tta_hook = PrepareTTAHook(tta_cfg)
        self.assertIsInstance(prepare_tta_hook, Hook)
        self.assertIs(tta_cfg, prepare_tta_hook.tta_cfg)

    def test_before_test(self):
        # Test with epoch based runner.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks.append(
            dict(
                type='PrepareTTAHook',
                tta_cfg=dict(type='ToyTestTimeAugModel')))
        cfg.model = dict(type='ToyModel')
        cfg.test_dataloader.dataset = dict(
            type='ToyDatasetTTA', pipeline=dict(type='ToyTTAPipeline'))
        runner = self.build_runner(cfg)
        self.assertNotIsInstance(runner.model, BaseTTAModel)
        runner.test()
        self.assertIsInstance(runner.model, BaseTTAModel)

        # Test with iteration based runner
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.custom_hooks.append(
            dict(
                type='PrepareTTAHook',
                tta_cfg=dict(type='ToyTestTimeAugModel')))
        cfg.model = dict(type='ToyModel')
        cfg.test_dataloader.dataset = dict(
            type='ToyDatasetTTA', pipeline=dict(type='ToyTTAPipeline'))
        runner = self.build_runner(cfg)
        self.assertNotIsInstance(runner.model, BaseTTAModel)
        runner.test()
        self.assertIsInstance(runner.model, BaseTTAModel)

        # Test with ddp
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            self.setup_dist_env()
            cfg.launcher = 'pytorch'
            runner = self.build_runner(cfg)
            self.assertNotIsInstance(runner.model, BaseTTAModel)
            runner.test()
            self.assertIsInstance(runner.model, BaseTTAModel)


class TestBuildRunenrWithTTA(TestPrepareTTAHook):

    def test_build_runner_with_tta(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model = dict(type='ToyModel')
        cfg.test_dataloader.dataset = dict(type='ToyDatasetTTA')
        cfg.tta_pipeline = dict(type='ToyTTAPipeline')
        cfg.tta_model = dict(type='ToyTestTimeAugModel')
        runner = build_runner_with_tta(cfg)
        runner.test()
        self.assertIsInstance(runner.model, ToyTestTimeAugModel)
