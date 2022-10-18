# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch

from mmengine.hooks import Hook, PrepareTTAHook
from mmengine.model import BaseTTAModel
from mmengine.registry import MODEL_WRAPPERS
from mmengine.testing import RunnerTestCase


@MODEL_WRAPPERS.register_module()
class ToyTTAModel(BaseTTAModel):

    def merge_preds(self, data_samples_list):
        return data_samples_list[0]


class TestPrepareTTAHook(RunnerTestCase):

    def test_init(self):
        tta_cfg = dict(type='ToyTTAModel')
        prepare_tta_hook = PrepareTTAHook(tta_cfg)
        self.assertIsInstance(prepare_tta_hook, Hook)
        self.assertIs(tta_cfg, prepare_tta_hook.tta_cfg)

    def test_before_test(self):
        # Test with epoch based runner.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks.append(
            dict(type='PrepareTTAHook', tta_cfg=dict(type='ToyTTAModel')))
        runner = self.build_epoch_based_runner(cfg)
        self.assertNotIsInstance(runner.model, BaseTTAModel)
        runner.test()
        self.assertIsInstance(runner.model, BaseTTAModel)

        # Test with iteration based runner
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.custom_hooks.append(
            dict(type='PrepareTTAHook', tta_cfg=dict(type='ToyTTAModel')))
        runner = self.build_iter_based_runner(cfg)
        self.assertNotIsInstance(runner.model, BaseTTAModel)
        self.assertNotIsInstance(runner.model, BaseTTAModel)
        runner.test()
        self.assertIsInstance(runner.model, BaseTTAModel)

        # Test with ddp
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            self.setup_dist_env()
            cfg = copy.deepcopy(self.epoch_based_cfg)
            cfg.launcher = 'pytorch'
            cfg.custom_hooks.append(
                dict(type='PrepareTTAHook', tta_cfg=dict(type='ToyTTAModel')))
            runner = self.build_epoch_based_runner(cfg)
            self.assertNotIsInstance(runner.model, BaseTTAModel)
            runner.test()
            self.assertIsInstance(runner.model, BaseTTAModel)
