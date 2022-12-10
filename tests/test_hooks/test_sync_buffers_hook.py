# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest.mock import patch

import torch.distributed as torch_dist
import torch.nn as nn

from mmengine.hooks import SyncBuffersHook
from mmengine.testing._internal import MultiProcessTestCase
from mmengine.testing.runner_test_case import RunnerTestCase, ToyModel


class ToyModuleWithNorm(ToyModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        self.linear1 = nn.Sequential(
            self.linear1,
            nn.BatchNorm1d(2),
        )


class TestSyncBuffersHook(MultiProcessTestCase, RunnerTestCase):

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def test_sync_buffers_hook(self):
        RunnerTestCase.setup_dist_env(self)
        os.environ['RANK'] = str(self.rank)
        torch_dist.init_process_group(
            backend='gloo', rank=self.rank, world_size=self.world_size)
        ...
        # cfg = self.epoch_based_cfg
        # cfg.custom_hooks = [dict(type='SyncBuffersHook')]
        # cfg.launch = 'pytorch'
        # self.setup_dist_env()
        # runner = self.build_runner(cfg)
        # hook = self._get_sync_buffers_hook(runner)
        # hook.after_train_epoch(runner)

    def test_with_runner(self):
        cfg = self.epoch_based_cfg
        cfg.custom_hooks = [dict(type='SyncBuffersHook')]
        runner = self.build_runner(cfg)
        runner.train()

    def _get_sync_buffers_hook(self, runner):
        for hook in runner.hooks:
            if isinstance(hook, SyncBuffersHook):
                return hook
