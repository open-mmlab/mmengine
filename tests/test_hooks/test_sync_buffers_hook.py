# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest.mock import MagicMock

import torch
import torch.distributed as torch_dist
import torch.nn as nn

from mmengine.dist import all_gather
from mmengine.hooks import SyncBuffersHook
from mmengine.registry import MODELS
from mmengine.testing._internal import MultiProcessTestCase
from mmengine.testing.runner_test_case import RunnerTestCase, ToyModel


class ToyModuleWithNorm(ToyModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        bn = nn.BatchNorm1d(2)
        self.linear1 = nn.Sequential(self.linear1, bn)

    def init_weights(self):
        for buffer in self.buffers():
            buffer.fill_(
                torch.tensor(int(os.environ['RANK']), dtype=torch.float32))
        return super().init_weights()


class TestSyncBuffersHook(MultiProcessTestCase, RunnerTestCase):

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def prepare_subprocess(self):
        MODELS.register_module(module=ToyModuleWithNorm, force=True)
        super(MultiProcessTestCase, self).setUp()

    def test_sync_buffers_hook(self):
        self.setup_dist_env()
        runner = MagicMock()
        runner.model = ToyModuleWithNorm()
        runner.model.init_weights()

        for buffer in runner.model.buffers():
            buffer1, buffer2 = all_gather(buffer)
            self.assertFalse(torch.allclose(buffer1, buffer2))

        hook = SyncBuffersHook()
        hook.after_train_epoch(runner)

        for buffer in runner.model.buffers():
            buffer1, buffer2 = all_gather(buffer)
            self.assertTrue(torch.allclose(buffer1, buffer2))

    def test_with_runner(self):
        self.setup_dist_env()
        cfg = self.epoch_based_cfg
        cfg.model = dict(type='ToyModuleWithNorm')
        cfg.launch = 'pytorch'
        cfg.custom_hooks = [dict(type='SyncBuffersHook')]
        runner = self.build_runner(cfg)
        runner.train()

        for buffer in runner.model.buffers():
            buffer1, buffer2 = all_gather(buffer)
            self.assertTrue(torch.allclose(buffer1, buffer2))

    def setup_dist_env(self):
        super().setup_dist_env()
        os.environ['RANK'] = str(self.rank)
        torch_dist.init_process_group(
            backend='gloo', rank=self.rank, world_size=self.world_size)
