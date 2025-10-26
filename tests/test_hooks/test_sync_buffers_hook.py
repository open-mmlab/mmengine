# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

import torch
import torch.distributed as torch_dist
import torch.nn as nn
from torch.testing._internal.common_distributed import DistributedTestBase

from mmengine.dist import all_gather
from mmengine.hooks import SyncBuffersHook
from mmengine.registry import MODELS
from mmengine.testing.runner_test_case import RunnerTestCase, ToyModel


class ToyModuleWithNorm(ToyModel):

    def __init__(self, data_preprocessor=None):
        super().__init__(data_preprocessor=data_preprocessor)
        bn = nn.BatchNorm1d(2)
        self.linear1 = nn.Sequential(self.linear1, bn)

    def init_weights(self):
        for buffer in self.buffers():
            buffer.fill_(
                torch.tensor(torch_dist.get_rank(), dtype=torch.float32))
        return super().init_weights()


class TestSyncBuffersHook(DistributedTestBase, RunnerTestCase):

    def test_sync_buffers_hook(self):
        self.create_pg('cuda')
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
        torch_dist.destroy_process_group()

    def test_with_runner(self):
        MODELS.register_module(module=ToyModuleWithNorm, force=True)
        self.create_pg('cuda')
        RunnerTestCase.setUp(self)
        cfg = self.epoch_based_cfg
        cfg.model = dict(type='ToyModuleWithNorm')
        cfg.launch = 'pytorch'
        cfg.custom_hooks = [dict(type='SyncBuffersHook')]
        runner = self.build_runner(cfg)
        runner.train()

        for buffer in runner.model.buffers():
            buffer1, buffer2 = all_gather(buffer)
            self.assertTrue(torch.allclose(buffer1, buffer2))

    @property
    def world_size(self) -> int:
        return 2
