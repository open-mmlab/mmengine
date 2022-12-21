# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import MagicMock

from mmengine.hooks import DistSamplerSeedHook
from mmengine.testing import RunnerTestCase


class TestDistSamplerSeedHook(RunnerTestCase):

    def test_before_train_epoch(self):

        hook = DistSamplerSeedHook()
        # Test dataset sampler
        runner = MagicMock()
        runner.epoch = 1
        hook.before_train_epoch(runner)
        runner.train_loop.dataloader.sampler.set_epoch.assert_called()
        # Test batch sampler
        runner.train_loop.dataloader = MagicMock(spec_set=['batch_sampler'])
        hook.before_train_epoch(runner)
        runner.train_loop.dataloader.\
            batch_sampler.sampler.set_epoch.assert_called()

    def test_with_runner(self):
        cfg = self.epoch_based_cfg
        cfg.custom_hooks = [dict(type='DistSamplerSeedHook')]
        runner = self.build_runner(cfg)
        runner.train()
