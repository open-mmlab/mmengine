# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import DistSamplerSeedHook


class TestDistSamplerSeedHook:

    def test_before_epoch(self):

        hook = DistSamplerSeedHook()
        # Test dataset sampler
        runner = Mock()
        runner.epoch = 1
        runner.train_loop.dataloader = Mock()
        runner.train_loop.dataloader.sampler = Mock()
        runner.train_loop.dataloader.sampler.set_epoch = Mock()
        hook.before_train_epoch(runner)
        runner.train_loop.dataloader.sampler.set_epoch.assert_called()
        # Test batch sampler
        runner = Mock()
        runner.train_loop.dataloader = Mock()
        runner.train_loop.dataloader.sampler = Mock(spec_set=True)
        runner.train_loop.dataloader.batch_sampler = Mock()
        runner.train_loop.dataloader.batch_sampler.sampler = Mock()
        runner.train_loop.dataloader.batch_sampler.sampler.set_epoch = Mock()
        hook.before_train_epoch(runner)
        runner.train_loop.dataloader.\
            batch_sampler.sampler.set_epoch.assert_called()
