# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import DistSamplerSeedHook


class TestDistSamplerSeedHook:

    def test_before_epoch(self):

        hook = DistSamplerSeedHook()
        # Test dataset sampler
        runner = Mock()
        runner.epoch = 1
        runner.cur_dataloader = Mock()
        runner.cur_dataloader.sampler = Mock()
        runner.cur_dataloader.sampler.set_epoch = Mock()
        hook.before_train_epoch(runner)
        runner.cur_dataloader.sampler.set_epoch.assert_called()
        # Test batch sampler
        runner = Mock()
        runner.cur_dataloader = Mock()
        runner.cur_dataloader.sampler = Mock(spec_set=True)
        runner.cur_dataloader.batch_sampler = Mock()
        runner.cur_dataloader.batch_sampler.sampler = Mock()
        runner.cur_dataloader.batch_sampler.sampler.set_epoch = Mock()
        hook.before_train_epoch(runner)
        runner.cur_dataloader.batch_sampler.sampler.set_epoch.assert_called()
