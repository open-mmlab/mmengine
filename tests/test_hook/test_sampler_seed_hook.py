# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import DistSamplerSeedHook


class TestDistSamplerSeedHook:

    def test_before_epoch(self):

        hook = DistSamplerSeedHook()
        # Test dataset sampler
        runner = Mock()
        runner.dataloader = Mock()
        runner.dataloader.sampler = Mock()
        runner.dataloader.sampler.set_epoch = Mock()
        hook.before_epoch(runner)
        # Test batch sampler
        runner = Mock()
        runner.data_loader = Mock()
        runner.data_loader.sampler = Mock(spec_set=True)
        runner.data_loader.batch_sampler = Mock()
        runner.data_loader.batch_sampler.sampler = Mock()
        runner.data_loader.batch_sampler.sampler.set_epoch = Mock()
        hook.before_epoch(runner)
