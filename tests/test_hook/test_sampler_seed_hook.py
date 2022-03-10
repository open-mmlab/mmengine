# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import DistSamplerSeedHook


class TestDistSamplerSeedHook:

    def test_before_epoch(self):

        hook = DistSamplerSeedHook()
        # Test dataset sampler
        runner = Mock()
        runner.epoch = 1
        runner.data_loader = Mock()
        runner.data_loader.sampler = Mock()
        runner.data_loader.sampler.set_epoch = Mock()
        hook._before_epoch(runner)
        runner.data_loader.sampler.set_epoch.assert_called()
        # Test batch sampler
        runner = Mock()
        runner.data_loader = Mock()
        runner.data_loader.sampler = Mock(spec_set=True)
        runner.data_loader.batch_sampler = Mock()
        runner.data_loader.batch_sampler.sampler = Mock()
        runner.data_loader.batch_sampler.sampler.set_epoch = Mock()
        hook._before_epoch(runner)
        runner.data_loader.batch_sampler.sampler.set_epoch.assert_called()
