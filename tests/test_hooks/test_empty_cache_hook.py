# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import pytest

from mmengine.device import is_cuda_available
from mmengine.testing import RunnerTestCase


class TestEmptyCacheHook(RunnerTestCase):

    @pytest.mark.skipif(
        not is_cuda_available(), reason='cuda should be available')
    def test_with_runner(self):
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            cfg = self.epoch_based_cfg
            cfg.custom_hooks = [dict(type='EmptyCacheHook')]
            cfg.train_cfg.val_interval = 1e6  # disable validation during training  # noqa: E501
            runner = self.build_runner(cfg)

            runner.train()
            runner.test()
            runner.val()

            # Call `torch.cuda.empty_cache` after each epoch:
            #   runner.train: `max_epochs` times.
            #   runner.val: `1` time.
            #   runner.test: `1` time.
            target_called_times = runner.max_epochs + 2
            self.assertEqual(mock_empty_cache.call_count, target_called_times)

        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            cfg.custom_hooks = [dict(type='EmptyCacheHook', before_epoch=True)]
            runner = self.build_runner(cfg)

            runner.train()
            runner.val()
            runner.test()

            # Call `torch.cuda.empty_cache` after/before each epoch:
            #   runner.train: `max_epochs*2` times.
            #   runner.val: `1*2` times.
            #   runner.test: `1*2` times.

            target_called_times = runner.max_epochs * 2 + 4
            self.assertEqual(mock_empty_cache.call_count, target_called_times)

        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            cfg.custom_hooks = [
                dict(
                    type='EmptyCacheHook', after_iter=True, before_epoch=True)
            ]
            runner = self.build_runner(cfg)

            runner.train()
            runner.val()
            runner.test()

            # Call `torch.cuda.empty_cache` after/before each epoch,
            # after each iteration:
            #   runner.train: `max_epochs*2 + len(dataloader)*max_epochs` times.  # noqa: E501
            #   runner.val: `1*2 + len(val_dataloader)` times.
            #   runner.test: `1*2 + len(val_dataloader)` times.

            target_called_times = \
                runner.max_epochs * 2 + 4 + \
                len(runner.train_dataloader) * runner.max_epochs + \
                len(runner.val_dataloader) + \
                len(runner.test_dataloader)
            self.assertEqual(mock_empty_cache.call_count, target_called_times)
