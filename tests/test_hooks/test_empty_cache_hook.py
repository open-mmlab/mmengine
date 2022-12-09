# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

from mmengine.testing import RunnerTestCase


class TestEmptyCacheHook(RunnerTestCase):

    def test_with_runner(self):
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            cfg = self.epoch_based_cfg
            cfg.custom_hooks = [dict(type='EmptyCacheHook')]
            cfg.train_cfg.val_interval = 1e6  # disable val during train
            runner = self.build_runner(cfg)

            runner.train()
            runner.test()
            runner.val()

            target_called_times = runner.max_epochs + 2
            self.assertEqual(mock_empty_cache.call_count, target_called_times)

        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            cfg.custom_hooks = [dict(type='EmptyCacheHook', before_epoch=True)]
            runner = self.build_runner(cfg)
            runner.train()
            runner.val()
            runner.test()
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
            target_called_times = \
                runner.max_epochs * 2 + 4 + \
                len(runner.train_dataloader) * runner.max_epochs + \
                len(runner.val_dataloader) + \
                len(runner.test_dataloader)
            self.assertEqual(mock_empty_cache.call_count, target_called_times)
