# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest.mock import patch

from mmengine.hooks import IterTimerHook
from mmengine.testing import RunnerTestCase


class patched_time:
    count = 0

    @classmethod
    def time(cls):
        result = cls.count
        cls.count += 1
        return result


class TestIterTimerHook(RunnerTestCase):

    @patch('mmengine.hooks.iter_timer_hook.time', patched_time)
    def test_before_iter(self):
        runner = self.build_runner(self.epoch_based_cfg)
        hook = self._get_iter_timer_hook(runner)
        for mode in ('train', 'val', 'test'):
            hook._before_epoch(runner)
            hook._before_iter(runner, batch_idx=1, mode=mode)
            time = runner.message_hub.get_scalar(
                f'{mode}/data_time')._log_history
            self.assertEqual(list(time)[-1], 1)

    @patch('mmengine.hooks.iter_timer_hook.time', patched_time)
    def test_after_iter(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.train_cfg.max_iters = 100
        runner = self.build_runner(cfg)
        hook = self._get_iter_timer_hook(runner)

        hook.before_run(runner)
        hook._before_epoch(runner)

        # 4 iteration per epoch, totally 2 epochs
        # Under pathced_time, before_iter will cost "1s" and after_iter will
        # cost "1s", so the total time for each iteration is 2s.
        for i in range(10):
            hook.before_train_iter(runner, i)
            hook.after_train_iter(runner, i)
            runner.train_loop._iter += 1

        # Left 90 iterations, so the ETA should be 90 * 2s
        self.assertEqual(runner.message_hub.get_info('eta'), 180)
        hook.after_train_epoch(runner)

        for i in range(2):
            hook.before_val_iter(runner, i)
            hook.after_val_iter(runner, batch_idx=i)
        self.assertEqual(runner.message_hub.get_info('eta'), 4)

        for i in range(2, 4):
            hook.before_val_iter(runner, i)
            hook.after_val_iter(runner, batch_idx=i)
        hook.after_val_epoch(runner)
        self.assertEqual(runner.message_hub.get_info('eta'), 0)

        for i in range(2):
            hook.before_test_iter(runner, i)
            hook.after_test_iter(runner, batch_idx=i)
        self.assertEqual(runner.message_hub.get_info('eta'), 4)

        for i in range(2, 4):
            hook.before_test_iter(runner, i)
            hook.after_test_iter(runner, batch_idx=i)
        hook.after_test_epoch(runner)
        self.assertEqual(runner.message_hub.get_info('eta'), 0)

    def test_with_runner(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        cfg.train_cfg.val_interval = 1e6  # disable validation

        with patch('mmengine.hooks.iter_timer_hook.time', patched_time):
            runner.train()

        # 4 iteration per epoch, totally 2 epochs
        # Under pathced_time, before_iter will cost "1s" and after_iter will
        # cost "1s", so the total time for each iteration is 2s.
        train_time = runner.message_hub.log_scalars['train/time']._log_history
        self.assertEqual(len(train_time), 8)
        self.assertListEqual(list(train_time), [2] * 8)
        eta = runner.message_hub.runtime_info['eta']
        self.assertEqual(eta, 0)

    def _get_iter_timer_hook(self, runner):
        for hook in runner.hooks:
            if isinstance(hook, IterTimerHook):
                return hook
