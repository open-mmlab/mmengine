# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from mmengine.hooks import IterTimerHook
from mmengine.logging import MessageHub


def time_patch():
    if not hasattr(time_patch, 'time'):
        time_patch.time = 0
    else:
        time_patch.time += 1
    return time_patch.time


class TestIterTimerHook(TestCase):

    def setUp(self) -> None:
        self.hook = IterTimerHook()

    def test_init(self):
        assert self.hook.time_sec_tot == 0
        assert self.hook.start_iter == 0

    def test_before_train(self):
        runner = MagicMock()
        runner.iter = 1
        self.hook.before_train(runner)
        assert self.hook.start_iter == 1

    def test_before_epoch(self):
        runner = Mock()
        self.hook._before_epoch(runner)
        assert isinstance(self.hook.t, float)

    @patch('time.time', MagicMock(return_value=1))
    def test_before_iter(self):
        runner = MagicMock()
        runner.log_buffer = dict()
        self.hook._before_epoch(runner)
        for mode in ('train', 'val', 'test'):
            self.hook._before_iter(runner, batch_idx=1, mode=mode)
            runner.message_hub.update_scalar.assert_called_with(
                f'{mode}/data_time', 0)

    @patch('time.time', time_patch)
    def test_after_iter(self):
        runner = MagicMock()
        runner.log_buffer = dict()
        runner.log_processor.window_size = 10
        runner.max_iters = 100
        runner.iter = 0
        runner.test_dataloader = [0] * 20
        runner.val_dataloader = [0] * 20
        runner.message_hub = MessageHub.get_instance('test_iter_timer_hook')

        self.hook.before_run(runner)
        self.hook._before_epoch(runner)
        # eta = (100 - 10) / 1
        for _ in range(10):
            self.hook._after_iter(runner, 1)
            runner.iter += 1
        assert runner.message_hub.get_info('eta') == 90

        for i in range(10):
            self.hook._after_iter(runner, batch_idx=i, mode='val')
        assert runner.message_hub.get_info('eta') == 10

        for i in range(11, 20):
            self.hook._after_iter(runner, batch_idx=i, mode='val')
        assert runner.message_hub.get_info('eta') == 0

        self.hook.after_val_epoch(runner)

        for i in range(10):
            self.hook._after_iter(runner, batch_idx=i, mode='test')
        assert runner.message_hub.get_info('eta') == 10

        for i in range(11, 20):
            self.hook._after_iter(runner, batch_idx=i, mode='test')
        assert runner.message_hub.get_info('eta') == 0
