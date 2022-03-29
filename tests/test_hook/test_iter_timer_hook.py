# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock, MagicMock, patch
from unittest import TestCase

from mmengine.logging import MessageHub
from mmengine.hooks import IterTimerHook


class TestIterTimerHook(TestCase):
    def setUp(self) -> None:
        self.hook = IterTimerHook()
    
    def test_init(self):
        assert self.hook.time_sec_tot == 0
        assert self.hook.start_iter == 0
    
    def test_before_run(self):
        self.hook.before_run(Mock())
        self.hook.start_iter = 1
        assert self.hook.start_iter == 0
    
    def test_before_epoch(self):
        runner = Mock()
        self.hook._before_epoch(runner)
        assert isinstance(self.hook.t, float)

    @patch('time.time', MagicMock(return_value=1))
    def test_before_iter(self):
        runner = MagicMock()
        runner.log_buffer = dict()
        self.hook._before_epoch(runner)
        for mode in ['train', 'val', 'test']:
            self.hook._before_iter(runner, mode=mode)
            runner.message_hub.update_scalar.assert_called_with(
                f'{mode}/data_time', 0)

    @patch('time.time', MagicMock(return_value=1))
    def test_after_iter(self):
        runner = MagicMock()
        runner.log_buffer = dict()
        runner.log_processor.window_size = 10
        runner.train_loop.max_iters = 100
        runner.iter = 0
        self.hook._before_epoch(runner)
        self.hook.before_run(runner)
        self.hook._after_iter(runner)
        runner.message_hub.update_scalar.assert_called()
        runner.message_hub.get_log.assert_not_called()
        runner.message_hub.update_info.assert_not_called()
        runner.message_hub = MessageHub.get_instance('test_iter_timer_hook')
        runner.iter = 9
        self.hook._after_iter(runner)
        assert runner.message_hub.get_info('eta') == 0
