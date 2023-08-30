# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.testing import RunnerTestCase
from mmengine.tune._report_hook import ReportingHook
from unittest.mock import MagicMock

class TestReportingHook(RunnerTestCase):
    def test_append_score(self):
        hook = ReportingHook(monitor='acc', max_scoreboard_len=3)

        # Adding scores to the scoreboard
        hook._append_score(0.5)
        hook._append_score(0.6)
        hook._append_score(0.7)
        self.assertEqual(hook.scoreboard, [0.5, 0.6, 0.7])

        # When exceeding max length, it should pop the first item
        hook._append_score(0.8)
        self.assertEqual(hook.scoreboard, [0.6, 0.7, 0.8])

    def test_should_stop(self):
        runner = MagicMock(iter=3, epoch=1)

        # Test with tuning_iter
        hook1 = ReportingHook(monitor='acc', tuning_iter=5)
        self.assertFalse(hook1._should_stop(runner))
        runner.iter = 4
        self.assertTrue(hook1._should_stop(runner))

        # Test with tuning_epoch
        hook2 = ReportingHook(monitor='acc', tuning_epoch=3)
        self.assertFalse(hook2._should_stop(runner))
        runner.epoch = 2
        self.assertTrue(hook2._should_stop(runner))

    def test_report_score(self):
        hook1 = ReportingHook(monitor='acc', report_op='latest')
        hook1.scoreboard = [0.5, 0.6, 0.7]
        self.assertEqual(hook1.report_score(), 0.7)

        hook2 = ReportingHook(monitor='acc', report_op='mean')
        hook2.scoreboard = [0.5, 0.6, 0.7]
        self.assertEqual(hook2.report_score(), 0.6)

        # Test with an empty scoreboard
        hook3 = ReportingHook(monitor='acc', report_op='mean')
        self.assertIsNone(hook3.report_score())

    def test_clear(self):
        hook = ReportingHook(monitor='acc')
        hook.scoreboard = [0.5, 0.6, 0.7]
        hook.clear()
        self.assertEqual(hook.scoreboard, [])

    def test_after_train_iter(self):
        runner = MagicMock(iter=3, epoch=1)
        runner.log_processor.get_log_after_iter = MagicMock(return_value=({'acc': 0.9}, 'log_str'))
        
        # Check if the monitored score gets appended correctly
        hook = ReportingHook(monitor='acc')
        hook.after_train_iter(runner, 0)
        self.assertEqual(hook.scoreboard[-1], 0.9)

        # Check if no score is appended for a non-existent metric
        hook2 = ReportingHook(monitor='non_existent')
        hook2.after_train_iter(runner, 0)
        self.assertEqual(len(hook2.scoreboard), 0)
        
        # Check that training stops if tuning_iter is reached
        runner.iter = 5
        hook3 = ReportingHook(monitor='acc', tuning_iter=5)
        hook3.after_train_iter(runner, 0)
        self.assertTrue(runner.train_loop.stop_training)

    def test_after_val_epoch(self):
        runner = MagicMock(iter=3, epoch=1)

        # Check if the monitored score gets appended correctly from metrics
        metrics = {'acc': 0.9}
        hook = ReportingHook(monitor='acc')
        hook.after_val_epoch(runner, metrics=metrics)
        self.assertEqual(hook.scoreboard[-1], 0.9)

        # Check that no score is appended if the metric is missing from metrics
        metrics = {'loss': 0.1}
        hook2 = ReportingHook(monitor='acc')
        hook2.after_val_epoch(runner, metrics=metrics)
        self.assertEqual(len(hook2.scoreboard), 0)

    def test_with_runner(self):
        runner = self.build_runner(self.epoch_based_cfg)
        acc_hook = ReportingHook(monitor='test/acc', tuning_epoch=1)
        runner.register_hook(acc_hook, priority='VERY_LOW')
        runner.train()
        self.assertEqual(runner.epoch, 1)
        score = acc_hook.report_score()
        self.assertAlmostEqual(score, 1)
