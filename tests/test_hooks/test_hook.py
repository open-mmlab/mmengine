# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

from mmengine.hooks import Hook
from mmengine.testing import RunnerTestCase


class TestHook(RunnerTestCase):

    def test_before_run(self):
        hook = Hook()
        runner = Mock()
        hook.before_run(runner)

    def test_after_run(self):
        hook = Hook()
        runner = Mock()
        hook.after_run(runner)

    def test_before_epoch(self):
        hook = Hook()
        runner = Mock()
        hook._before_epoch(runner)

    def test_after_epoch(self):
        hook = Hook()
        runner = Mock()
        hook._after_epoch(runner)

    def test_before_iter(self):
        hook = Hook()
        runner = Mock()
        data_batch = {}
        hook._before_iter(runner, data_batch)

    def test_after_iter(self):
        hook = Hook()
        runner = Mock()
        data_batch = {}
        outputs = {}
        hook._after_iter(runner, data_batch, outputs)

    def test_before_save_checkpoint(self):
        hook = Hook()
        runner = Mock()
        checkpoint = {}
        hook.before_save_checkpoint(runner, checkpoint)

    def test_after_load_checkpoint(self):
        hook = Hook()
        runner = Mock()
        checkpoint = {}
        hook.after_load_checkpoint(runner, checkpoint)

    def test_before_train_epoch(self):
        hook = Hook()
        runner = Mock()
        hook.before_train_epoch(runner)

    def test_before_val_epoch(self):
        hook = Hook()
        runner = Mock()
        hook.before_val_epoch(runner)

    def test_before_test_epoch(self):
        hook = Hook()
        runner = Mock()
        hook.before_test_epoch(runner)

    def test_after_train_epoch(self):
        hook = Hook()
        runner = Mock()
        hook.after_train_epoch(runner)

    def test_after_val_epoch(self):
        hook = Hook()
        runner = Mock()
        hook.after_val_epoch(runner, {})

    def test_after_test_epoch(self):
        hook = Hook()
        runner = Mock()
        hook.after_test_epoch(runner, {})

    def test_before_train_iter(self):
        hook = Hook()
        runner = Mock()
        data_batch = {}
        hook.before_train_iter(runner, data_batch)

    def test_before_val_iter(self):
        hook = Hook()
        runner = Mock()
        data_batch = {}
        hook.before_val_iter(runner, data_batch)

    def test_before_test_iter(self):
        hook = Hook()
        runner = Mock()
        data_batch = {}
        hook.before_test_iter(runner, data_batch)

    def test_after_train_iter(self):
        hook = Hook()
        runner = Mock()
        data_batch = {}
        outputs = {}
        hook.after_train_iter(runner, data_batch, outputs)

    def test_after_val_iter(self):
        hook = Hook()
        runner = Mock()
        data_batch = {}
        outputs = {}
        hook.after_val_iter(runner, data_batch, outputs)

    def test_after_test_iter(self):
        hook = Hook()
        runner = Mock()
        data_batch = {}
        outputs = {}
        hook.after_test_iter(runner, data_batch, outputs)

    def test_every_n_epochs(self):
        hook = Hook()
        runner = Mock()

        for i in range(100):
            runner.epoch = i
            return_val = hook.every_n_epochs(runner, 3)
            if (i + 1) % 3 == 0:
                assert return_val
            else:
                assert not return_val

    def test_every_n_inner_iters(self):
        hook = Hook()

        for i in range(100):
            return_val = hook.every_n_inner_iters(i, 3)
            if (i + 1) % 3 == 0:
                assert return_val
            else:
                assert not return_val

    def test_every_n_train_iters(self):
        hook = Hook()
        runner = Mock()
        for i in range(100):
            runner.iter = i
            return_val = hook.every_n_train_iters(runner, 3)
            if (i + 1) % 3 == 0:
                assert return_val
            else:
                assert not return_val

    def test_end_of_epoch(self):
        hook = Hook()

        # last inner iter
        batch_idx = 1
        dataloader = Mock()
        dataloader.__len__ = Mock(return_value=2)
        return_val = hook.end_of_epoch(dataloader, batch_idx)
        assert return_val

        # not the last inner iter
        batch_idx = 0
        return_val = hook.end_of_epoch(dataloader, batch_idx)
        assert not return_val

    def test_is_last_train_epoch(self):
        hook = Hook()
        runner = Mock()

        # last epoch
        runner.epoch = 1
        runner.max_epochs = 2
        return_val = hook.is_last_train_epoch(runner)
        assert return_val

        # not the last epoch
        runner.max_epochs = 0
        return_val = hook.is_last_train_epoch(runner)
        assert not return_val

    def test_is_last_train_iter(self):
        hook = Hook()
        runner = Mock()

        # last iter
        runner.iter = 1
        runner.max_iters = 2
        return_val = hook.is_last_train_iter(runner)
        assert return_val

    def test_get_triggered_stages(self):

        class CustomHook(Hook):

            def after_train(self, runner):
                return super().after_train(runner)

        hook = CustomHook()
        triggered_stages = hook.get_triggered_stages()
        self.assertListEqual(triggered_stages, ['after_train'])

        class CustomHook(Hook):

            def _before_iter(self, runner):
                ...

        hook = CustomHook()
        triggered_stages = hook.get_triggered_stages()
        self.assertEqual(len(triggered_stages), 3)
        self.assertSetEqual(
            set(triggered_stages),
            {'before_train_iter', 'before_val_iter', 'before_test_iter'})

        class CustomHook(Hook):

            def is_last_train_epoch(self, runner):
                ...

        hook = CustomHook()
        triggered_stages = hook.get_triggered_stages()
        self.assertEqual(len(triggered_stages), 0)
