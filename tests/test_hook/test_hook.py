# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import pytest

from mmengine.hooks import Hook


class TestHook:

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
        hook.after_val_epoch(runner)

    def test_after_test_epoch(self):
        hook = Hook()
        runner = Mock()
        hook.after_test_epoch(runner)

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

    def test_every_n_iters(self):
        hook = Hook()
        runner = Mock()
        for i in range(100):
            runner.iter = i
            return_val = hook.every_n_iters(runner, 3)
            if (i + 1) % 3 == 0:
                assert return_val
            else:
                assert not return_val

    def test_end_of_epoch(self):
        hook = Hook()
        runner = Mock()

        # last inner iter
        batch_idx = 1
        runner.cur_dataloader.__len__ = Mock(return_value=2)
        runner.cur_dataloader.__len__ = Mock(return_value=2)
        return_val = hook.end_of_epoch(runner, batch_idx)
        assert return_val

        # not the last inner iter
        batch_idx = 0
        return_val = hook.end_of_epoch(runner, batch_idx)
        assert not return_val

    def test_is_last_train_epoch(self):
        hook = Hook()
        runner = Mock()

        # last epoch
        runner.epoch = 1
        runner.train_loop.max_epochs = 2
        return_val = hook.is_last_train_epoch(runner)
        assert return_val

        # not the last epoch
        runner.train_loop.max_epochs = 0
        return_val = hook.is_last_train_epoch(runner)
        assert not return_val

    def test_is_last_iter(self):
        hook = Hook()
        runner = Mock()

        # last iter
        runner.iter = 1
        runner.train_loop.max_iters = 2
        return_val = hook.is_last_iter(runner)
        assert return_val

        # not the last iter
        runner.val_loop.max_iters = 0
        return_val = hook.is_last_iter(runner, mode='val')
        assert not return_val

        runner.test_loop.max_iters = 0
        return_val = hook.is_last_iter(runner, mode='test')
        assert not return_val

        with pytest.raises(ValueError):
            hook.is_last_iter(runner, mode='error_mode')
