# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest.mock import Mock

from mmengine.hooks import CheckpointHook


class TestCheckpointHook:

    def test_before_run(self):
        runner = Mock()
        runner.work_dir = './tmp'

        # the out_dir of the checkpoint hook is None
        checkpoint_hook = CheckpointHook(interval=1, by_epoch=True)
        checkpoint_hook.before_run(runner)
        assert checkpoint_hook.out_dir == runner.work_dir

        # the out_dir of the checkpoint hook is not None
        checkpoint_hook = CheckpointHook(
            interval=1, by_epoch=True, out_dir='test_dir')
        checkpoint_hook.before_run(runner)
        assert checkpoint_hook.out_dir == 'test_dir/tmp'

        # create_symlink in args and create_symlink is True
        checkpoint_hook = CheckpointHook(
            interval=1, by_epoch=True, out_dir='test_dir', create_symlink=True)
        checkpoint_hook.before_run(runner)
        assert checkpoint_hook.args['create_symlink']

    def test_after_train_epoch(self):
        runner = Mock()
        runner.work_dir = './tmp'
        runner.epoch = 9
        runner.meta = dict()

        # by epoch is True
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert (runner.epoch + 1) % 2 == 0

        # by epoch is False
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=False)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert (runner.epoch + 1) % 2 == 0

        # max_keep_ckpts > 0
        os.mkdir('./tmp')
        os.system('touch ./tmp/epoch_8.pth')
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, max_keep_ckpts=1)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert (runner.epoch + 1) % 2 == 0
        os.system('rm -rf ./tmp')

    def test_after_train_iter(self):
        runner = Mock()
        runner.work_dir = './tmp'
        runner.iter = 9
        runner.meta = dict()

        # by epoch is True
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_iter(runner)
        assert (runner.iter + 1) % 2 == 0

        # by epoch is False
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=False)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_iter(runner)
        assert (runner.iter + 1) % 2 == 0

        # max_keep_ckpts > 0
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=False, max_keep_ckpts=1)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_iter(runner)
        assert (runner.iter + 1) % 2 == 0
