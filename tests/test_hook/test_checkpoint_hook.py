# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch

from mmengine.hooks import CheckpointHook

sys.modules['file_client'] = sys.modules['mmengine.fileio.file_client']


class MockPetrel:

    _allow_symlink = False

    def __init__(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def allow_symlink(self):
        return self._allow_symlink


prefix_to_backends = {'s3': MockPetrel}


class TestCheckpointHook:

    @patch('file_client.FileClient._prefix_to_backends', prefix_to_backends)
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

        runner.work_dir = 's3://path/of/file'
        checkpoint_hook = CheckpointHook(
            interval=1, by_epoch=True, create_symlink=True)
        checkpoint_hook.before_run(runner)
        assert not checkpoint_hook.args['create_symlink']

    def test_after_train_epoch(self):
        runner = Mock()
        runner.work_dir = './tmp'
        runner.epoch = 9
        runner.meta = dict()
        runner.model = Mock()
        runner.model.buffers = Mock(return_value=[torch.ones(0)])

        # by epoch is True
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, sync_buffer=True)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert (runner.epoch + 1) % 2 == 0
        assert runner.meta['hook_msgs']['last_ckpt'] == './tmp/epoch_10.pth'
        runner.model.buffers.assert_called()

        # epoch can not be evenly divided by 2
        runner.epoch = 10
        checkpoint_hook.after_train_epoch(runner)
        assert runner.meta['hook_msgs']['last_ckpt'] == './tmp/epoch_10.pth'

        # by epoch is False
        runner.epoch = 9
        runner.meta = dict()
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=False)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert runner.meta.get('hook_msgs', None) is None

        # max_keep_ckpts > 0
        with TemporaryDirectory() as tempo_dir:
            runner.work_dir = tempo_dir
            os.system(f'touch {tempo_dir}/epoch_8.pth')
            checkpoint_hook = CheckpointHook(
                interval=2, by_epoch=True, max_keep_ckpts=1)
            checkpoint_hook.before_run(runner)
            checkpoint_hook.after_train_epoch(runner)
            assert (runner.epoch + 1) % 2 == 0
            assert not os.path.exists(f'{tempo_dir}/epoch_8.pth')

    def test_after_train_iter(self):
        runner = Mock()
        runner.work_dir = './tmp'
        runner.iter = 9
        runner.meta = dict()
        runner.model = Mock()
        runner.model.buffers = Mock(return_value=[torch.ones(0)])

        # by epoch is True
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_iter(runner)
        assert runner.meta.get('hook_msgs', None) is None

        # by epoch is False
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=False, sync_buffer=True)
        checkpoint_hook.before_run(runner)
        checkpoint_hook.after_train_iter(runner)
        assert (runner.iter + 1) % 2 == 0
        assert runner.meta['hook_msgs']['last_ckpt'] == './tmp/iter_10.pth'
        runner.model.buffers.assert_called()

        # epoch can not be evenly divided by 2
        runner.iter = 10
        checkpoint_hook.after_train_epoch(runner)
        assert runner.meta['hook_msgs']['last_ckpt'] == './tmp/iter_10.pth'

        # max_keep_ckpts > 0
        runner.iter = 9
        with TemporaryDirectory() as tempo_dir:
            runner.work_dir = tempo_dir
            os.system(f'touch {tempo_dir}/iter_8.pth')
            checkpoint_hook = CheckpointHook(
                interval=2, by_epoch=False, max_keep_ckpts=1)
            checkpoint_hook.before_run(runner)
            checkpoint_hook.after_train_iter(runner)
            assert not os.path.exists(f'{tempo_dir}/iter_8.pth')
