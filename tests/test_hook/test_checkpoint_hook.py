# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from unittest.mock import Mock, patch

from mmengine.hooks import CheckpointHook


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

    @patch('mmengine.fileio.file_client.FileClient._prefix_to_backends',
           prefix_to_backends)
    def test_before_train(self, tmp_path):
        runner = Mock()
        work_dir = str(tmp_path)
        runner.work_dir = work_dir

        # the out_dir of the checkpoint hook is None
        checkpoint_hook = CheckpointHook(interval=1, by_epoch=True)
        checkpoint_hook.before_train(runner)
        assert checkpoint_hook.out_dir == runner.work_dir

        # the out_dir of the checkpoint hook is not None
        checkpoint_hook = CheckpointHook(
            interval=1, by_epoch=True, out_dir='test_dir')
        checkpoint_hook.before_train(runner)
        assert checkpoint_hook.out_dir == (
            f'test_dir/{osp.basename(work_dir)}')

    def test_after_train_epoch(self, tmp_path):
        runner = Mock()
        work_dir = str(tmp_path)
        runner.work_dir = tmp_path
        runner.epoch = 9
        runner.meta = dict()
        runner.model = Mock()

        # by epoch is True
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert (runner.epoch + 1) % 2 == 0
        assert runner.meta['hook_msgs']['last_ckpt'] == (
            f'{work_dir}/epoch_10.pth')
        # epoch can not be evenly divided by 2
        runner.epoch = 10
        checkpoint_hook.after_train_epoch(runner)
        assert runner.meta['hook_msgs']['last_ckpt'] == (
            f'{work_dir}/epoch_10.pth')

        # by epoch is False
        runner.epoch = 9
        runner.meta = dict()
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=False)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert runner.meta.get('hook_msgs', None) is None

        # max_keep_ckpts > 0
        runner.work_dir = work_dir
        os.system(f'touch {work_dir}/epoch_8.pth')
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, max_keep_ckpts=1)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert (runner.epoch + 1) % 2 == 0
        assert not os.path.exists(f'{work_dir}/epoch_8.pth')

    def test_after_train_iter(self, tmp_path):
        work_dir = str(tmp_path)
        runner = Mock()
        runner.work_dir = str(work_dir)
        runner.iter = 9
        batch_idx = 9
        runner.meta = dict()
        runner.model = Mock()

        # by epoch is True
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_iter(runner, batch_idx=batch_idx)
        assert runner.meta.get('hook_msgs', None) is None

        # by epoch is False
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=False)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_iter(runner, batch_idx=batch_idx)
        assert (runner.iter + 1) % 2 == 0
        assert runner.meta['hook_msgs']['last_ckpt'] == (
            f'{work_dir}/iter_10.pth')

        # epoch can not be evenly divided by 2
        runner.iter = 10
        checkpoint_hook.after_train_epoch(runner)
        assert runner.meta['hook_msgs']['last_ckpt'] == (
            f'{work_dir}/iter_10.pth')

        # max_keep_ckpts > 0
        runner.iter = 9
        runner.work_dir = work_dir
        os.system(f'touch {work_dir}/iter_8.pth')
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=False, max_keep_ckpts=1)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_iter(runner, batch_idx=batch_idx)
        assert not os.path.exists(f'{work_dir}/iter_8.pth')
