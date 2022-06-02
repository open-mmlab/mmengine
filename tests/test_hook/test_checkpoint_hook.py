# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from mmengine.hooks import CheckpointHook
from mmengine.runner.runner import Runner


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


class MockModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, **kwargs):
        return self.param * x

    def train_step(self, data_batch, optimizer, **kwargs):
        return {'loss': torch.sum(self(data_batch['x']))}

    def val_step(self, data_batch, optimizer, **kwargs):
        return {'loss': torch.sum(self(data_batch['x']))}


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

        # create_symlink in args and create_symlink is True
        checkpoint_hook = CheckpointHook(
            interval=1, by_epoch=True, out_dir='test_dir', create_symlink=True)
        checkpoint_hook.before_train(runner)
        assert checkpoint_hook.args['create_symlink']

        runner.work_dir = 's3://path/of/file'
        checkpoint_hook = CheckpointHook(
            interval=1, by_epoch=True, create_symlink=True)
        checkpoint_hook.before_train(runner)
        assert not checkpoint_hook.args['create_symlink']

    def test_after_val_epoch(self, tmp_path):
        if not hasattr(self, 'runner'):
            mock_model = MockModel()
            self.runner = Runner(model=mock_model, work_dir=str(tmp_path))
        runner = self.runner
        # save_best is acc
        with pytest.raises(ValueError):
            # key_indicator must be valid when rule_map is None
            CheckpointHook(interval=2, by_epoch=True, save_best='unsupport')

        with pytest.raises(KeyError):
            # rule must be in keys of rule_map
            CheckpointHook(
                interval=2, by_epoch=True, save_best='auto', rule='unsupport')

        # if eval_res is an empty dict, print a warning information
        with pytest.warns(UserWarning) as record_warnings:
            eval_hook = CheckpointHook(
                interval=2, by_epoch=True, save_best='auto')
            eval_hook._get_metric_score(None)
        # Since there will be many warnings thrown, we just need to check
        # if the expected exceptions are thrown
        expected_message = (
            'Since `eval_res` is an empty dict, the behavior to '
            'save the best checkpoint will be skipped in this '
            'evaluation.')
        for warning in record_warnings:
            if str(warning.message) == expected_message:
                break
        else:
            assert False

        # if save_best is None,no best_ckpt meta should be stored
        optimizer = torch.optim.SGD(
            mock_model.parameters(), lr=0.001, momentum=0.9)
        runner.optimizer = optimizer
        eval_hook = CheckpointHook(interval=2, by_epoch=True, save_best=None)
        eval_hook.before_train(runner)
        eval_hook.after_val_epoch(runner, None)
        assert runner.meta is None or 'best_score' not in runner.meta[
            'hook_msgs']
        assert runner.meta is None or 'best_ckpt' not in runner.meta[
            'hook_msgs']

        # when `save_best` is set to `auto`, first metric will be used.
        metrics = {'acc': 0.5, 'map': 0.3}
        eval_hook = CheckpointHook(interval=2, by_epoch=True, save_best='auto')
        eval_hook.before_train(runner)
        eval_hook.after_val_epoch(runner, metrics)
        best_ckpt_name = 'best_acc_epoch_1.pth'
        best_ckpt_path = eval_hook.file_client.join_path(
            eval_hook.out_dir, best_ckpt_name)
        assert eval_hook.key_indicator == 'acc'
        assert eval_hook.rule == 'greater'
        assert 'best_score' in runner.meta['hook_msgs'] and \
            runner.meta['hook_msgs']['best_score'] == 0.5
        assert 'best_ckpt' in runner.meta['hook_msgs'] and \
            runner.meta['hook_msgs']['best_ckpt'] == best_ckpt_path

        # when `save_best` is set to `acc`, it should update greater value
        eval_hook = CheckpointHook(interval=2, by_epoch=True, save_best='acc')
        eval_hook.before_train(runner)
        metrics['acc'] = 0.8
        eval_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.meta['hook_msgs'] and \
            runner.meta['hook_msgs']['best_score'] == 0.8

        # when `save_best` is set to `loss`, it should update less value
        eval_hook = CheckpointHook(interval=2, by_epoch=True, save_best='loss')
        eval_hook.before_train(runner)
        metrics['loss'] = 0.8
        eval_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.meta['hook_msgs'] and \
            runner.meta['hook_msgs']['best_score'] == 0.8
        metrics['loss'] = 0.5
        eval_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.meta['hook_msgs'] and \
            runner.meta['hook_msgs']['best_score'] == 0.5

        # when `rule` is set to `less`,then it should update less value
        # no matter what `save_best` is
        eval_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='acc', rule='less')
        eval_hook.before_train(runner)
        metrics['acc'] = 0.3
        eval_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.meta['hook_msgs'] and \
            runner.meta['hook_msgs']['best_score'] == 0.3

        # when `rule` is set to `greater`,then it should update greater value
        # no matter what `save_best` is
        eval_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='loss', rule='greater')
        eval_hook.before_train(runner)
        metrics['loss'] = 1.0
        eval_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.meta['hook_msgs'] and \
            runner.meta['hook_msgs']['best_score'] == 1.0

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
