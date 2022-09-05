# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from unittest.mock import Mock, patch

import pytest

from mmengine.hooks import CheckpointHook
from mmengine.logging import MessageHub


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
        assert checkpoint_hook.out_dir == osp.join('test_dir', osp.join(osp.basename(work_dir)))

        runner.message_hub = MessageHub.get_instance('test_before_train')
        # no 'best_ckpt_path' in runtime_info
        checkpoint_hook = CheckpointHook(interval=1, save_best=['acc', 'mIoU'])
        checkpoint_hook.before_train(runner)
        assert checkpoint_hook.best_ckpt_path_dict == dict(acc=None, mIoU=None)
        assert not hasattr(checkpoint_hook, 'best_ckpt_path')

        # only one 'best_ckpt_path' in runtime_info
        runner.message_hub.update_info('best_ckpt_acc', 'best_acc')
        checkpoint_hook.before_train(runner)
        assert checkpoint_hook.best_ckpt_path_dict == dict(
            acc='best_acc', mIoU=None)

        # no 'best_ckpt_path' in runtime_info
        checkpoint_hook = CheckpointHook(interval=1, save_best='acc')
        checkpoint_hook.before_train(runner)
        assert checkpoint_hook.best_ckpt_path is None
        assert not hasattr(checkpoint_hook, 'best_ckpt_path_dict')

        # 'best_ckpt_path' in runtime_info
        runner.message_hub.update_info('best_ckpt', 'best_ckpt')
        checkpoint_hook.before_train(runner)
        assert checkpoint_hook.best_ckpt_path == 'best_ckpt'

    def test_after_val_epoch(self, tmp_path):
        runner = Mock()
        runner.work_dir = tmp_path
        runner.epoch = 9
        runner.model = Mock()
        runner.message_hub = MessageHub.get_instance('test_after_val_epoch')

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
            eval_hook._get_metric_score(None, None)
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

        # test error when number of rules and metrics are not same
        with pytest.raises(AssertionError) as assert_error:
            CheckpointHook(
                interval=1,
                save_best=['mIoU', 'acc'],
                rule=['greater', 'greater', 'less'],
                by_epoch=True)
        error_message = ('Number of "rule" must be 1 or the same as number of '
                         '"save_best", but got 3.')
        assert error_message in str(assert_error.value)

        # if save_best is None,no best_ckpt meta should be stored
        eval_hook = CheckpointHook(interval=2, by_epoch=True, save_best=None)
        eval_hook.before_train(runner)
        eval_hook.after_val_epoch(runner, None)
        assert 'best_score' not in runner.message_hub.runtime_info
        assert 'best_ckpt' not in runner.message_hub.runtime_info

        # when `save_best` is set to `auto`, first metric will be used.
        metrics = {'acc': 0.5, 'map': 0.3}
        eval_hook = CheckpointHook(interval=2, by_epoch=True, save_best='auto')
        eval_hook.before_train(runner)
        eval_hook.after_val_epoch(runner, metrics)
        best_ckpt_name = 'best_acc_epoch_9.pth'
        best_ckpt_path = eval_hook.file_client.join_path(
            eval_hook.out_dir, best_ckpt_name)
        assert eval_hook.key_indicators == ['acc']
        assert eval_hook.rules == ['greater']
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.5
        assert 'best_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_ckpt') == best_ckpt_path

        # # when `save_best` is set to `acc`, it should update greater value
        eval_hook = CheckpointHook(interval=2, by_epoch=True, save_best='acc')
        eval_hook.before_train(runner)
        metrics['acc'] = 0.8
        eval_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.8

        # # when `save_best` is set to `loss`, it should update less value
        eval_hook = CheckpointHook(interval=2, by_epoch=True, save_best='loss')
        eval_hook.before_train(runner)
        metrics['loss'] = 0.8
        eval_hook.after_val_epoch(runner, metrics)
        metrics['loss'] = 0.5
        eval_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.5

        # when `rule` is set to `less`,then it should update less value
        # no matter what `save_best` is
        eval_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='acc', rule='less')
        eval_hook.before_train(runner)
        metrics['acc'] = 0.3
        eval_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.3

        # # when `rule` is set to `greater`,then it should update greater value
        # # no matter what `save_best` is
        eval_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='loss', rule='greater')
        eval_hook.before_train(runner)
        metrics['loss'] = 1.0
        eval_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 1.0

        # test multi `save_best` with one rule
        eval_hook = CheckpointHook(
            interval=2, save_best=['acc', 'mIoU'], rule='greater')
        assert eval_hook.key_indicators == ['acc', 'mIoU']
        assert eval_hook.rules == ['greater', 'greater']

        # test multi `save_best` with multi rules
        eval_hook = CheckpointHook(
            interval=2, save_best=['FID', 'IS'], rule=['less', 'greater'])
        assert eval_hook.key_indicators == ['FID', 'IS']
        assert eval_hook.rules == ['less', 'greater']

        # test multi `save_best` with default rule
        eval_hook = CheckpointHook(interval=2, save_best=['acc', 'mIoU'])
        assert eval_hook.key_indicators == ['acc', 'mIoU']
        assert eval_hook.rules == ['greater', 'greater']
        runner.message_hub = MessageHub.get_instance(
            'test_after_val_epoch_save_multi_best')
        eval_hook.before_train(runner)
        metrics = dict(acc=0.5, mIoU=0.6)
        eval_hook.after_val_epoch(runner, metrics)
        best_acc_name = 'best_acc_epoch_9.pth'
        best_acc_path = eval_hook.file_client.join_path(
            eval_hook.out_dir, best_acc_name)
        best_mIoU_name = 'best_mIoU_epoch_9.pth'
        best_mIoU_path = eval_hook.file_client.join_path(
            eval_hook.out_dir, best_mIoU_name)
        assert 'best_score_acc' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score_acc') == 0.5
        assert 'best_score_mIoU' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score_mIoU') == 0.6
        assert 'best_ckpt_acc' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_ckpt_acc') == best_acc_path
        assert 'best_ckpt_mIoU' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_ckpt_mIoU') == best_mIoU_path

        # test behavior when by_epoch is False
        runner = Mock()
        runner.work_dir = tmp_path
        runner.iter = 9
        runner.model = Mock()
        runner.message_hub = MessageHub.get_instance(
            'test_after_val_epoch_by_epoch_is_false')

        # check best ckpt name and best score
        metrics = {'acc': 0.5, 'map': 0.3}
        eval_hook = CheckpointHook(
            interval=2, by_epoch=False, save_best='acc', rule='greater')
        eval_hook.before_train(runner)
        eval_hook.after_val_epoch(runner, metrics)
        assert eval_hook.key_indicators == ['acc']
        assert eval_hook.rules == ['greater']
        best_ckpt_name = 'best_acc_iter_9.pth'
        best_ckpt_path = eval_hook.file_client.join_path(
            eval_hook.out_dir, best_ckpt_name)
        assert 'best_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_ckpt') == best_ckpt_path
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.5

        # check best score updating
        metrics['acc'] = 0.666
        eval_hook.after_val_epoch(runner, metrics)
        best_ckpt_name = 'best_acc_iter_9.pth'
        best_ckpt_path = eval_hook.file_client.join_path(
            eval_hook.out_dir, best_ckpt_name)
        assert 'best_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_ckpt') == best_ckpt_path
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.666
        # error when 'auto' in `save_best` list
        with pytest.raises(AssertionError):
            CheckpointHook(interval=2, save_best=['auto', 'acc'])
        # error when one `save_best` with multi `rule`
        with pytest.raises(AssertionError):
            CheckpointHook(
                interval=2, save_best='acc', rule=['greater', 'less'])

        # check best checkpoint name with `by_epoch` is False
        eval_hook = CheckpointHook(
            interval=2, by_epoch=False, save_best=['acc', 'mIoU'])
        assert eval_hook.key_indicators == ['acc', 'mIoU']
        assert eval_hook.rules == ['greater', 'greater']
        runner.message_hub = MessageHub.get_instance(
            'test_after_val_epoch_save_multi_best_by_epoch_is_false')
        eval_hook.before_train(runner)
        metrics = dict(acc=0.5, mIoU=0.6)
        eval_hook.after_val_epoch(runner, metrics)
        best_acc_name = 'best_acc_iter_9.pth'
        best_acc_path = eval_hook.file_client.join_path(
            eval_hook.out_dir, best_acc_name)
        best_mIoU_name = 'best_mIoU_iter_9.pth'
        best_mIoU_path = eval_hook.file_client.join_path(
            eval_hook.out_dir, best_mIoU_name)
        assert 'best_score_acc' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score_acc') == 0.5
        assert 'best_score_mIoU' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score_mIoU') == 0.6
        assert 'best_ckpt_acc' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_ckpt_acc') == best_acc_path
        assert 'best_ckpt_mIoU' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_ckpt_mIoU') == best_mIoU_path

        # after_val_epoch should not save last_checkpoint.
        assert not osp.isfile(osp.join(runner.work_dir, 'last_checkpoint'))

    def test_after_train_epoch(self, tmp_path):
        runner = Mock()
        work_dir = str(tmp_path)
        runner.work_dir = tmp_path
        runner.epoch = 9
        runner.model = Mock()
        runner.message_hub = MessageHub.get_instance('test_after_train_epoch')

        # by epoch is True
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert (runner.epoch + 1) % 2 == 0
        assert 'last_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('last_ckpt') == osp.join(work_dir, 'epoch_10.pth') 
        last_ckpt_path = osp.join(work_dir, 'last_checkpoint')
        assert osp.isfile(last_ckpt_path)
        with open(last_ckpt_path) as f:
            filepath = f.read()
            assert filepath == osp.join(work_dir, 'epoch_10.pth') 

        # epoch can not be evenly divided by 2
        runner.epoch = 10
        checkpoint_hook.after_train_epoch(runner)
        assert 'last_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('last_ckpt') ==  osp.join(work_dir, 'epoch_10.pth') 

        # by epoch is False
        runner.epoch = 9
        runner.message_hub = MessageHub.get_instance('test_after_train_epoch1')
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=False)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_epoch(runner)
        assert 'last_ckpt' not in runner.message_hub.runtime_info

        # # max_keep_ckpts > 0
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
        runner.model = Mock()
        runner.message_hub = MessageHub.get_instance('test_after_train_iter')

        # by epoch is True
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_iter(runner, batch_idx=batch_idx)
        assert 'last_ckpt' not in runner.message_hub.runtime_info

        # by epoch is False
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=False)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_iter(runner, batch_idx=batch_idx)
        assert (runner.iter + 1) % 2 == 0
        assert 'last_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('last_ckpt') == osp.join(work_dir, 'iter_10.pth') 

        # epoch can not be evenly divided by 2
        runner.iter = 10
        checkpoint_hook.after_train_epoch(runner)
        assert 'last_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('last_ckpt') == osp.join(work_dir, 'iter_10.pth') 

        # max_keep_ckpts > 0
        runner.iter = 9
        runner.work_dir = work_dir
        os.system(f'touch {osp.join(work_dir, "iter_8.pth")}')
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=False, max_keep_ckpts=1)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_iter(runner, batch_idx=batch_idx)
        os.system(f'touch {osp.join(work_dir, "iter_8.pth")}')
