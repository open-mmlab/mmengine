# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.evaluator import BaseMetric
from mmengine.fileio import FileClient, LocalBackend
from mmengine.hooks import CheckpointHook
from mmengine.logging import MessageHub
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, data_sample, mode='tensor'):
        labels = torch.stack(data_sample)
        inputs = torch.stack(inputs)
        outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class TriangleMetric(BaseMetric):

    default_prefix: str = 'test'

    def __init__(self, length):
        super().__init__()
        self.length = length
        self.best_idx = length // 2
        self.cur_idx = 0

    def process(self, *args, **kwargs):
        self.results.append(0)

    def compute_metrics(self, *args, **kwargs):
        self.cur_idx += 1
        acc = 1.0 - abs(self.cur_idx - self.best_idx) / self.length
        return dict(acc=acc)


class TestCheckpointHook:

    def test_init(self, tmp_path):
        # Test file_client_args and backend_args
        with pytest.warns(
                DeprecationWarning,
                match='"file_client_args" will be deprecated in future'):
            CheckpointHook(file_client_args={'backend': 'disk'})

        with pytest.raises(
                ValueError,
                match='"file_client_args" and "backend_args" cannot be set '
                'at the same time'):
            CheckpointHook(
                file_client_args={'backend': 'disk'},
                backend_args={'backend': 'local'})

    def test_before_train(self, tmp_path):
        runner = Mock()
        work_dir = str(tmp_path)
        runner.work_dir = work_dir

        # file_client_args is None
        checkpoint_hook = CheckpointHook()
        checkpoint_hook.before_train(runner)
        assert isinstance(checkpoint_hook.file_client, FileClient)
        assert isinstance(checkpoint_hook.file_backend, LocalBackend)

        # file_client_args is not None
        checkpoint_hook = CheckpointHook(file_client_args={'backend': 'disk'})
        checkpoint_hook.before_train(runner)
        assert isinstance(checkpoint_hook.file_client, FileClient)
        # file_backend is the alias of file_client
        assert checkpoint_hook.file_backend is checkpoint_hook.file_client

        # the out_dir of the checkpoint hook is None
        checkpoint_hook = CheckpointHook(interval=1, by_epoch=True)
        checkpoint_hook.before_train(runner)
        assert checkpoint_hook.out_dir == runner.work_dir

        # the out_dir of the checkpoint hook is not None
        checkpoint_hook = CheckpointHook(
            interval=1, by_epoch=True, out_dir='test_dir')
        checkpoint_hook.before_train(runner)
        assert checkpoint_hook.out_dir == osp.join(
            'test_dir', osp.join(osp.basename(work_dir)))

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
        runner.logger.warning = Mock()
        runner.message_hub = MessageHub.get_instance('test_after_val_epoch')

        with pytest.raises(ValueError):
            # key_indicator must be valid when rule_map is None
            CheckpointHook(interval=2, by_epoch=True, save_best='unsupport')

        with pytest.raises(KeyError):
            # rule must be in keys of rule_map
            CheckpointHook(
                interval=2, by_epoch=True, save_best='auto', rule='unsupport')

        # if metrics is an empty dict, print a warning information
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='auto')
        checkpoint_hook.after_val_epoch(runner, {})
        runner.logger.warning.assert_called_once()

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

        # if save_best is None, no best_ckpt meta should be stored
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best=None)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_val_epoch(runner, {})
        assert 'best_score' not in runner.message_hub.runtime_info
        assert 'best_ckpt' not in runner.message_hub.runtime_info

        # when `save_best` is set to `auto`, first metric will be used.
        metrics = {'acc': 0.5, 'map': 0.3}
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='auto')
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_val_epoch(runner, metrics)
        best_ckpt_name = 'best_acc_epoch_9.pth'
        best_ckpt_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_ckpt_name)
        assert checkpoint_hook.key_indicators == ['acc']
        assert checkpoint_hook.rules == ['greater']
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.5
        assert 'best_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_ckpt') == best_ckpt_path

        # # when `save_best` is set to `acc`, it should update greater value
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='acc')
        checkpoint_hook.before_train(runner)
        metrics['acc'] = 0.8
        checkpoint_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.8

        # # when `save_best` is set to `loss`, it should update less value
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='loss')
        checkpoint_hook.before_train(runner)
        metrics['loss'] = 0.8
        checkpoint_hook.after_val_epoch(runner, metrics)
        metrics['loss'] = 0.5
        checkpoint_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.5

        # when `rule` is set to `less`,then it should update less value
        # no matter what `save_best` is
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='acc', rule='less')
        checkpoint_hook.before_train(runner)
        metrics['acc'] = 0.3
        checkpoint_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.3

        # # when `rule` is set to `greater`,then it should update greater value
        # # no matter what `save_best` is
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='loss', rule='greater')
        checkpoint_hook.before_train(runner)
        metrics['loss'] = 1.0
        checkpoint_hook.after_val_epoch(runner, metrics)
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 1.0

        # test multi `save_best` with one rule
        checkpoint_hook = CheckpointHook(
            interval=2, save_best=['acc', 'mIoU'], rule='greater')
        assert checkpoint_hook.key_indicators == ['acc', 'mIoU']
        assert checkpoint_hook.rules == ['greater', 'greater']

        # test multi `save_best` with multi rules
        checkpoint_hook = CheckpointHook(
            interval=2, save_best=['FID', 'IS'], rule=['less', 'greater'])
        assert checkpoint_hook.key_indicators == ['FID', 'IS']
        assert checkpoint_hook.rules == ['less', 'greater']

        # test multi `save_best` with default rule
        checkpoint_hook = CheckpointHook(interval=2, save_best=['acc', 'mIoU'])
        assert checkpoint_hook.key_indicators == ['acc', 'mIoU']
        assert checkpoint_hook.rules == ['greater', 'greater']
        runner.message_hub = MessageHub.get_instance(
            'test_after_val_epoch_save_multi_best')
        checkpoint_hook.before_train(runner)
        metrics = dict(acc=0.5, mIoU=0.6)
        checkpoint_hook.after_val_epoch(runner, metrics)
        best_acc_name = 'best_acc_epoch_9.pth'
        best_acc_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_acc_name)
        best_mIoU_name = 'best_mIoU_epoch_9.pth'
        best_mIoU_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_mIoU_name)
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
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=False, save_best='acc', rule='greater')
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_val_epoch(runner, metrics)
        assert checkpoint_hook.key_indicators == ['acc']
        assert checkpoint_hook.rules == ['greater']
        best_ckpt_name = 'best_acc_iter_9.pth'
        best_ckpt_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_ckpt_name)
        assert 'best_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_ckpt') == best_ckpt_path
        assert 'best_score' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('best_score') == 0.5

        # check best score updating
        metrics['acc'] = 0.666
        checkpoint_hook.after_val_epoch(runner, metrics)
        best_ckpt_name = 'best_acc_iter_9.pth'
        best_ckpt_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_ckpt_name)
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
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=False, save_best=['acc', 'mIoU'])
        assert checkpoint_hook.key_indicators == ['acc', 'mIoU']
        assert checkpoint_hook.rules == ['greater', 'greater']
        runner.message_hub = MessageHub.get_instance(
            'test_after_val_epoch_save_multi_best_by_epoch_is_false')
        checkpoint_hook.before_train(runner)
        metrics = dict(acc=0.5, mIoU=0.6)
        checkpoint_hook.after_val_epoch(runner, metrics)
        best_acc_name = 'best_acc_iter_9.pth'
        best_acc_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_acc_name)
        best_mIoU_name = 'best_mIoU_iter_9.pth'
        best_mIoU_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_mIoU_name)
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
            runner.message_hub.get_info('last_ckpt') == \
               osp.join(work_dir, 'epoch_10.pth')
        last_ckpt_path = osp.join(work_dir, 'last_checkpoint')
        assert osp.isfile(last_ckpt_path)
        with open(last_ckpt_path) as f:
            filepath = f.read()
            assert filepath == osp.join(work_dir, 'epoch_10.pth')

        # epoch can not be evenly divided by 2
        runner.epoch = 10
        checkpoint_hook.after_train_epoch(runner)
        assert 'last_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('last_ckpt') == \
               osp.join(work_dir, 'epoch_10.pth')

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

        # save_checkpoint of runner should be called with expected arguments
        runner = Mock()
        work_dir = str(tmp_path)
        runner.work_dir = tmp_path
        runner.epoch = 1
        runner.message_hub = MessageHub.get_instance('test_after_train_epoch2')

        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_epoch(runner)

        runner.save_checkpoint.assert_called_once_with(
            runner.work_dir,
            'epoch_2.pth',
            None,
            backend_args=None,
            by_epoch=True,
            save_optimizer=True,
            save_param_scheduler=True)

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
            runner.message_hub.get_info('last_ckpt') == \
               osp.join(work_dir, 'iter_10.pth')

        # epoch can not be evenly divided by 2
        runner.iter = 10
        checkpoint_hook.after_train_epoch(runner)
        assert 'last_ckpt' in runner.message_hub.runtime_info and \
            runner.message_hub.get_info('last_ckpt') == \
               osp.join(work_dir, 'iter_10.pth')

        # max_keep_ckpts > 0
        runner.iter = 9
        runner.work_dir = work_dir
        os.system(f'touch {osp.join(work_dir, "iter_8.pth")}')
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=False, max_keep_ckpts=1)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_iter(runner, batch_idx=batch_idx)
        assert not os.path.exists(f'{work_dir}/iter_8.pth')

    def test_with_runner(self, tmp_path):
        max_epoch = 10
        work_dir = osp.join(str(tmp_path), 'runner_test')
        tmpl = '{}.pth'
        save_interval = 2
        checkpoint_cfg = dict(
            type='CheckpointHook',
            interval=save_interval,
            filename_tmpl=tmpl,
            by_epoch=True)
        runner = Runner(
            model=ToyModel(),
            work_dir=work_dir,
            train_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=DummyDataset(),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=dict(type=TriangleMetric, length=max_epoch),
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(
                by_epoch=True, max_epochs=max_epoch, val_interval=1),
            val_cfg=dict(),
            default_hooks=dict(checkpoint=checkpoint_cfg))
        runner.train()
        for epoch in range(max_epoch):
            if epoch % save_interval != 0 or epoch == 0:
                continue
            path = osp.join(work_dir, tmpl.format(epoch))
            assert osp.isfile(path=path)
