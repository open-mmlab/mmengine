# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import re
import sys
from unittest.mock import MagicMock, patch

import torch
from parameterized import parameterized

from mmengine.evaluator import BaseMetric
from mmengine.fileio import FileClient, LocalBackend
from mmengine.hooks import CheckpointHook
from mmengine.logging import MessageHub
from mmengine.registry import METRICS
from mmengine.testing import RunnerTestCase


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


class TestCheckpointHook(RunnerTestCase):

    def setUp(self):
        super().setUp()
        METRICS.register_module(module=TriangleMetric, force=True)

    def tearDown(self):
        return METRICS.module_dict.clear()

    def test_init(self):
        # Test file_client_args and backend_args
        # TODO: Refactor this test case
        # with self.assertWarnsRegex(
        #         DeprecationWarning,
        #         '"file_client_args" will be deprecated in future'):
        #     CheckpointHook(file_client_args={'backend': 'disk'})

        with self.assertRaisesRegex(
                ValueError,
                '"file_client_args" and "backend_args" cannot be set '
                'at the same time'):
            CheckpointHook(
                file_client_args={'backend': 'disk'},
                backend_args={'backend': 'local'})

        # Test save best
        CheckpointHook(save_best='acc')
        CheckpointHook(save_best=['acc'])

        with self.assertRaisesRegex(AssertionError, '"save_best" should be'):
            CheckpointHook(save_best=dict(acc='acc'))

        # error when 'auto' in `save_best` list
        with self.assertRaisesRegex(AssertionError, 'Only support one'):
            CheckpointHook(interval=2, save_best=['auto', 'acc'])

        # Test rules
        CheckpointHook(save_best=['acc', 'mAcc'], rule='greater')
        with self.assertRaisesRegex(AssertionError, '"rule" should be a str'):
            CheckpointHook(save_best=['acc'], rule=1)

        with self.assertRaisesRegex(AssertionError,
                                    'Number of "rule" must be'):
            CheckpointHook(save_best=['acc'], rule=['greater', 'loss'])

        # Test greater_keys
        hook = CheckpointHook(greater_keys='acc')
        self.assertEqual(hook.greater_keys, ('acc', ))

        hook = CheckpointHook(greater_keys=['acc'])
        self.assertEqual(hook.greater_keys, ['acc'])

        hook = CheckpointHook(
            interval=2, by_epoch=False, save_best=['acc', 'mIoU'])
        self.assertEqual(hook.key_indicators, ['acc', 'mIoU'])
        self.assertEqual(hook.rules, ['greater', 'greater'])

        # Test less keys
        hook = CheckpointHook(less_keys='loss_cls')
        self.assertEqual(hook.less_keys, ('loss_cls', ))

        hook = CheckpointHook(less_keys=['loss_cls'])
        self.assertEqual(hook.less_keys, ['loss_cls'])

    def test_before_train(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        # file_client_args is None
        checkpoint_hook = CheckpointHook()
        checkpoint_hook.before_train(runner)
        self.assertIsInstance(checkpoint_hook.file_client, FileClient)
        self.assertIsInstance(checkpoint_hook.file_backend, LocalBackend)

        # file_client_args is not None
        checkpoint_hook = CheckpointHook(file_client_args={'backend': 'disk'})
        checkpoint_hook.before_train(runner)
        self.assertIsInstance(checkpoint_hook.file_client, FileClient)
        # file_backend is the alias of file_client
        self.assertIs(checkpoint_hook.file_backend,
                      checkpoint_hook.file_client)

        # the out_dir of the checkpoint hook is None
        checkpoint_hook = CheckpointHook(interval=1, by_epoch=True)
        checkpoint_hook.before_train(runner)
        self.assertEqual(checkpoint_hook.out_dir, runner.work_dir)

        # the out_dir of the checkpoint hook is not None
        checkpoint_hook = CheckpointHook(
            interval=1, by_epoch=True, out_dir='test_dir')
        checkpoint_hook.before_train(runner)
        self.assertEqual(checkpoint_hook.out_dir,
                         osp.join('test_dir', osp.basename(cfg.work_dir)))

        # If `save_best` is a list of string, the path to save the best
        # checkpoint will be defined in attribute `best_ckpt_path_dict`.
        checkpoint_hook = CheckpointHook(interval=1, save_best=['acc', 'mIoU'])
        checkpoint_hook.before_train(runner)
        self.assertEqual(checkpoint_hook.best_ckpt_path_dict,
                         dict(acc=None, mIoU=None))
        self.assertFalse(hasattr(checkpoint_hook, 'best_ckpt_path'))

        # Resume 'best_ckpt_path' from message_hub
        runner.message_hub.update_info('best_ckpt_acc', 'best_acc')
        checkpoint_hook.before_train(runner)
        self.assertEqual(checkpoint_hook.best_ckpt_path_dict,
                         dict(acc='best_acc', mIoU=None))

        # If `save_best` is a string, the path to save best ckpt will be
        # defined in attribute `best_ckpt_path`
        checkpoint_hook = CheckpointHook(interval=1, save_best='acc')
        checkpoint_hook.before_train(runner)
        self.assertIsNone(checkpoint_hook.best_ckpt_path)
        self.assertFalse(hasattr(checkpoint_hook, 'best_ckpt_path_dict'))

        # Resume `best_ckpt` path from message_hub
        runner.message_hub.update_info('best_ckpt', 'best_ckpt')
        checkpoint_hook.before_train(runner)
        self.assertEqual(checkpoint_hook.best_ckpt_path, 'best_ckpt')

    def test_after_val_epoch(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        runner.train_loop._epoch = 9

        # if metrics is an empty dict, print a warning information
        with self.assertLogs(runner.logger, level='WARNING'):
            checkpoint_hook = CheckpointHook(
                interval=2, by_epoch=True, save_best='auto')
            checkpoint_hook.after_val_epoch(runner, {})

        # if save_best is None,no best_ckpt meta should be stored
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best=None)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_val_epoch(runner, {})
        self.assertNotIn('best_score', runner.message_hub.runtime_info)
        self.assertNotIn('best_ckpt', runner.message_hub.runtime_info)

        # when `save_best` is set to `auto`, first metric will be used.
        metrics = {'acc': 0.5, 'map': 0.3}
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='auto')
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_val_epoch(runner, metrics)
        best_ckpt_name = 'best_acc_epoch_9.pth'
        best_ckpt_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_ckpt_name)
        self.assertEqual(checkpoint_hook.key_indicators, ['acc'])
        self.assertEqual(checkpoint_hook.rules, ['greater'])
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.5)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt'), best_ckpt_path)

        # # when `save_best` is set to `acc`, it should update greater value
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='acc')
        checkpoint_hook.before_train(runner)
        metrics['acc'] = 0.8
        checkpoint_hook.after_val_epoch(runner, metrics)
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.8)

        # # when `save_best` is set to `loss`, it should update less value
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='loss')
        checkpoint_hook.before_train(runner)
        metrics['loss'] = 0.8
        checkpoint_hook.after_val_epoch(runner, metrics)
        metrics['loss'] = 0.5
        checkpoint_hook.after_val_epoch(runner, metrics)
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.5)

        # when `rule` is set to `less`,then it should update less value
        # no matter what `save_best` is
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='acc', rule='less')
        checkpoint_hook.before_train(runner)
        metrics['acc'] = 0.3
        checkpoint_hook.after_val_epoch(runner, metrics)
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.3)

        # # when `rule` is set to `greater`,then it should update greater value
        # # no matter what `save_best` is
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='loss', rule='greater')
        checkpoint_hook.before_train(runner)
        metrics['loss'] = 1.0
        checkpoint_hook.after_val_epoch(runner, metrics)
        self.assertEqual(runner.message_hub.get_info('best_score'), 1.0)

        # test multi `save_best` with one rule
        checkpoint_hook = CheckpointHook(
            interval=2, save_best=['acc', 'mIoU'], rule='greater')
        self.assertEqual(checkpoint_hook.key_indicators, ['acc', 'mIoU'])
        self.assertEqual(checkpoint_hook.rules, ['greater', 'greater'])

        # test multi `save_best` with multi rules
        checkpoint_hook = CheckpointHook(
            interval=2, save_best=['FID', 'IS'], rule=['less', 'greater'])
        self.assertEqual(checkpoint_hook.key_indicators, ['FID', 'IS'])
        self.assertEqual(checkpoint_hook.rules, ['less', 'greater'])

        # test multi `save_best` with default rule
        checkpoint_hook = CheckpointHook(interval=2, save_best=['acc', 'mIoU'])
        self.assertEqual(checkpoint_hook.key_indicators, ['acc', 'mIoU'])
        self.assertEqual(checkpoint_hook.rules, ['greater', 'greater'])
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
        self.assertEqual(runner.message_hub.get_info('best_score_acc'), 0.5)
        self.assertEqual(runner.message_hub.get_info('best_score_mIoU'), 0.6)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt_acc'), best_acc_path)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt_mIoU'), best_mIoU_path)

        # test behavior when by_epoch is False
        cfg = copy.deepcopy(self.iter_based_cfg)
        runner = self.build_runner(cfg)
        runner.train_loop._iter = 9

        # check best ckpt name and best score
        metrics = {'acc': 0.5, 'map': 0.3}
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=False, save_best='acc', rule='greater')
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_val_epoch(runner, metrics)
        self.assertEqual(checkpoint_hook.key_indicators, ['acc'])
        self.assertEqual(checkpoint_hook.rules, ['greater'])
        best_ckpt_name = 'best_acc_iter_9.pth'
        best_ckpt_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_ckpt_name)

        self.assertEqual(
            runner.message_hub.get_info('best_ckpt'), best_ckpt_path)
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.5)

        # check best score updating
        metrics['acc'] = 0.666
        checkpoint_hook.after_val_epoch(runner, metrics)
        best_ckpt_name = 'best_acc_iter_9.pth'
        best_ckpt_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_ckpt_name)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt'), best_ckpt_path)
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.666)

        # check best checkpoint name with `by_epoch` is False
        checkpoint_hook = CheckpointHook(
            interval=2, by_epoch=False, save_best=['acc', 'mIoU'])
        checkpoint_hook.before_train(runner)
        metrics = dict(acc=0.5, mIoU=0.6)
        checkpoint_hook.after_val_epoch(runner, metrics)
        best_acc_name = 'best_acc_iter_9.pth'
        best_acc_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_acc_name)
        best_mIoU_name = 'best_mIoU_iter_9.pth'
        best_mIoU_path = checkpoint_hook.file_client.join_path(
            checkpoint_hook.out_dir, best_mIoU_name)

        self.assertEqual(runner.message_hub.get_info('best_score_acc'), 0.5)
        self.assertEqual(runner.message_hub.get_info('best_score_mIoU'), 0.6)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt_acc'), best_acc_path)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt_mIoU'), best_mIoU_path)

        # after_val_epoch should not save last_checkpoint
        self.assertFalse(
            osp.isfile(osp.join(runner.work_dir, 'last_checkpoint')))

        # There should only one best checkpoint be reserved
        # dist backend
        for by_epoch, cfg in [(True, self.epoch_based_cfg),
                              (False, self.iter_based_cfg)]:
            self.clear_work_dir()
            cfg = copy.deepcopy(cfg)
            runner = self.build_runner(cfg)
            checkpoint_hook = CheckpointHook(
                interval=2, by_epoch=by_epoch, save_best='acc')
            checkpoint_hook.before_train(runner)
            checkpoint_hook.after_val_epoch(runner, metrics)
            all_files = os.listdir(runner.work_dir)
            best_ckpts = [
                file for file in all_files if file.startswith('best')
            ]
            self.assertTrue(len(best_ckpts) == 1)

        # petrel backend
        # TODO use real petrel oss bucket to test
        petrel_client = MagicMock()
        for by_epoch, cfg in [(True, self.epoch_based_cfg),
                              (False, self.iter_based_cfg)]:
            isfile = MagicMock(return_value=True)
            self.clear_work_dir()
            with patch.dict(sys.modules, {'petrel_client': petrel_client}), \
                 patch('mmengine.fileio.backends.PetrelBackend.put') as put_mock, \
                 patch('mmengine.fileio.backends.PetrelBackend.remove') as remove_mock, \
                 patch('mmengine.fileio.backends.PetrelBackend.isfile') as isfile:  # noqa: E501
                cfg = copy.deepcopy(cfg)
                runner = self.build_runner(cfg)
                metrics = dict(acc=0.5)
                petrel_client.client.Client = MagicMock(
                    return_value=petrel_client)
                checkpoint_hook = CheckpointHook(
                    interval=2,
                    by_epoch=by_epoch,
                    save_best='acc',
                    backend_args=dict(backend='petrel'))
                checkpoint_hook.before_train(runner)
                checkpoint_hook.after_val_epoch(runner, metrics)
                put_mock.assert_called_once()
                metrics['acc'] += 0.1
                runner.train_loop._epoch += 1
                runner.train_loop._iter += 1
                checkpoint_hook.after_val_epoch(runner, metrics)
                isfile.assert_called_once()
                remove_mock.assert_called_once()

    def test_after_train_epoch(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        runner.train_loop._epoch = 9
        runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)

        # by epoch is True
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_epoch(runner)
        self.assertEqual((runner.epoch + 1) % 2, 0)
        self.assertEqual(
            runner.message_hub.get_info('last_ckpt'),
            osp.join(cfg.work_dir, 'epoch_10.pth'))

        last_ckpt_path = osp.join(cfg.work_dir, 'last_checkpoint')
        self.assertTrue(osp.isfile(last_ckpt_path))

        with open(last_ckpt_path) as f:
            filepath = f.read()
            self.assertEqual(filepath, osp.join(cfg.work_dir, 'epoch_10.pth'))

        # epoch can not be evenly divided by 2
        runner.train_loop._epoch = 10
        checkpoint_hook.after_train_epoch(runner)
        self.assertEqual(
            runner.message_hub.get_info('last_ckpt'),
            osp.join(cfg.work_dir, 'epoch_10.pth'))
        runner.message_hub.runtime_info.clear()

        # by epoch is False
        runner.train_loop._epoch = 9
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=False)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_epoch(runner)
        self.assertNotIn('last_ckpt', runner.message_hub.runtime_info)
        runner.message_hub.runtime_info.clear()

    def test_after_train_iter(self):
        # by epoch is True
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        runner.train_loop._iter = 9
        runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=True)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_iter(runner, batch_idx=9)
        self.assertNotIn('last_ckpt', runner.message_hub.runtime_info)

        # by epoch is False
        checkpoint_hook = CheckpointHook(interval=2, by_epoch=False)
        checkpoint_hook.before_train(runner)
        checkpoint_hook.after_train_iter(runner, batch_idx=9)
        self.assertIn('last_ckpt', runner.message_hub.runtime_info)
        self.assertEqual(
            runner.message_hub.get_info('last_ckpt'),
            osp.join(cfg.work_dir, 'iter_10.pth'))

        # epoch can not be evenly divided by 2
        runner.train_loop._iter = 10
        checkpoint_hook.after_train_epoch(runner)
        self.assertEqual(
            runner.message_hub.get_info('last_ckpt'),
            osp.join(cfg.work_dir, 'iter_10.pth'))

    @parameterized.expand([['iter'], ['epoch']])
    def test_with_runner(self, training_type):
        common_cfg = getattr(self, f'{training_type}_based_cfg')
        setattr(common_cfg.train_cfg, f'max_{training_type}s', 11)
        checkpoint_cfg = dict(
            type='CheckpointHook',
            interval=1,
            by_epoch=training_type == 'epoch')
        common_cfg.default_hooks = dict(checkpoint=checkpoint_cfg)

        # Test interval in epoch based training
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.interval = 2
        runner = self.build_runner(cfg)
        runner.train()

        for i in range(1, 11):
            self.assertEqual(
                osp.isfile(osp.join(cfg.work_dir, f'{training_type}_{i}.pth')),
                i % 2 == 0)

        # save_last=True
        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_11.pth')))

        self.clear_work_dir()

        # Test save_optimizer=False
        cfg = copy.deepcopy(common_cfg)
        runner = self.build_runner(cfg)
        runner.train()
        ckpt = torch.load(osp.join(cfg.work_dir, f'{training_type}_11.pth'))
        self.assertIn('optimizer', ckpt)

        cfg.default_hooks.checkpoint.save_optimizer = False
        runner = self.build_runner(cfg)
        runner.train()
        ckpt = torch.load(osp.join(cfg.work_dir, f'{training_type}_11.pth'))
        self.assertNotIn('optimizer', ckpt)

        # Test save_param_scheduler=False
        cfg = copy.deepcopy(common_cfg)
        cfg.param_scheduler = [
            dict(
                type='LinearLR',
                start_factor=0.1,
                begin=0,
                end=500,
                by_epoch=training_type == 'epoch')
        ]
        runner = self.build_runner(cfg)
        runner.train()
        ckpt = torch.load(osp.join(cfg.work_dir, f'{training_type}_11.pth'))
        self.assertIn('param_schedulers', ckpt)

        cfg.default_hooks.checkpoint.save_param_scheduler = False
        runner = self.build_runner(cfg)
        runner.train()
        ckpt = torch.load(osp.join(cfg.work_dir, f'{training_type}_11.pth'))
        self.assertNotIn('param_schedulers', ckpt)

        self.clear_work_dir()

        # Test out_dir
        cfg = copy.deepcopy(common_cfg)
        out_dir = osp.join(self.temp_dir.name, 'out_dir')
        cfg.default_hooks.checkpoint.out_dir = out_dir
        runner = self.build_runner(cfg)
        runner.train()
        self.assertTrue(
            osp.isfile(
                osp.join(out_dir, osp.basename(cfg.work_dir),
                         f'{training_type}_11.pth')))

        self.clear_work_dir()

        # Test max_keep_ckpts=1
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.max_keep_ckpts = 1
        runner = self.build_runner(cfg)
        runner.train()
        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_11.pth')))

        for i in range(11):
            self.assertFalse(
                osp.isfile(osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))

        self.clear_work_dir()

        # Test max_keep_ckpts=3
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.max_keep_ckpts = 3
        runner = self.build_runner(cfg)
        runner.train()
        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_9.pth')))
        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_10.pth')))
        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_11.pth')))

        for i in range(9):
            self.assertFalse(
                osp.isfile(osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))

        ckpt = torch.load(osp.join(cfg.work_dir, f'{training_type}_11.pth'))
        self.assertEqual(ckpt['message_hub']['runtime_info']['keep_ckpt_ids'],
                         [9, 10, 11])

        # Test max_keep_ckpts when resuming traing
        cfg = copy.deepcopy(common_cfg)
        setattr(cfg.train_cfg, f'max_{training_type}s', 12)
        cfg.default_hooks.checkpoint.max_keep_ckpts = 2
        cfg.load_from = osp.join(cfg.work_dir, f'{training_type}_11.pth')
        cfg.resume = True
        runner = self.build_runner(cfg)
        runner.train()
        self.assertFalse(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_9.pth')))
        self.assertFalse(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_10.pth')))
        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_11.pth')))
        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_12.pth')))

        self.clear_work_dir()

        # Test filename_tmpl
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.filename_tmpl = 'test_{}.pth'
        runner = self.build_runner(cfg)
        runner.train()
        self.assertTrue(osp.isfile(osp.join(cfg.work_dir, 'test_11.pth')))

        self.clear_work_dir()

        # Test save_best
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.save_best = 'test/acc'
        cfg.val_evaluator = dict(type='TriangleMetric', length=11)
        cfg.train_cfg.val_interval = 1
        runner = self.build_runner(cfg)
        runner.train()
        best_ckpt_path = osp.join(cfg.work_dir,
                                  f'best_test_acc_{training_type}_5.pth')
        best_ckpt = torch.load(best_ckpt_path)

        ckpt = torch.load(osp.join(cfg.work_dir, f'{training_type}_5.pth'))
        self.assertEqual(best_ckpt_path,
                         ckpt['message_hub']['runtime_info']['best_ckpt'])

        if training_type == 'epoch':
            self.assertEqual(ckpt['meta']['epoch'], 5)
            self.assertEqual(ckpt['meta']['iter'], 20)
            self.assertEqual(best_ckpt['meta']['epoch'], 5)
            self.assertEqual(best_ckpt['meta']['iter'], 20)
        else:
            self.assertEqual(ckpt['meta']['epoch'], 0)
            self.assertEqual(ckpt['meta']['iter'], 5)
            self.assertEqual(best_ckpt['meta']['epoch'], 0)
            self.assertEqual(best_ckpt['meta']['iter'], 5)

        self.clear_work_dir()

        # Test save_best with interval=2
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.save_best = 'test/acc'
        cfg.default_hooks.checkpoint.interval = 2
        cfg.val_evaluator = dict(type='TriangleMetric', length=11)
        cfg.train_cfg.val_interval = 1
        runner = self.build_runner(cfg)
        runner.train()
        best_ckpt_path = osp.join(cfg.work_dir,
                                  f'best_test_acc_{training_type}_5.pth')
        best_ckpt = torch.load(best_ckpt_path)

        # if the current ckpt is the best, the interval will be ignored the
        # the ckpt will also be saved
        ckpt = torch.load(osp.join(cfg.work_dir, f'{training_type}_5.pth'))
        self.assertEqual(best_ckpt_path,
                         ckpt['message_hub']['runtime_info']['best_ckpt'])

        if training_type == 'epoch':
            self.assertEqual(ckpt['meta']['epoch'], 5)
            self.assertEqual(ckpt['meta']['iter'], 20)
            self.assertEqual(best_ckpt['meta']['epoch'], 5)
            self.assertEqual(best_ckpt['meta']['iter'], 20)
        else:
            self.assertEqual(ckpt['meta']['epoch'], 0)
            self.assertEqual(ckpt['meta']['iter'], 5)
            self.assertEqual(best_ckpt['meta']['epoch'], 0)
            self.assertEqual(best_ckpt['meta']['iter'], 5)

        # Test save published keys
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.published_keys = ['meta', 'state_dict']
        runner = self.build_runner(cfg)
        runner.train()
        ckpt_files = os.listdir(runner.work_dir)
        self.assertTrue(
            any(re.findall(r'-[\d\w]{8}\.pth', file) for file in ckpt_files))

        self.clear_work_dir()

        # Test save_begin with interval=2, save_begin=5
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.interval = 2
        cfg.default_hooks.checkpoint.save_begin = 5
        runner = self.build_runner(cfg)
        runner.train()

        for i in range(5):
            self.assertFalse(
                osp.isfile(osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))
        for i in range(5, 11):
            if (i - 5) % 2 == 1:
                self.assertFalse(
                    osp.isfile(
                        osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))
            else:
                self.assertTrue(
                    osp.isfile(
                        osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))
        self.clear_work_dir()

        # Test save_begin with interval=2, save_begin=0
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.interval = 2
        runner = self.build_runner(cfg)
        runner.train()

        for i in range(1, 11):
            if i % 2 == 1:
                self.assertFalse(
                    osp.isfile(
                        osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))
            else:
                self.assertTrue(
                    osp.isfile(
                        osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))
        self.clear_work_dir()

        # Test save_begin with interval=2, save_begin=1
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.checkpoint.interval = 2
        cfg.default_hooks.checkpoint.save_begin = 1
        runner = self.build_runner(cfg)
        runner.train()

        for i in range(1, 11):
            if i % 2 == 1:
                self.assertTrue(
                    osp.isfile(
                        osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))
            else:
                self.assertFalse(
                    osp.isfile(
                        osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))
        self.clear_work_dir()
