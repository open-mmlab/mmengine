# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

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
        with self.assertWarnsRegex(
                DeprecationWarning,
                '"file_client_args" will be deprecated in future'):
            CheckpointHook(file_client_args={'backend': 'disk'})

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
        self.assertEqual(
            checkpoint_hook.out_dir,
            osp.join('test_dir', osp.join(osp.basename(cfg.work_dir))))

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

        # if save_best is None,no best_ckpt meta should be stored
        ckpt_hook = CheckpointHook(interval=2, by_epoch=True, save_best=None)
        ckpt_hook.before_train(runner)
        ckpt_hook.after_val_epoch(runner, None)
        self.assertNotIn('best_score', runner.message_hub.runtime_info)
        self.assertNotIn('best_ckpt', runner.message_hub.runtime_info)

        # when `save_best` is set to `auto`, first metric will be used.
        metrics = {'acc': 0.5, 'map': 0.3}
        ckpt_hook = CheckpointHook(interval=2, by_epoch=True, save_best='auto')
        ckpt_hook.before_train(runner)
        ckpt_hook.after_val_epoch(runner, metrics)
        best_ckpt_name = 'best_acc_epoch_9.pth'
        best_ckpt_path = ckpt_hook.file_client.join_path(
            ckpt_hook.out_dir, best_ckpt_name)
        self.assertEqual(ckpt_hook.key_indicators, ['acc'])
        self.assertEqual(ckpt_hook.rules, ['greater'])
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.5)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt'), best_ckpt_path)

        # # when `save_best` is set to `acc`, it should update greater value
        ckpt_hook = CheckpointHook(interval=2, by_epoch=True, save_best='acc')
        ckpt_hook.before_train(runner)
        metrics['acc'] = 0.8
        ckpt_hook.after_val_epoch(runner, metrics)
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.8)

        # # when `save_best` is set to `loss`, it should update less value
        ckpt_hook = CheckpointHook(interval=2, by_epoch=True, save_best='loss')
        ckpt_hook.before_train(runner)
        metrics['loss'] = 0.8
        ckpt_hook.after_val_epoch(runner, metrics)
        metrics['loss'] = 0.5
        ckpt_hook.after_val_epoch(runner, metrics)
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.5)

        # when `rule` is set to `less`,then it should update less value
        # no matter what `save_best` is
        ckpt_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='acc', rule='less')
        ckpt_hook.before_train(runner)
        metrics['acc'] = 0.3
        ckpt_hook.after_val_epoch(runner, metrics)
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.3)

        # # when `rule` is set to `greater`,then it should update greater value
        # # no matter what `save_best` is
        ckpt_hook = CheckpointHook(
            interval=2, by_epoch=True, save_best='loss', rule='greater')
        ckpt_hook.before_train(runner)
        metrics['loss'] = 1.0
        ckpt_hook.after_val_epoch(runner, metrics)
        self.assertEqual(runner.message_hub.get_info('best_score'), 1.0)

        # test multi `save_best` with one rule
        ckpt_hook = CheckpointHook(
            interval=2, save_best=['acc', 'mIoU'], rule='greater')
        self.assertEqual(ckpt_hook.key_indicators, ['acc', 'mIoU'])
        self.assertEqual(ckpt_hook.rules, ['greater', 'greater'])

        # test multi `save_best` with multi rules
        ckpt_hook = CheckpointHook(
            interval=2, save_best=['FID', 'IS'], rule=['less', 'greater'])
        self.assertEqual(ckpt_hook.key_indicators, ['FID', 'IS'])
        self.assertEqual(ckpt_hook.rules, ['less', 'greater'])

        # test multi `save_best` with default rule
        ckpt_hook = CheckpointHook(interval=2, save_best=['acc', 'mIoU'])
        self.assertEqual(ckpt_hook.key_indicators, ['acc', 'mIoU'])
        self.assertEqual(ckpt_hook.rules, ['greater', 'greater'])
        runner.message_hub = MessageHub.get_instance(
            'test_after_val_epoch_save_multi_best')
        ckpt_hook.before_train(runner)
        metrics = dict(acc=0.5, mIoU=0.6)
        ckpt_hook.after_val_epoch(runner, metrics)
        best_acc_name = 'best_acc_epoch_9.pth'
        best_acc_path = ckpt_hook.file_client.join_path(
            ckpt_hook.out_dir, best_acc_name)
        best_mIoU_name = 'best_mIoU_epoch_9.pth'
        best_mIoU_path = ckpt_hook.file_client.join_path(
            ckpt_hook.out_dir, best_mIoU_name)
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
        ckpt_hook = CheckpointHook(
            interval=2, by_epoch=False, save_best='acc', rule='greater')
        ckpt_hook.before_train(runner)
        ckpt_hook.after_val_epoch(runner, metrics)
        self.assertEqual(ckpt_hook.key_indicators, ['acc'])
        self.assertEqual(ckpt_hook.rules, ['greater'])
        best_ckpt_name = 'best_acc_iter_9.pth'
        best_ckpt_path = ckpt_hook.file_client.join_path(
            ckpt_hook.out_dir, best_ckpt_name)

        self.assertEqual(
            runner.message_hub.get_info('best_ckpt'), best_ckpt_path)
        self.assertEqual(runner.message_hub.get_info('best_score'), 0.5)

        # check best score updating
        metrics['acc'] = 0.666
        ckpt_hook.after_val_epoch(runner, metrics)
        best_ckpt_name = 'best_acc_iter_9.pth'
        best_ckpt_path = ckpt_hook.file_client.join_path(
            ckpt_hook.out_dir, best_ckpt_name)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt'), best_ckpt_path)

        self.assertEqual(runner.message_hub.get_info('best_score'), 0.666)

        # check best checkpoint name with `by_epoch` is False
        ckpt_hook = CheckpointHook(
            interval=2, by_epoch=False, save_best=['acc', 'mIoU'])
        ckpt_hook.before_train(runner)
        metrics = dict(acc=0.5, mIoU=0.6)
        ckpt_hook.after_val_epoch(runner, metrics)
        best_acc_name = 'best_acc_iter_9.pth'
        best_acc_path = ckpt_hook.file_client.join_path(
            ckpt_hook.out_dir, best_acc_name)
        best_mIoU_name = 'best_mIoU_iter_9.pth'
        best_mIoU_path = ckpt_hook.file_client.join_path(
            ckpt_hook.out_dir, best_mIoU_name)

        self.assertEqual(runner.message_hub.get_info('best_score_acc'), 0.5)
        self.assertEqual(runner.message_hub.get_info('best_score_mIoU'), 0.6)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt_acc'), best_acc_path)
        self.assertEqual(
            runner.message_hub.get_info('best_ckpt_mIoU'), best_mIoU_path)

        # after_val_epoch should not save last_checkpoint
        self.assertFalse(
            osp.isfile(osp.join(runner.work_dir, 'last_checkpoint')))

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
        # Test interval in epoch based training
        save_iterval = 2
        cfg = copy.deepcopy(getattr(self, f'{training_type}_based_cfg'))
        setattr(cfg.train_cfg, f'max_{training_type}s', 11)
        checkpoint_cfg = dict(
            type='CheckpointHook',
            interval=save_iterval,
            by_epoch=training_type == 'epoch')
        cfg.default_hooks = dict(checkpoint=checkpoint_cfg)
        runner = self.build_runner(cfg)
        runner.train()

        for i in range(1, 11):
            if i == 0:
                self.assertFalse(
                    osp.isfile(
                        osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))
            if i % 2 == 0:
                self.assertTrue(
                    osp.isfile(
                        osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))

        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_11.pth')))

        # Test save_optimizer=False
        ckpt = torch.load(osp.join(cfg.work_dir, f'{training_type}_11.pth'))
        self.assertIn('optimizer', ckpt)
        cfg.default_hooks.checkpoint.save_optimizer = False
        runner = self.build_runner(cfg)
        runner.train()
        ckpt = torch.load(osp.join(cfg.work_dir, f'{training_type}_11.pth'))
        self.assertNotIn('optimizer', ckpt)

        # Test save_param_scheduler=False
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

        # Test out_dir
        out_dir = osp.join(self.temp_dir.name, 'out_dir')
        cfg.default_hooks.checkpoint.out_dir = out_dir
        runner = self.build_runner(cfg)
        runner.train()
        self.assertTrue(
            osp.isfile(
                osp.join(out_dir, osp.basename(cfg.work_dir),
                         f'{training_type}_11.pth')))

        # Test max_keep_ckpts.
        del cfg.default_hooks.checkpoint.out_dir
        cfg.default_hooks.checkpoint.max_keep_ckpts = 1
        runner = self.build_runner(cfg)
        runner.train()
        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, f'{training_type}_10.pth')))

        for i in range(10):
            self.assertFalse(
                osp.isfile(osp.join(cfg.work_dir, f'{training_type}_{i}.pth')))

        # Test filename_tmpl
        cfg.default_hooks.checkpoint.filename_tmpl = 'test_{}.pth'
        runner = self.build_runner(cfg)
        runner.train()
        self.assertTrue(osp.isfile(osp.join(cfg.work_dir, 'test_10.pth')))

        # Test save_best
        cfg.default_hooks.checkpoint.save_best = 'test/acc'
        cfg.val_evaluator = dict(type='TriangleMetric', length=11)
        cfg.train_cfg.val_interval = 1
        runner = self.build_runner(cfg)
        runner.train()
        self.assertTrue(
            osp.isfile(osp.join(cfg.work_dir, 'best_test_acc_test_5.pth')))
