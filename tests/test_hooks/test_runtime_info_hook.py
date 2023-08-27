# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest.mock import Mock

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD

from mmengine.hooks import RuntimeInfoHook
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.registry import DATASETS
from mmengine.testing import RunnerTestCase


class DatasetWithoutMetainfo:
    ...

    def __len__(self):
        return 12


class DatasetWithMetainfo(DatasetWithoutMetainfo):
    metainfo: dict = dict()


class TestRuntimeInfoHook(RunnerTestCase):

    def setUp(self) -> None:
        DATASETS.register_module(module=DatasetWithoutMetainfo, force=True)
        DATASETS.register_module(module=DatasetWithMetainfo, force=True)
        return super().setUp()

    def tearDown(self):
        DATASETS.module_dict.pop('DatasetWithoutMetainfo')
        DATASETS.module_dict.pop('DatasetWithMetainfo')
        return super().tearDown()

    def test_before_and_after_train(self):

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.train_dataloader.dataset.type = 'DatasetWithoutMetainfo'
        runner = self.build_runner(cfg)
        hook = self._get_runtime_info_hook(runner)
        hook.before_train(runner)
        self.assertEqual(runner.message_hub.get_info('loop_stage'), 'train')
        self.assertEqual(runner.message_hub.get_info('epoch'), 0)
        self.assertEqual(runner.message_hub.get_info('iter'), 0)
        self.assertEqual(runner.message_hub.get_info('max_epochs'), 2)
        self.assertEqual(runner.message_hub.get_info('max_iters'), 8)
        hook.after_train(runner)
        self.assertIsNone(runner.message_hub.get_info('loop_stage'))

        cfg.train_dataloader.dataset.type = 'DatasetWithMetainfo'
        runner = self.build_runner(cfg)
        hook.before_train(runner)
        self.assertEqual(runner.message_hub.get_info('dataset_meta'), dict())

    def test_before_train_epoch(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        runner.train_loop._epoch = 9
        hook = self._get_runtime_info_hook(runner)
        hook.before_train_epoch(runner)
        self.assertEqual(runner.message_hub.get_info('epoch'), 9)

    def test_before_train_iter(self):
        # single optimizer
        cfg = copy.deepcopy(self.epoch_based_cfg)
        lr = cfg.optim_wrapper.optimizer.lr
        runner = self.build_runner(cfg)
        # set iter
        runner.train_loop._iter = 9
        # build optim wrapper
        runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
        hook = self._get_runtime_info_hook(runner)
        hook.before_train_iter(runner, batch_idx=2, data_batch=None)
        self.assertEqual(runner.message_hub.get_info('iter'), 9)
        self.assertEqual(
            runner.message_hub.get_scalar('train/lr').current(), lr)

        with self.assertRaisesRegex(AssertionError,
                                    'runner.optim_wrapper.get_lr()'):
            runner.optim_wrapper = Mock()
            runner.optim_wrapper.get_lr = Mock(return_value='error type')
            hook.before_train_iter(runner, batch_idx=2, data_batch=None)

        # multiple optimizers
        model = nn.ModuleDict(
            dict(
                layer1=nn.Linear(1, 1),
                layer2=nn.Linear(1, 1),
            ))
        optim1 = SGD(model.layer1.parameters(), lr=0.01)
        optim2 = SGD(model.layer2.parameters(), lr=0.02)
        optim_wrapper1 = OptimWrapper(optim1)
        optim_wrapper2 = OptimWrapper(optim2)
        optim_wrapper_dict = OptimWrapperDict(
            key1=optim_wrapper1, key2=optim_wrapper2)
        runner.optim_wrapper = optim_wrapper_dict
        hook.before_train_iter(runner, batch_idx=2, data_batch=None)
        self.assertEqual(
            runner.message_hub.get_scalar('train/key1.lr').current(), 0.01)
        self.assertEqual(
            runner.message_hub.get_scalar('train/key2.lr').current(), 0.02)

    def test_after_train_iter(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        hook = self._get_runtime_info_hook(runner)
        hook.after_train_iter(
            runner, batch_idx=2, data_batch=None, outputs={'loss_cls': 1.111})
        self.assertEqual(
            runner.message_hub.get_scalar('train/loss_cls').current(), 1.111)

    def test_before_and_after_val(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        hook = self._get_runtime_info_hook(runner)
        hook.before_val(runner)
        self.assertEqual(runner.message_hub.get_info('loop_stage'), 'val')
        self.assertIsNone(hook.last_loop_stage)
        hook.after_val(runner)
        self.assertIsNone(runner.message_hub.get_info('loop_stage'))

        # Simulate the workflow of calling the ValLoop within the TrainLoop
        runner = self.build_runner(cfg)
        hook = self._get_runtime_info_hook(runner)
        hook.before_train(runner)
        self.assertEqual(runner.message_hub.get_info('loop_stage'), 'train')
        hook.before_val(runner)
        self.assertEqual(runner.message_hub.get_info('loop_stage'), 'val')
        self.assertEqual(hook.last_loop_stage, 'train')
        hook.after_val(runner)
        self.assertEqual(runner.message_hub.get_info('loop_stage'), 'train')
        self.assertIsNone(hook.last_loop_stage)

    def test_after_val_epoch(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        hook = self._get_runtime_info_hook(runner)
        hook.after_val_epoch(runner, metrics={'acc': 0.8})
        self.assertEqual(
            runner.message_hub.get_scalar('val/acc').current(), 0.8)

    def test_before_and_after_test(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        hook = self._get_runtime_info_hook(runner)
        hook.before_test(runner)
        self.assertEqual(runner.message_hub.get_info('loop_stage'), 'test')
        hook.after_test(runner)
        self.assertIsNone(runner.message_hub.get_info('loop_stage'))

    def test_after_test_epoch(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        hook = self._get_runtime_info_hook(runner)
        hook.after_test_epoch(runner, metrics={'acc': 0.8})
        self.assertEqual(
            runner.message_hub.get_scalar('test/acc').current(), 0.8)

    def test_scalar_check(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        hook = self._get_runtime_info_hook(runner)

        # check other scalar dtypes
        val = np.mean([5])  # this is not ndarray but dtype is np.float64.
        hook.after_val_epoch(
            runner,
            metrics={
                'acc_f32': val.astype(np.float32),
                'acc_i32': val.astype(np.int32),
                'acc_u8': val.astype(np.uint8),
                'acc_ndarray': np.array([5]),
            })
        self.assertEqual(
            runner.message_hub.get_scalar('val/acc_f32').current(), 5)
        self.assertEqual(
            runner.message_hub.get_scalar('val/acc_i32').current(), 5)
        self.assertEqual(
            runner.message_hub.get_scalar('val/acc_u8').current(), 5)
        self.assertEqual(
            runner.message_hub.get_scalar('val/acc_ndarray').current(), 5)

        val = torch.tensor([5.0]).mean()
        hook.after_val_epoch(
            runner,
            metrics={
                'acc_f32': val.float(),
                'acc_i64': val.long(),
                'acc_tensor': torch.tensor([5]),
            })
        self.assertEqual(
            runner.message_hub.get_scalar('val/acc_f32').current(), 5)
        self.assertEqual(
            runner.message_hub.get_scalar('val/acc_i64').current(), 5)
        self.assertEqual(
            runner.message_hub.get_scalar('val/acc_tensor').current(), 5)

    def _get_runtime_info_hook(self, runner):
        for hook in runner.hooks:
            if isinstance(hook, RuntimeInfoHook):
                return hook
