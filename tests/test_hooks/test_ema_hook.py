# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import torch
import torch.nn as nn

from mmengine.config import ConfigDict
from mmengine.device import is_musa_available
from mmengine.hooks import EMAHook
from mmengine.model import BaseModel, ExponentialMovingAverage
from mmengine.registry import MODELS
from mmengine.testing import RunnerTestCase, assert_allclose
from mmengine.testing.runner_test_case import ToyModel


class DummyWrapper(BaseModel):

    def __init__(self, model):
        super().__init__()
        if not isinstance(model, nn.Module):
            model = MODELS.build(model)
        self.module = model

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ToyModel2(ToyModel):

    def __init__(self):
        super().__init__()
        self.linear3 = nn.Linear(2, 1)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class ToyModel3(ToyModel):

    def __init__(self):
        super().__init__()
        self.linear2 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


# TODO:haowen.han@mtheads.com
@unittest.skipIf(is_musa_available(),
                 "musa backend do not support 'aten::lerp.Scalar_out'")
class TestEMAHook(RunnerTestCase):

    def setUp(self) -> None:
        MODELS.register_module(name='DummyWrapper', module=DummyWrapper)
        MODELS.register_module(name='ToyModel2', module=ToyModel2)
        MODELS.register_module(name='ToyModel3', module=ToyModel3)
        return super().setUp()

    def tearDown(self):
        MODELS.module_dict.pop('DummyWrapper')
        MODELS.module_dict.pop('ToyModel2')
        MODELS.module_dict.pop('ToyModel3')
        return super().tearDown()

    def test_init(self):
        EMAHook()

        with self.assertRaisesRegex(AssertionError, '`begin_iter` must'):
            EMAHook(begin_iter=-1)

        with self.assertRaisesRegex(AssertionError, '`begin_epoch` must'):
            EMAHook(begin_epoch=-1)

        with self.assertRaisesRegex(AssertionError,
                                    '`begin_iter` and `begin_epoch`'):
            EMAHook(begin_iter=1, begin_epoch=1)

    def _get_ema_hook(self, runner):
        for hook in runner.hooks:
            if isinstance(hook, EMAHook):
                return hook

    def test_before_run(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks = [dict(type='EMAHook')]
        runner = self.build_runner(cfg)
        ema_hook = self._get_ema_hook(runner)
        ema_hook.before_run(runner)
        self.assertIsInstance(ema_hook.ema_model, ExponentialMovingAverage)
        self.assertIs(ema_hook.src_model, runner.model)

    def test_before_train(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks = [
            dict(type='EMAHook', begin_epoch=cfg.train_cfg.max_epochs - 1)
        ]
        runner = self.build_runner(cfg)
        ema_hook = self._get_ema_hook(runner)
        ema_hook.before_train(runner)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks = [
            dict(type='EMAHook', begin_epoch=cfg.train_cfg.max_epochs + 1)
        ]
        runner = self.build_runner(cfg)
        ema_hook = self._get_ema_hook(runner)

        with self.assertRaisesRegex(AssertionError, 'self.begin_epoch'):
            ema_hook.before_train(runner)

        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.custom_hooks = [
            dict(type='EMAHook', begin_iter=cfg.train_cfg.max_iters + 1)
        ]
        runner = self.build_runner(cfg)
        ema_hook = self._get_ema_hook(runner)

        with self.assertRaisesRegex(AssertionError, 'self.begin_iter'):
            ema_hook.before_train(runner)

    def test_after_train_iter(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks = [dict(type='EMAHook')]
        runner = self.build_runner(cfg)
        ema_hook = self._get_ema_hook(runner)

        ema_hook = self._get_ema_hook(runner)
        ema_hook.before_run(runner)
        ema_hook.before_train(runner)

        src_model = runner.model
        ema_model = ema_hook.ema_model

        with torch.no_grad():
            for parameter in src_model.parameters():
                parameter.data.copy_(torch.randn(parameter.shape))

        ema_hook.after_train_iter(runner, 1)
        for src, ema in zip(src_model.parameters(), ema_model.parameters()):
            assert_allclose(src.data, ema.data)

        with torch.no_grad():
            for parameter in src_model.parameters():
                parameter.data.copy_(torch.randn(parameter.shape))

        ema_hook.after_train_iter(runner, 1)

        for src, ema in zip(src_model.parameters(), ema_model.parameters()):
            self.assertFalse((src.data == ema.data).all())

    def test_before_val_epoch(self):
        self._test_swap_parameters('before_val_epoch')

    def test_after_val_epoch(self):
        self._test_swap_parameters('after_val_epoch')

    def test_before_test_epoch(self):
        self._test_swap_parameters('before_test_epoch')

    def test_after_test_epoch(self):
        self._test_swap_parameters('after_test_epoch')

    def test_before_save_checkpoint(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        checkpoint = dict(state_dict=ToyModel().state_dict())
        ema_hook = EMAHook()
        ema_hook.before_run(runner)
        ema_hook.before_train(runner)

        ori_checkpoint = copy.deepcopy(checkpoint)
        ema_hook.before_save_checkpoint(runner, checkpoint)

        for key in ori_checkpoint['state_dict'].keys():
            assert_allclose(
                ori_checkpoint['state_dict'][key].cpu(),
                checkpoint['ema_state_dict'][f'module.{key}'].cpu())

            assert_allclose(
                ema_hook.ema_model.state_dict()[f'module.{key}'].cpu(),
                checkpoint['state_dict'][key].cpu())

    def test_after_load_checkpoint(self):
        # Test load a checkpoint without ema_state_dict.
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)
        checkpoint = dict(state_dict=ToyModel().state_dict())
        ema_hook = EMAHook()
        ema_hook.before_run(runner)
        ema_hook.before_train(runner)
        ema_hook.after_load_checkpoint(runner, checkpoint)

        for key in checkpoint['state_dict'].keys():
            assert_allclose(
                checkpoint['state_dict'][key].cpu(),
                ema_hook.ema_model.state_dict()[f'module.{key}'].cpu())

        # Test a warning should be raised when resuming from a checkpoint
        # without `ema_state_dict`
        runner._resume = True
        ema_hook.after_load_checkpoint(runner, checkpoint)
        with self.assertLogs(runner.logger, level='WARNING') as cm:
            ema_hook.after_load_checkpoint(runner, checkpoint)
            self.assertRegex(cm.records[0].msg, 'There is no `ema_state_dict`')

        # Check the weight of state_dict and ema_state_dict have been swapped.
        # when runner._resume is True
        runner._resume = True
        checkpoint = dict(
            state_dict=ToyModel().state_dict(),
            ema_state_dict=ExponentialMovingAverage(ToyModel()).state_dict())
        ori_checkpoint = copy.deepcopy(checkpoint)
        ema_hook.after_load_checkpoint(runner, checkpoint)
        for key in ori_checkpoint['state_dict'].keys():
            assert_allclose(
                ori_checkpoint['state_dict'][key].cpu(),
                ema_hook.ema_model.state_dict()[f'module.{key}'].cpu())

        runner._resume = False
        ema_hook.after_load_checkpoint(runner, checkpoint)

    def test_with_runner(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks = [ConfigDict(type='EMAHook')]
        runner = self.build_runner(cfg)
        ema_hook = self._get_ema_hook(runner)
        runner.train()
        self.assertTrue(
            isinstance(ema_hook.ema_model, ExponentialMovingAverage))

        checkpoint = torch.load(osp.join(self.temp_dir.name, 'epoch_2.pth'))
        self.assertTrue('ema_state_dict' in checkpoint)
        self.assertTrue(checkpoint['ema_state_dict']['steps'] == 8)

        # load and testing
        cfg.load_from = osp.join(self.temp_dir.name, 'epoch_2.pth')
        runner = self.build_runner(cfg)
        runner.test()

        # with model wrapper
        cfg.model = ConfigDict(type='DummyWrapper', model=cfg.model)
        runner = self.build_runner(cfg)
        runner.test()

        # Test load checkpoint without ema_state_dict
        checkpoint = torch.load(osp.join(self.temp_dir.name, 'epoch_2.pth'))
        checkpoint.pop('ema_state_dict')
        torch.save(checkpoint,
                   osp.join(self.temp_dir.name, 'without_ema_state_dict.pth'))

        cfg.load_from = osp.join(self.temp_dir.name,
                                 'without_ema_state_dict.pth')
        runner = self.build_runner(cfg)
        runner.test()

        # Test does not load checkpoint strictly (different name).
        # Test load checkpoint without ema_state_dict
        cfg.model = ConfigDict(type='ToyModel2')
        cfg.custom_hooks = [ConfigDict(type='EMAHook', strict_load=False)]
        runner = self.build_runner(cfg)
        runner.test()

        # Test does not load ckpt strictly (different weight size).
        # Test load checkpoint without ema_state_dict
        cfg.model = ConfigDict(type='ToyModel3')
        runner = self.build_runner(cfg)
        runner.test()

        # Test enable ema at 5 epochs.
        cfg.train_cfg.max_epochs = 10
        cfg.custom_hooks = [ConfigDict(type='EMAHook', begin_epoch=5)]
        runner = self.build_runner(cfg)
        runner.train()
        state_dict = torch.load(
            osp.join(self.temp_dir.name, 'epoch_4.pth'), map_location='cpu')
        self.assertIn('ema_state_dict', state_dict)
        for k, v in state_dict['state_dict'].items():
            assert_allclose(v, state_dict['ema_state_dict']['module.' + k])

        # Test enable ema at 5 iterations.
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.train_cfg.val_interval = 1
        cfg.custom_hooks = [ConfigDict(type='EMAHook', begin_iter=5)]
        cfg.default_hooks.checkpoint.interval = 1
        runner = self.build_runner(cfg)
        runner.train()
        state_dict = torch.load(
            osp.join(self.temp_dir.name, 'iter_4.pth'), map_location='cpu')
        self.assertIn('ema_state_dict', state_dict)
        for k, v in state_dict['state_dict'].items():
            assert_allclose(v, state_dict['ema_state_dict']['module.' + k])
        state_dict = torch.load(
            osp.join(self.temp_dir.name, 'iter_5.pth'), map_location='cpu')
        self.assertIn('ema_state_dict', state_dict)

    def _test_swap_parameters(self, func_name, *args, **kwargs):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks = [dict(type='EMAHook')]
        runner = self.build_runner(cfg)
        ema_hook = self._get_ema_hook(runner)

        runner.train()

        with torch.no_grad():
            for parameter in ema_hook.src_model.parameters():
                parameter.data.copy_(torch.randn(parameter.shape))

        src_model = copy.deepcopy(runner.model)
        ema_model = copy.deepcopy(ema_hook.ema_model)

        func = getattr(ema_hook, func_name)
        func(runner, *args, **kwargs)

        swapped_src = ema_hook.src_model
        swapped_ema = ema_hook.ema_model

        for src, ema, swapped_src, swapped_ema in zip(
                src_model.parameters(), ema_model.parameters(),
                swapped_src.parameters(), swapped_ema.parameters()):
            self.assertTrue((src.data == swapped_ema.data).all())
            self.assertTrue((ema.data == swapped_src.data).all())
