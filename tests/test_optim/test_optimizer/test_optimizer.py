# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import unittest
from unittest import TestCase
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from torch.distributed.rpc import is_available

from mmengine.dist import get_rank
from mmengine.logging import MMLogger
from mmengine.optim import (OPTIM_WRAPPER_CONSTRUCTORS, OPTIMIZERS,
                            DefaultOptimWrapperConstructor, OptimWrapper,
                            build_optim_wrapper)
from mmengine.optim.optimizer.builder import (BITSANDBYTES_OPTIMIZERS,
                                              DADAPTATION_OPTIMIZERS,
                                              LION_OPTIMIZERS,
                                              TORCH_OPTIMIZERS,
                                              TRANSFORMERS_OPTIMIZERS)
from mmengine.registry import DefaultScope, Registry, build_from_cfg
from mmengine.testing._internal import MultiProcessTestCase
from mmengine.utils.dl_utils import TORCH_VERSION, mmcv_full_available
from mmengine.utils.version_utils import digit_version

MMCV_FULL_AVAILABLE = mmcv_full_available()
if not MMCV_FULL_AVAILABLE:
    sys.modules['mmcv.ops'] = MagicMock(
        DeformConv2d=dict, ModulatedDeformConv2d=dict)


def has_dadaptation() -> bool:
    try:
        import dadaptation  # noqa: F401
        return True
    except ImportError:
        return False


def has_lion() -> bool:
    try:
        import lion_pytorch  # noqa: F401
        return True
    except ImportError:
        return False


def has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def has_transformers() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


class ExampleModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()
        if MMCV_FULL_AVAILABLE:
            from mmcv.ops import DeformConv2dPack
            self.dcn = DeformConv2dPack(
                3, 4, kernel_size=3, deformable_groups=1)


class ExampleDuplicateModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = nn.Parameter(torch.ones(1))
        self.conv1 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
        self.conv2 = nn.Sequential(nn.Conv2d(4, 2, kernel_size=1))
        self.bn = nn.BatchNorm2d(2)
        self.sub = SubModel()
        self.conv3 = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
        self.conv3[0] = self.conv1[0]
        if MMCV_FULL_AVAILABLE:
            from mmcv.ops import DeformConv2dPack
            self.dcn = DeformConv2dPack(
                3, 4, kernel_size=3, deformable_groups=1)

    def forward(self, x):
        return x


class SubModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, kernel_size=1, groups=2)
        self.gn = nn.GroupNorm(2, 2)
        self.param1 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x


class PseudoDataParallel(nn.Module):

    def __init__(self):
        super().__init__()
        self.module = ExampleModel()

    def forward(self, x):
        return x


class TestBuilder(TestCase):

    def setUp(self):
        self.model = ExampleModel()
        self.base_lr = 0.01
        self.momentum = 0.0001
        self.base_wd = 0.9

    def _check_default_optimizer(self, optimizer, model, prefix=''):
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == self.base_lr
        assert optimizer.defaults['momentum'] == self.momentum
        assert optimizer.defaults['weight_decay'] == self.base_wd
        param_groups = optimizer.param_groups[0]
        if MMCV_FULL_AVAILABLE:
            param_names = [
                'param1', 'conv1.weight', 'conv2.weight', 'conv2.bias',
                'bn.weight', 'bn.bias', 'sub.param1', 'sub.conv1.weight',
                'sub.conv1.bias', 'sub.gn.weight', 'sub.gn.bias', 'dcn.weight',
                'dcn.conv_offset.weight', 'dcn.conv_offset.bias'
            ]
        else:
            param_names = [
                'param1', 'conv1.weight', 'conv2.weight', 'conv2.bias',
                'bn.weight', 'bn.bias', 'sub.param1', 'sub.conv1.weight',
                'sub.conv1.bias', 'sub.gn.weight', 'sub.gn.bias'
            ]
        param_dict = dict(model.named_parameters())
        assert len(param_groups['params']) == len(param_names)
        for i in range(len(param_groups['params'])):
            assert torch.equal(param_groups['params'][i],
                               param_dict[prefix + param_names[i]])

    def _check_sgd_optimizer(self,
                             optimizer,
                             model,
                             prefix='',
                             bias_lr_mult=1,
                             bias_decay_mult=1,
                             norm_decay_mult=1,
                             dwconv_decay_mult=1,
                             dcn_offset_lr_mult=1,
                             flat_decay_mult=1,
                             bypass_duplicate=False):
        param_groups = optimizer.param_groups
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == self.base_lr
        assert optimizer.defaults['momentum'] == self.momentum
        assert optimizer.defaults['weight_decay'] == self.base_wd
        model_parameters = list(model.parameters())
        assert len(param_groups) == len(model_parameters)
        for i, param in enumerate(model_parameters):
            param_group = param_groups[i]
            assert torch.equal(param_group['params'][0], param)
            assert param_group['momentum'] == self.momentum

        # param1
        param1 = param_groups[0]
        assert param1['lr'] == self.base_lr
        assert param1['weight_decay'] == self.base_wd * flat_decay_mult
        # conv1.weight
        conv1_weight = param_groups[1]
        assert conv1_weight['lr'] == self.base_lr
        assert conv1_weight['weight_decay'] == self.base_wd
        # conv2.weight
        conv2_weight = param_groups[2]
        assert conv2_weight['lr'] == self.base_lr
        assert conv2_weight['weight_decay'] == self.base_wd
        # conv2.bias
        conv2_bias = param_groups[3]
        assert conv2_bias['lr'] == self.base_lr * bias_lr_mult
        assert conv2_bias['weight_decay'] == self.base_wd * bias_decay_mult
        # bn.weight
        bn_weight = param_groups[4]
        assert bn_weight['lr'] == self.base_lr
        assert bn_weight['weight_decay'] == self.base_wd * norm_decay_mult
        # bn.bias
        bn_bias = param_groups[5]
        assert bn_bias['lr'] == self.base_lr
        assert bn_bias['weight_decay'] == self.base_wd * norm_decay_mult
        # sub.param1
        sub_param1 = param_groups[6]
        assert sub_param1['lr'] == self.base_lr
        assert sub_param1['weight_decay'] == self.base_wd * flat_decay_mult
        # sub.conv1.weight
        sub_conv1_weight = param_groups[7]
        assert sub_conv1_weight['lr'] == self.base_lr
        assert sub_conv1_weight[
            'weight_decay'] == self.base_wd * dwconv_decay_mult
        # sub.conv1.bias
        sub_conv1_bias = param_groups[8]
        assert sub_conv1_bias['lr'] == self.base_lr * bias_lr_mult
        assert sub_conv1_bias['weight_decay'] == self.base_wd * bias_decay_mult
        # sub.gn.weight
        sub_gn_weight = param_groups[9]
        assert sub_gn_weight['lr'] == self.base_lr
        assert sub_gn_weight['weight_decay'] == self.base_wd * norm_decay_mult
        # sub.gn.bias
        sub_gn_bias = param_groups[10]
        assert sub_gn_bias['lr'] == self.base_lr
        assert sub_gn_bias['weight_decay'] == self.base_wd * norm_decay_mult

        # test dcn which requires cuda is available and
        # mmcv-full has been installed
        if torch.cuda.is_available() and MMCV_FULL_AVAILABLE:
            dcn_conv_weight = param_groups[11]
            assert dcn_conv_weight['lr'] == self.base_lr
            assert dcn_conv_weight['weight_decay'] == self.base_wd

            dcn_offset_weight = param_groups[12]
            assert dcn_offset_weight['lr'] == self.base_lr * dcn_offset_lr_mult
            assert dcn_offset_weight['weight_decay'] == self.base_wd

            dcn_offset_bias = param_groups[13]
            assert dcn_offset_bias['lr'] == self.base_lr * dcn_offset_lr_mult
            assert dcn_offset_bias['weight_decay'] == self.base_wd

    def test_torch_optimizers(self):
        torch_optimizers = [
            'ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS',
            'Optimizer', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam'
        ]
        assert set(torch_optimizers).issubset(set(TORCH_OPTIMIZERS))

    @unittest.skipIf(not has_dadaptation(), 'dadaptation is not installed')
    def test_dadaptation_optimizers(self):
        dadaptation_optimizers = ['DAdaptAdaGrad', 'DAdaptAdam', 'DAdaptSGD']
        assert set(dadaptation_optimizers).issubset(
            set(DADAPTATION_OPTIMIZERS))

    @unittest.skipIf(not has_lion(), 'lion-pytorch is not installed')
    def test_lion_optimizers(self):
        assert 'Lion' in LION_OPTIMIZERS

    @unittest.skipIf(not has_bitsandbytes(), 'bitsandbytes is not installed')
    def test_bitsandbytes_optimizers(self):
        bitsandbytes_optimizers = [
            'AdamW8bit', 'Adam8bit', 'Adagrad8bit', 'PagedAdam8bit',
            'PagedAdamW8bit', 'LAMB8bit', 'LARS8bit', 'RMSprop8bit',
            'Lion8bit', 'PagedLion8bit', 'SGD8bit'
        ]
        assert set(bitsandbytes_optimizers).issubset(
            set(BITSANDBYTES_OPTIMIZERS))

    @unittest.skipIf(not has_transformers(), 'transformers is not installed')
    def test_transformers_optimizers(self):
        transformers_optimizers = ['Adafactor']
        assert set(transformers_optimizers).issubset(
            set(TRANSFORMERS_OPTIMIZERS))

    def test_build_optimizer(self):
        # test build function without ``constructor`` and ``paramwise_cfg``
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        optim_wrapper = build_optim_wrapper(self.model, optim_wrapper_cfg)
        self._check_default_optimizer(optim_wrapper.optimizer, self.model)

        # test build optimizer without type in optim_wrapper_cfg
        optim_wrapper_cfg = dict(
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        optim_wrapper = build_optim_wrapper(self.model, optim_wrapper_cfg)
        self.assertIsInstance(optim_wrapper, OptimWrapper)
        self._check_default_optimizer(optim_wrapper.optimizer, self.model)

        # test build function with invalid ``constructor``
        with self.assertRaises(KeyError):
            optim_wrapper_cfg['constructor'] = 'INVALID_CONSTRUCTOR'
            build_optim_wrapper(self.model, optim_wrapper_cfg)

        # test build function with invalid ``paramwise_cfg``
        with self.assertRaises(KeyError):
            optim_wrapper_cfg['paramwise_cfg'] = dict(invalid_mult=1)
            build_optim_wrapper(self.model, optim_wrapper_cfg)

        optim_wrapper_cfg.pop('optimizer')
        optim_wrapper_cfg.pop('constructor')
        optim_wrapper_cfg.pop('paramwise_cfg')
        self.assertRaisesRegex(
            AssertionError, '`optim_wrapper_cfg` must contain',
            lambda: build_optim_wrapper(self.model, optim_wrapper_cfg))

    def test_build_default_optimizer_constructor(self):
        optim_wrapper = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        paramwise_cfg = dict(
            bias_lr_mult=2,
            bias_decay_mult=0.5,
            norm_decay_mult=0,
            dwconv_decay_mult=0.1,
            dcn_offset_lr_mult=0.1,
            flat_decay_mult=0.3)
        optim_constructor_cfg = dict(
            type='DefaultOptimWrapperConstructor',
            optim_wrapper_cfg=optim_wrapper,
            paramwise_cfg=paramwise_cfg)
        optim_constructor = OPTIM_WRAPPER_CONSTRUCTORS.build(
            optim_constructor_cfg)
        optim_wrapper = optim_constructor(self.model)
        self._check_sgd_optimizer(optim_wrapper.optimizer, self.model,
                                  **paramwise_cfg)

    def test_build_custom_optimizer_constructor(self):
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))

        @OPTIM_WRAPPER_CONSTRUCTORS.register_module()
        class MyOptimizerConstructor(DefaultOptimWrapperConstructor):

            def __call__(self, model):
                if hasattr(model, 'module'):
                    model = model.module

                conv1_lr_mult = self.paramwise_cfg.get('conv1_lr_mult', 1.)
                params = []

                for name, param in model.named_parameters():
                    param_group = {'params': [param]}
                    if name.startswith('conv1') and param.requires_grad:
                        param_group['lr'] = self.base_lr * conv1_lr_mult
                    params.append(param_group)
                self.optimizer_cfg['params'] = params

                return build_from_cfg(self.optimizer_cfg, OPTIMIZERS)

        paramwise_cfg = dict(conv1_lr_mult=5)
        optim_constructor_cfg = dict(
            type='MyOptimizerConstructor',
            optim_wrapper_cfg=optim_wrapper_cfg,
            paramwise_cfg=paramwise_cfg)
        optim_constructor = OPTIM_WRAPPER_CONSTRUCTORS.build(
            optim_constructor_cfg)
        optimizer = optim_constructor(self.model)

        param_groups = optimizer.param_groups
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == self.base_lr
        assert optimizer.defaults['momentum'] == self.momentum
        assert optimizer.defaults['weight_decay'] == self.base_wd
        for i, param in enumerate(self.model.parameters()):
            param_group = param_groups[i]
            assert torch.equal(param_group['params'][0], param)
            assert param_group['momentum'] == self.momentum
        # conv1.weight
        assert param_groups[1][
            'lr'] == self.base_lr * paramwise_cfg['conv1_lr_mult']
        assert param_groups[1]['weight_decay'] == self.base_wd

    def test_default_optimizer_constructor(self):
        with self.assertRaises(TypeError):
            # optimizer_cfg must be a dict
            optimizer_cfg = []
            optim_constructor = DefaultOptimWrapperConstructor(optimizer_cfg)
            optim_constructor(self.model)

        with self.assertRaises(TypeError):
            # paramwise_cfg must be a dict or None
            optim_wrapper_cfg = dict(
                type='OptimWrapper',
                optimizer=dict(lr=0.0001, weight_decay=None))
            paramwise_cfg = ['error']
            optim_constructor = DefaultOptimWrapperConstructor(
                optim_wrapper_cfg, paramwise_cfg)
            optim_constructor(self.model)

        with self.assertRaises(ValueError):
            # bias_decay_mult/norm_decay_mult is specified but weight_decay
            # is None
            optim_wrapper_cfg = dict(
                type='OptimWrapper',
                optimizer=dict(lr=0.0001, weight_decay=None))
            paramwise_cfg = dict(bias_decay_mult=1, norm_decay_mult=1)
            optim_constructor = DefaultOptimWrapperConstructor(
                optim_wrapper_cfg, paramwise_cfg)
            optim_constructor(self.model)

        # basic config with ExampleModel
        optimizer_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        optim_constructor = DefaultOptimWrapperConstructor(optimizer_cfg)
        optim_wrapper = optim_constructor(self.model)
        self._check_default_optimizer(optim_wrapper.optimizer, self.model)

        # Support building custom optimizers
        CUSTOM_OPTIMIZERS = Registry(
            'custom optimizer', scope='custom optimizer', parent=OPTIMIZERS)

        class CustomOptimizer(torch.optim.SGD):

            def __init__(self, model_params, *args, **kwargs):
                super().__init__(params=model_params, *args, **kwargs)

        CUSTOM_OPTIMIZERS.register_module()(CustomOptimizer)
        optimizer_cfg = dict(optimizer=dict(type='CustomOptimizer', lr=0.1), )
        with DefaultScope.overwrite_default_scope('custom optimizer'):
            optim_constructor = DefaultOptimWrapperConstructor(optimizer_cfg)
            optim_wrapper = optim_constructor(self.model)
        OPTIMIZERS.children.pop('custom optimizer')

    def test_default_optimizer_constructor_with_model_wrapper(self):
        # basic config with pseudo data parallel
        model = PseudoDataParallel()
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        paramwise_cfg = None
        optim_constructor = DefaultOptimWrapperConstructor(optim_wrapper_cfg)
        optim_wrapper = optim_constructor(model)
        self._check_default_optimizer(
            optim_wrapper.optimizer, model, prefix='module.')

        # paramwise_cfg with pseudo data parallel
        model = PseudoDataParallel()
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        paramwise_cfg = dict(
            bias_lr_mult=2,
            bias_decay_mult=0.5,
            norm_decay_mult=0,
            dwconv_decay_mult=0.1,
            dcn_offset_lr_mult=0.1,
            flat_decay_mult=0.3)
        optim_constructor = DefaultOptimWrapperConstructor(
            optim_wrapper_cfg, paramwise_cfg)
        optim_wrapper = optim_constructor(model)
        self._check_sgd_optimizer(
            optim_wrapper.optimizer, model, prefix='module.', **paramwise_cfg)

        # basic config with DataParallel
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(ExampleModel())
            optim_wrapper_cfg = dict(
                type='OptimWrapper',
                optimizer=dict(
                    type='SGD',
                    lr=self.base_lr,
                    weight_decay=self.base_wd,
                    momentum=self.momentum))
            paramwise_cfg = None
            optim_constructor = DefaultOptimWrapperConstructor(
                optim_wrapper_cfg)
            optim_wrapper = optim_constructor(model)
            self._check_default_optimizer(
                optim_wrapper.optimizer, model, prefix='module.')

        # paramwise_cfg with DataParallel
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(self.model)
            optim_wrapper_cfg = dict(
                type='OptimWrapper',
                optimizer=dict(
                    type='SGD',
                    lr=self.base_lr,
                    weight_decay=self.base_wd,
                    momentum=self.momentum))
            paramwise_cfg = dict(
                bias_lr_mult=2,
                bias_decay_mult=0.5,
                norm_decay_mult=0,
                dwconv_decay_mult=0.1,
                dcn_offset_lr_mult=0.1,
                flat_decay_mult=0.3)
            optim_constructor = DefaultOptimWrapperConstructor(
                optim_wrapper_cfg, paramwise_cfg)
            optim_wrapper = optim_constructor(model)
            self._check_sgd_optimizer(
                optim_wrapper.optimizer,
                model,
                prefix='module.',
                **paramwise_cfg)

    def test_default_optimizer_constructor_with_empty_paramwise_cfg(self):
        # Empty paramwise_cfg with ExampleModel
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        paramwise_cfg = dict()
        optim_constructor = DefaultOptimWrapperConstructor(
            optim_wrapper_cfg, paramwise_cfg)
        optim_wrapper = optim_constructor(self.model)
        self._check_default_optimizer(optim_wrapper.optimizer, self.model)

        # Empty paramwise_cfg with ExampleModel and no grad
        model = ExampleModel()
        for param in model.parameters():
            param.requires_grad = False
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        paramwise_cfg = dict()
        optim_constructor = DefaultOptimWrapperConstructor(optim_wrapper_cfg)
        optim_wrapper = optim_constructor(model)
        self._check_default_optimizer(optim_wrapper.optimizer, model)

    def test_default_optimizer_constructor_with_paramwise_cfg(self):
        # paramwise_cfg with ExampleModel
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        paramwise_cfg = dict(
            bias_lr_mult=2,
            bias_decay_mult=0.5,
            norm_decay_mult=0,
            dwconv_decay_mult=0.1,
            dcn_offset_lr_mult=0.1,
            flat_decay_mult=0.3)
        optim_constructor = DefaultOptimWrapperConstructor(
            optim_wrapper_cfg, paramwise_cfg)
        optim_wrapper = optim_constructor(self.model)
        self._check_sgd_optimizer(optim_wrapper.optimizer, self.model,
                                  **paramwise_cfg)

    def test_default_optimizer_constructor_no_grad(self):
        # paramwise_cfg with ExampleModel and no grad
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        paramwise_cfg = dict(
            bias_lr_mult=2,
            bias_decay_mult=0.5,
            norm_decay_mult=0,
            dwconv_decay_mult=0.1,
            dcn_offset_lr_mult=0.1)

        for param in self.model.parameters():
            param.requires_grad = False
        optim_constructor = DefaultOptimWrapperConstructor(
            optim_wrapper_cfg, paramwise_cfg)
        optim_wrapper = optim_constructor(self.model)
        optimizer = optim_wrapper.optimizer
        param_groups = optimizer.param_groups
        assert isinstance(optim_wrapper.optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == self.base_lr
        assert optimizer.defaults['momentum'] == self.momentum
        assert optimizer.defaults['weight_decay'] == self.base_wd
        for i, (name, param) in enumerate(self.model.named_parameters()):
            param_group = param_groups[i]
            assert torch.equal(param_group['params'][0], param)
            assert param_group['momentum'] == self.momentum
            assert param_group['lr'] == self.base_lr
            assert param_group['weight_decay'] == self.base_wd

    def test_default_optimizer_constructor_bypass_duplicate(self):
        # paramwise_cfg with bypass_duplicate option
        model = ExampleDuplicateModel()
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        paramwise_cfg = dict(
            bias_lr_mult=2,
            bias_decay_mult=0.5,
            norm_decay_mult=0,
            dwconv_decay_mult=0.1)

        with self.assertRaisesRegex(
                ValueError,
                'some parameters appear in more than one parameter group'):
            optim_constructor = DefaultOptimWrapperConstructor(
                optim_wrapper_cfg, paramwise_cfg)
            optim_constructor(model)

        paramwise_cfg = dict(
            bias_lr_mult=2,
            bias_decay_mult=0.5,
            norm_decay_mult=0,
            dwconv_decay_mult=0.1,
            dcn_offset_lr_mult=0.1,
            flat_decay_mult=0.3,
            bypass_duplicate=True)
        optim_constructor = DefaultOptimWrapperConstructor(
            optim_wrapper_cfg, paramwise_cfg)

        with self.assertLogs(MMLogger.get_current_instance(), level='WARNING'):
            # Warning should be raised since conv3.0 is a duplicate param.
            optim_constructor(model)
        optim_wrapper = optim_constructor(model)
        model_parameters = list(model.parameters())
        num_params = 14 if MMCV_FULL_AVAILABLE else 11
        assert len(optim_wrapper.optimizer.param_groups) == len(
            model_parameters) == num_params
        self._check_sgd_optimizer(optim_wrapper.optimizer, model,
                                  **paramwise_cfg)

        # test DefaultOptimWrapperConstructor when the params in shared
        # modules do not require grad
        model.conv1[0].requires_grad_(False)
        with self.assertLogs(MMLogger.get_current_instance(), level='WARNING'):
            # Warning should be raised since conv3.0 is a duplicate param.
            optim_constructor(model)
        optim_wrapper = optim_constructor(model)
        model_parameters = list(model.parameters())
        num_params = 14 if MMCV_FULL_AVAILABLE else 11
        assert len(optim_wrapper.optimizer.param_groups) == len(
            model_parameters) == num_params
        self._check_sgd_optimizer(optim_wrapper.optimizer, model,
                                  **paramwise_cfg)

    def test_default_optimizer_constructor_custom_key(self):
        # test DefaultOptimWrapperConstructor with custom_keys and
        # ExampleModel
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        paramwise_cfg = dict(
            custom_keys={
                'param1': dict(lr_mult=10),
                'sub': dict(lr_mult=0.1, decay_mult=0),
                'sub.gn': dict(lr_mult=0.01),
                'non_exist_key': dict(lr_mult=0.0)
            },
            norm_decay_mult=0.5)

        with self.assertRaises(TypeError):
            # custom_keys should be a dict
            paramwise_cfg_ = dict(custom_keys=[0.1, 0.0001])
            optim_constructor = DefaultOptimWrapperConstructor(
                optim_wrapper_cfg, paramwise_cfg_)
            optimizer = optim_constructor(self.model)

        with self.assertRaises(ValueError):
            # if 'decay_mult' is specified in custom_keys, weight_decay
            # should be specified
            optim_wrapper_cfg_ = dict(
                type='OptimWrapper', optimizer=dict(type='SGD', lr=0.01))
            paramwise_cfg_ = dict(
                custom_keys={'.backbone': dict(decay_mult=0.5)})
            optim_constructor = DefaultOptimWrapperConstructor(
                optim_wrapper_cfg_, paramwise_cfg_)
            optim_constructor(self.model)

        optim_constructor = DefaultOptimWrapperConstructor(
            optim_wrapper_cfg, paramwise_cfg)
        optimizer = optim_constructor(self.model).optimizer
        # check optimizer type and default config
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == self.base_lr
        assert optimizer.defaults['momentum'] == self.momentum
        assert optimizer.defaults['weight_decay'] == self.base_wd

        # check params groups
        param_groups = optimizer.param_groups

        groups = []
        group_settings = []
        # group 1, matches of 'param1'
        # 'param1' is the longest match for 'sub.param1'
        groups.append(['param1', 'sub.param1'])
        group_settings.append({
            'lr': self.base_lr * 10,
            'momentum': self.momentum,
            'weight_decay': self.base_wd,
        })
        # group 2, matches of 'sub.gn'
        groups.append(['sub.gn.weight', 'sub.gn.bias'])
        group_settings.append({
            'lr': self.base_lr * 0.01,
            'momentum': self.momentum,
            'weight_decay': self.base_wd,
        })
        # group 3, matches of 'sub'
        groups.append(['sub.conv1.weight', 'sub.conv1.bias'])
        group_settings.append({
            'lr': self.base_lr * 0.1,
            'momentum': self.momentum,
            'weight_decay': 0,
        })
        # group 4, bn is configured by 'norm_decay_mult'
        groups.append(['bn.weight', 'bn.bias'])
        group_settings.append({
            'lr': self.base_lr,
            'momentum': self.momentum,
            'weight_decay': self.base_wd * 0.5,
        })
        # group 5, default group
        groups.append(['conv1.weight', 'conv2.weight', 'conv2.bias'])
        group_settings.append({
            'lr': self.base_lr,
            'momentum': self.momentum,
            'weight_decay': self.base_wd
        })

        num_params = 14 if MMCV_FULL_AVAILABLE else 11
        assert len(param_groups) == num_params
        for i, (name, param) in enumerate(self.model.named_parameters()):
            assert torch.equal(param_groups[i]['params'][0], param)
            for group, settings in zip(groups, group_settings):
                if name in group:
                    for setting in settings:
                        assert param_groups[i][setting] == settings[
                            setting], f'{name} {setting}'

        # test DefaultOptimWrapperConstructor with custom_keys and
        # ExampleModel 2
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD', lr=self.base_lr, momentum=self.momentum))
        paramwise_cfg = dict(custom_keys={'param1': dict(lr_mult=10)})

        optim_constructor = DefaultOptimWrapperConstructor(
            optim_wrapper_cfg, paramwise_cfg)
        optimizer = optim_constructor(self.model).optimizer
        # check optimizer type and default config
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults['lr'] == self.base_lr
        assert optimizer.defaults['momentum'] == self.momentum
        assert optimizer.defaults['weight_decay'] == 0

        # check params groups
        param_groups = optimizer.param_groups

        groups = []
        group_settings = []
        # group 1, matches of 'param1'
        groups.append(['param1', 'sub.param1'])
        group_settings.append({
            'lr': self.base_lr * 10,
            'momentum': self.momentum,
            'weight_decay': 0,
        })
        # group 2, default group
        groups.append([
            'sub.conv1.weight', 'sub.conv1.bias', 'sub.gn.weight',
            'sub.gn.bias', 'conv1.weight', 'conv2.weight', 'conv2.bias',
            'bn.weight', 'bn.bias'
        ])
        group_settings.append({
            'lr': self.base_lr,
            'momentum': self.momentum,
            'weight_decay': 0
        })

        num_params = 14 if MMCV_FULL_AVAILABLE else 11
        assert len(param_groups) == num_params
        for i, (name, param) in enumerate(self.model.named_parameters()):
            assert torch.equal(param_groups[i]['params'][0], param)
            for group, settings in zip(groups, group_settings):
                if name in group:
                    for setting in settings:
                        assert param_groups[i][setting] == settings[
                            setting], f'{name} {setting}'


@unittest.skipIf(
    (digit_version(TORCH_VERSION) < digit_version('1.8.0'))
    or not is_available(),
    reason='ZeRO requires pytorch>=1.8 with torch.distributed.rpc available.')
class TestZeroOptimizer(MultiProcessTestCase):

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _check_default_optimizer(self, optimizer, model):
        self.assertIsInstance(optimizer.optim, torch.optim.SGD)
        self.assertEqual(optimizer.defaults['lr'], self.base_lr)
        self.assertEqual(optimizer.defaults['momentum'], self.momentum)
        self.assertEqual(optimizer.defaults['weight_decay'], self.base_wd)
        param_groups = optimizer.param_groups
        params_set = set(model.parameters())
        self.assertEqual(
            sum(len(param_group['params']) for param_group in param_groups),
            len(params_set))
        self.assertTrue(
            all(param in params_set for param_group in param_groups
                for param in param_group['params']))
        state_dict = optimizer.state_dict()
        if get_rank() == 0:
            self.assertEqual(
                sum(len(pg['params']) for pg in state_dict['param_groups']),
                len(params_set))
        else:
            self.assertEqual(state_dict, {})

    def test_zero_redundancy_optimizer(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ExampleModel()
        self.base_lr = 0.01
        self.momentum = 0.0001
        self.base_wd = 0.9

        # test build function
        optim_wrapper_cfg = dict(
            optimizer=dict(
                type='ZeroRedundancyOptimizer',
                optimizer_type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum))
        optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
        self._check_default_optimizer(optim_wrapper.optimizer, model)

        # test build optimizer without ``optimizer_type``
        with self.assertRaises(TypeError):
            optim_wrapper_cfg = dict(
                optimizer=dict(
                    type='ZeroRedundancyOptimizer',
                    lr=self.base_lr,
                    weight_decay=self.base_wd,
                    momentum=self.momentum))
            optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)

    @unittest.skipIf(
        digit_version(TORCH_VERSION) < digit_version('1.12.0'),
        reason='ZeRO started to support param groups since pytorch 1.12.0')
    def test_zero_redundancy_optimizer_with_paramwise_cfg(self):
        self._init_dist_env(self.rank, self.world_size)
        model = ExampleModel()
        self.base_lr = 0.01
        self.momentum = 0.0001
        self.base_wd = 0.9

        # test build function
        paramwise_cfg = dict(
            custom_keys={
                'conv1': dict(lr_mult=0.0, decay_mult=0.0),
                'conv2': dict(lr_mult=1.0, decay_mult=2.0)
            })
        optim_wrapper_cfg = dict(
            optimizer=dict(
                type='ZeroRedundancyOptimizer',
                optimizer_type='SGD',
                lr=self.base_lr,
                weight_decay=self.base_wd,
                momentum=self.momentum),
            paramwise_cfg=paramwise_cfg)
        optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
        self._check_default_optimizer(optim_wrapper.optimizer, model)

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29510'
        os.environ['RANK'] = str(rank)
        torch.distributed.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)
