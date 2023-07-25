# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os.path as osp
import tempfile
from unittest import TestCase
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn.init import constant_

from mmengine.logging.logger import MMLogger
from mmengine.model import BaseModule, ModuleDict, ModuleList, Sequential
from mmengine.registry import Registry, build_from_cfg

COMPONENTS = Registry('component')
FOOMODELS = Registry('model')

Logger = MMLogger.get_current_instance()


@COMPONENTS.register_module()
class FooConv1d(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1d = nn.Conv1d(4, 1, 4)

    def forward(self, x):
        return self.conv1d(x)


@COMPONENTS.register_module()
class FooConv2d(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv2d = nn.Conv2d(3, 1, 3)

    def forward(self, x):
        return self.conv2d(x)


@COMPONENTS.register_module()
class FooLinear(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(3, 4)

    def forward(self, x):
        return self.linear(x)


@COMPONENTS.register_module()
class FooLinearConv1d(BaseModule):

    def __init__(self, linear=None, conv1d=None, init_cfg=None):
        super().__init__(init_cfg)
        if linear is not None:
            self.linear = build_from_cfg(linear, COMPONENTS)
        if conv1d is not None:
            self.conv1d = build_from_cfg(conv1d, COMPONENTS)

    def forward(self, x):
        x = self.linear(x)
        return self.conv1d(x)


@FOOMODELS.register_module()
class FooModel(BaseModule):

    def __init__(self,
                 component1=None,
                 component2=None,
                 component3=None,
                 component4=None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg)
        if component1 is not None:
            self.component1 = build_from_cfg(component1, COMPONENTS)
        if component2 is not None:
            self.component2 = build_from_cfg(component2, COMPONENTS)
        if component3 is not None:
            self.component3 = build_from_cfg(component3, COMPONENTS)
        if component4 is not None:
            self.component4 = build_from_cfg(component4, COMPONENTS)

        # its type is not BaseModule, it can be initialized
        # with "override" key.
        self.reg = nn.Linear(3, 4)


class TestBaseModule(TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.BaseModule = BaseModule()
        self.model_cfg = dict(
            type='FooModel',
            init_cfg=[
                dict(type='Constant', val=1, bias=2, layer='Linear'),
                dict(type='Constant', val=3, bias=4, layer='Conv1d'),
                dict(type='Constant', val=5, bias=6, layer='Conv2d')
            ],
            component1=dict(type='FooConv1d'),
            component2=dict(type='FooConv2d'),
            component3=dict(type='FooLinear'),
            component4=dict(
                type='FooLinearConv1d',
                linear=dict(type='FooLinear'),
                conv1d=dict(type='FooConv1d')))

        self.model = build_from_cfg(self.model_cfg, FOOMODELS)
        self.logger = MMLogger.get_instance(self._testMethodName)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
        logging.shutdown()
        MMLogger._instance_dict.clear()
        return super().tearDown()

    def test_is_init(self):
        assert self.BaseModule.is_init is False

    def test_init_weights(self):
        """
        Config
        model (FooModel, Linear: weight=1, bias=2, Conv1d: weight=3, bias=4,
                        Conv2d: weight=5, bias=6)
        ├──component1 (FooConv1d)
        ├──component2 (FooConv2d)
        ├──component3 (FooLinear)
        ├──component4 (FooLinearConv1d)
            ├──linear (FooLinear)
            ├──conv1d (FooConv1d)
        ├──reg (nn.Linear)
        Parameters after initialization
        model (FooModel)
        ├──component1 (FooConv1d, weight=3, bias=4)
        ├──component2 (FooConv2d, weight=5, bias=6)
        ├──component3 (FooLinear, weight=1, bias=2)
        ├──component4 (FooLinearConv1d)
            ├──linear (FooLinear, weight=1, bias=2)
            ├──conv1d (FooConv1d, weight=3, bias=4)
        ├──reg (nn.Linear, weight=1, bias=2)
        """
        self.model.init_weights()

        assert torch.equal(
            self.model.component1.conv1d.weight,
            torch.full(self.model.component1.conv1d.weight.shape, 3.0))
        assert torch.equal(
            self.model.component1.conv1d.bias,
            torch.full(self.model.component1.conv1d.bias.shape, 4.0))
        assert torch.equal(
            self.model.component2.conv2d.weight,
            torch.full(self.model.component2.conv2d.weight.shape, 5.0))
        assert torch.equal(
            self.model.component2.conv2d.bias,
            torch.full(self.model.component2.conv2d.bias.shape, 6.0))
        assert torch.equal(
            self.model.component3.linear.weight,
            torch.full(self.model.component3.linear.weight.shape, 1.0))
        assert torch.equal(
            self.model.component3.linear.bias,
            torch.full(self.model.component3.linear.bias.shape, 2.0))
        assert torch.equal(
            self.model.component4.linear.linear.weight,
            torch.full(self.model.component4.linear.linear.weight.shape, 1.0))
        assert torch.equal(
            self.model.component4.linear.linear.bias,
            torch.full(self.model.component4.linear.linear.bias.shape, 2.0))
        assert torch.equal(
            self.model.component4.conv1d.conv1d.weight,
            torch.full(self.model.component4.conv1d.conv1d.weight.shape, 3.0))
        assert torch.equal(
            self.model.component4.conv1d.conv1d.bias,
            torch.full(self.model.component4.conv1d.conv1d.bias.shape, 4.0))
        assert torch.equal(self.model.reg.weight,
                           torch.full(self.model.reg.weight.shape, 1.0))
        assert torch.equal(self.model.reg.bias,
                           torch.full(self.model.reg.bias.shape, 2.0))

        # Test build model from Pretrained weights

        class CustomLinear(BaseModule):

            def __init__(self, init_cfg=None):
                super().__init__(init_cfg)
                self.linear = nn.Linear(1, 1)

            def init_weights(self):
                constant_(self.linear.weight, 1)
                constant_(self.linear.bias, 2)

        @FOOMODELS.register_module()
        class PratrainedModel(FooModel):

            def __init__(self,
                         component1=None,
                         component2=None,
                         component3=None,
                         component4=None,
                         init_cfg=None) -> None:
                super().__init__(component1, component2, component3,
                                 component4, init_cfg)
                self.linear = CustomLinear()

        checkpoint_path = osp.join(self.temp_dir.name, 'test.pth')
        torch.save(self.model.state_dict(), checkpoint_path)
        model_cfg = copy.deepcopy(self.model_cfg)
        model_cfg['type'] = 'PratrainedModel'
        model_cfg['init_cfg'] = dict(
            type='Pretrained', checkpoint=checkpoint_path)
        model = FOOMODELS.build(model_cfg)
        ori_layer_weight = model.linear.linear.weight.clone()
        ori_layer_bias = model.linear.linear.bias.clone()
        model.init_weights()

        self.assertTrue((ori_layer_weight != model.linear.linear.weight).any())
        self.assertTrue((ori_layer_bias != model.linear.linear.bias).any())

        class FakeDDP(nn.Module):

            def __init__(self, module) -> None:
                super().__init__()
                self.module = module

        # Test initialization of nested modules in DDPModule which define
        # `init_weights`.
        with patch('mmengine.model.base_module.is_model_wrapper',
                   lambda x: isinstance(x, FakeDDP)):
            model = FOOMODELS.build(model_cfg)
            model.ddp = FakeDDP(CustomLinear())
            model.init_weights()
            self.assertTrue((model.ddp.module.linear.weight == 1).all())
            self.assertTrue((model.ddp.module.linear.bias == 2).all())

        # Test submodule.init_weights will be skipped if `is_init` is set
        # to True in root model
        model: FooModel = FOOMODELS.build(copy.deepcopy(self.model_cfg))
        for child in model.children():
            child.init_weights = Mock()
        model.is_init = True
        model.init_weights()
        for child in model.children():
            child.init_weights.assert_not_called()

        # Test submodule.init_weights will be skipped if submodule's `is_init`
        # is set to True
        model: FooModel = FOOMODELS.build(copy.deepcopy(self.model_cfg))
        for child in model.children():
            child.init_weights = Mock()
        model.component1.is_init = True
        model.reg.is_init = True
        model.init_weights()
        model.component1.init_weights.assert_not_called()
        model.component2.init_weights.assert_called_once()
        model.component3.init_weights.assert_called_once()
        model.component4.init_weights.assert_called_once()
        model.reg.init_weights.assert_not_called()

    def test_dump_init_info(self):
        import os
        import shutil
        dump_dir = 'tests/test_model/test_dump_info'
        if not (os.path.exists(dump_dir) and os.path.isdir(dump_dir)):
            os.makedirs(dump_dir)
        for filename in os.listdir(dump_dir):
            file_path = os.path.join(dump_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        MMLogger.get_instance('logger1')  # add logger without FileHandler
        model1 = build_from_cfg(self.model_cfg, FOOMODELS)
        model1.init_weights()
        assert len(os.listdir(dump_dir)) == 0
        log_path = os.path.join(dump_dir, 'out.log')
        MMLogger.get_instance(
            'logger2', log_file=log_path)  # add logger with FileHandler
        model2 = build_from_cfg(self.model_cfg, FOOMODELS)
        model2.init_weights()
        assert len(os.listdir(dump_dir)) == 1
        assert os.stat(log_path).st_size != 0
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        shutil.rmtree(dump_dir)


class TestModuleList(TestCase):

    def test_modulelist_weight_init(self):
        models_cfg = [
            dict(
                type='FooConv1d',
                init_cfg=dict(
                    type='Constant', layer='Conv1d', val=0., bias=1.)),
            dict(
                type='FooConv2d',
                init_cfg=dict(
                    type='Constant', layer='Conv2d', val=2., bias=3.)),
        ]
        layers = [build_from_cfg(cfg, COMPONENTS) for cfg in models_cfg]
        modellist = ModuleList(layers)
        modellist.init_weights()
        self.assertTrue(
            torch.equal(modellist[0].conv1d.weight,
                        torch.full(modellist[0].conv1d.weight.shape, 0.)))
        self.assertTrue(
            torch.equal(modellist[0].conv1d.bias,
                        torch.full(modellist[0].conv1d.bias.shape, 1.)))
        self.assertTrue(
            torch.equal(modellist[1].conv2d.weight,
                        torch.full(modellist[1].conv2d.weight.shape, 2.)))
        self.assertTrue(
            torch.equal(modellist[1].conv2d.bias,
                        torch.full(modellist[1].conv2d.bias.shape, 3.)))
        # inner init_cfg has higher priority
        layers = [build_from_cfg(cfg, COMPONENTS) for cfg in models_cfg]
        modellist = ModuleList(
            layers,
            init_cfg=dict(
                type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.))
        modellist.init_weights()
        self.assertTrue(
            torch.equal(modellist[0].conv1d.weight,
                        torch.full(modellist[0].conv1d.weight.shape, 0.)))
        self.assertTrue(
            torch.equal(modellist[0].conv1d.bias,
                        torch.full(modellist[0].conv1d.bias.shape, 1.)))
        self.assertTrue(
            torch.equal(modellist[1].conv2d.weight,
                        torch.full(modellist[1].conv2d.weight.shape, 2.)))
        self.assertTrue(
            torch.equal(modellist[1].conv2d.bias,
                        torch.full(modellist[1].conv2d.bias.shape, 3.)))


class TestModuleDict(TestCase):

    def test_moduledict_weight_init(self):
        models_cfg = dict(
            foo_conv_1d=dict(
                type='FooConv1d',
                init_cfg=dict(
                    type='Constant', layer='Conv1d', val=0., bias=1.)),
            foo_conv_2d=dict(
                type='FooConv2d',
                init_cfg=dict(
                    type='Constant', layer='Conv2d', val=2., bias=3.)),
        )
        layers = {
            name: build_from_cfg(cfg, COMPONENTS)
            for name, cfg in models_cfg.items()
        }
        modeldict = ModuleDict(layers)
        modeldict.init_weights()
        self.assertTrue(
            torch.equal(
                modeldict['foo_conv_1d'].conv1d.weight,
                torch.full(modeldict['foo_conv_1d'].conv1d.weight.shape, 0.)))
        self.assertTrue(
            torch.equal(
                modeldict['foo_conv_1d'].conv1d.bias,
                torch.full(modeldict['foo_conv_1d'].conv1d.bias.shape, 1.)))
        self.assertTrue(
            torch.equal(
                modeldict['foo_conv_2d'].conv2d.weight,
                torch.full(modeldict['foo_conv_2d'].conv2d.weight.shape, 2.)))
        self.assertTrue(
            torch.equal(
                modeldict['foo_conv_2d'].conv2d.bias,
                torch.full(modeldict['foo_conv_2d'].conv2d.bias.shape, 3.)))
        # inner init_cfg has higher priority
        layers = {
            name: build_from_cfg(cfg, COMPONENTS)
            for name, cfg in models_cfg.items()
        }
        modeldict = ModuleDict(
            layers,
            init_cfg=dict(
                type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.))
        modeldict.init_weights()
        self.assertTrue(
            torch.equal(
                modeldict['foo_conv_1d'].conv1d.weight,
                torch.full(modeldict['foo_conv_1d'].conv1d.weight.shape, 0.)))
        self.assertTrue(
            torch.equal(
                modeldict['foo_conv_1d'].conv1d.bias,
                torch.full(modeldict['foo_conv_1d'].conv1d.bias.shape, 1.)))
        self.assertTrue(
            torch.equal(
                modeldict['foo_conv_2d'].conv2d.weight,
                torch.full(modeldict['foo_conv_2d'].conv2d.weight.shape, 2.)))
        self.assertTrue(
            torch.equal(
                modeldict['foo_conv_2d'].conv2d.bias,
                torch.full(modeldict['foo_conv_2d'].conv2d.bias.shape, 3.)))


class TestSequential(TestCase):

    def test_sequential_model_weight_init(self):
        seq_model_cfg = [
            dict(
                type='FooConv1d',
                init_cfg=dict(
                    type='Constant', layer='Conv1d', val=0., bias=1.)),
            dict(
                type='FooConv2d',
                init_cfg=dict(
                    type='Constant', layer='Conv2d', val=2., bias=3.)),
        ]
        layers = [build_from_cfg(cfg, COMPONENTS) for cfg in seq_model_cfg]
        seq_model = Sequential(*layers)
        seq_model.init_weights()
        self.assertTrue(
            torch.equal(seq_model[0].conv1d.weight,
                        torch.full(seq_model[0].conv1d.weight.shape, 0.)))
        self.assertTrue(
            torch.equal(seq_model[0].conv1d.bias,
                        torch.full(seq_model[0].conv1d.bias.shape, 1.)))
        self.assertTrue(
            torch.equal(seq_model[1].conv2d.weight,
                        torch.full(seq_model[1].conv2d.weight.shape, 2.)))
        self.assertTrue(
            torch.equal(seq_model[1].conv2d.bias,
                        torch.full(seq_model[1].conv2d.bias.shape, 3.)))
        # inner init_cfg has higher priority
        layers = [build_from_cfg(cfg, COMPONENTS) for cfg in seq_model_cfg]
        seq_model = Sequential(
            *layers,
            init_cfg=dict(
                type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.))
        seq_model.init_weights()
        self.assertTrue(
            torch.equal(seq_model[0].conv1d.weight,
                        torch.full(seq_model[0].conv1d.weight.shape, 0.)))
        self.assertTrue(
            torch.equal(seq_model[0].conv1d.bias,
                        torch.full(seq_model[0].conv1d.bias.shape, 1.)))
        self.assertTrue(
            torch.equal(seq_model[1].conv2d.weight,
                        torch.full(seq_model[1].conv2d.weight.shape, 2.)))
        self.assertTrue(
            torch.equal(seq_model[1].conv2d.bias,
                        torch.full(seq_model[1].conv2d.bias.shape, 3.)))
