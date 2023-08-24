# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine import (PARAM_SCHEDULERS, Config, ConfigDict, ManagerMixin,
                      Registry, build_from_cfg, build_model_from_cfg)
from mmengine.utils import is_installed


@pytest.mark.parametrize('cfg_type', [dict, ConfigDict, Config])
def test_build_from_cfg(cfg_type):
    BACKBONES = Registry('backbone')

    @BACKBONES.register_module()
    class ResNet:

        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    @BACKBONES.register_module()
    class ResNeXt:

        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    # test `cfg` parameter
    # `cfg` should be a dict, ConfigDict or Config object
    with pytest.raises(
            TypeError,
            match=('cfg should be a dict, ConfigDict or Config, but got '
                   "<class 'str'>")):
        cfg = 'ResNet'
        model = build_from_cfg(cfg, BACKBONES)

    # `cfg` is a dict, ConfigDict or Config object
    cfg = cfg_type(dict(type='ResNet', depth=50))
    model = build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # `cfg` is a dict but it does not contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = dict(depth=50, stages=4)
        cfg = cfg_type(cfg)
        model = build_from_cfg(cfg, BACKBONES)

    # cfg['type'] should be a str or class
    with pytest.raises(
            TypeError,
            match="type must be a str or valid type, but got <class 'int'>"):
        cfg = dict(type=1000)
        cfg = cfg_type(cfg)
        model = build_from_cfg(cfg, BACKBONES)

    cfg = cfg_type(dict(type='ResNeXt', depth=50, stages=3))
    model = build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    cfg = cfg_type(dict(type=ResNet, depth=50))
    model = build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # non-registered class
    with pytest.raises(
            KeyError,
            match='VGG is not in the test_build_functions::backbone registry',
    ):
        cfg = cfg_type(dict(type='VGG'))
        model = build_from_cfg(cfg, BACKBONES)

    # `cfg` contains unexpected arguments
    with pytest.raises(TypeError):
        cfg = cfg_type(dict(type='ResNet', non_existing_arg=50))
        model = build_from_cfg(cfg, BACKBONES)

    # test `default_args` parameter
    cfg = cfg_type(dict(type='ResNet', depth=50))
    model = build_from_cfg(cfg, BACKBONES, cfg_type(dict(stages=3)))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 3

    # default_args must be a dict or None
    with pytest.raises(TypeError):
        cfg = cfg_type(dict(type='ResNet', depth=50))
        model = build_from_cfg(cfg, BACKBONES, default_args=1)

    # cfg or default_args should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = cfg_type(dict(depth=50))
        model = build_from_cfg(
            cfg, BACKBONES, default_args=cfg_type(dict(stages=4)))

    # "type" defined using default_args
    cfg = cfg_type(dict(depth=50))
    model = build_from_cfg(
        cfg, BACKBONES, default_args=cfg_type(dict(type='ResNet')))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = cfg_type(dict(depth=50))
    model = build_from_cfg(
        cfg, BACKBONES, default_args=cfg_type(dict(type=ResNet)))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # test `registry` parameter
    # incorrect registry type
    with pytest.raises(
            TypeError,
            match=('registry must be a mmengine.Registry object, but got '
                   "<class 'str'>")):
        cfg = cfg_type(dict(type='ResNet', depth=50))
        model = build_from_cfg(cfg, 'BACKBONES')

    VISUALIZER = Registry('visualizer')

    @VISUALIZER.register_module()
    class Visualizer(ManagerMixin):

        def __init__(self, name):
            super().__init__(name)

    with pytest.raises(RuntimeError):
        Visualizer.get_current_instance()
    cfg = dict(type='Visualizer', name='visualizer')
    build_from_cfg(cfg, VISUALIZER)
    Visualizer.get_current_instance()


@pytest.mark.skipif(not is_installed('torch'), reason='tests requires torch')
def test_build_model_from_cfg():
    import torch.nn as nn

    BACKBONES = Registry('backbone', build_func=build_model_from_cfg)

    @BACKBONES.register_module()
    class ResNet(nn.Module):

        def __init__(self, depth, stages=4):
            super().__init__()
            self.depth = depth
            self.stages = stages

        def forward(self, x):
            return x

    @BACKBONES.register_module()
    class ResNeXt(nn.Module):

        def __init__(self, depth, stages=4):
            super().__init__()
            self.depth = depth
            self.stages = stages

        def forward(self, x):
            return x

    cfg = dict(type='ResNet', depth=50)
    model = BACKBONES.build(cfg)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(type='ResNeXt', depth=50, stages=3)
    model = BACKBONES.build(cfg)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    cfg = [
        dict(type='ResNet', depth=50),
        dict(type='ResNeXt', depth=50, stages=3)
    ]
    model = BACKBONES.build(cfg)
    assert isinstance(model, nn.Sequential)
    assert isinstance(model[0], ResNet)
    assert model[0].depth == 50 and model[0].stages == 4
    assert isinstance(model[1], ResNeXt)
    assert model[1].depth == 50 and model[1].stages == 3

    # test inherit `build_func` from parent
    NEW_MODELS = Registry('models', parent=BACKBONES, scope='new')
    assert NEW_MODELS.build_func is build_model_from_cfg

    # test specify `build_func`
    def pseudo_build(cfg):
        return cfg

    NEW_MODELS = Registry('models', parent=BACKBONES, build_func=pseudo_build)
    assert NEW_MODELS.build_func is pseudo_build


@pytest.mark.skipif(not is_installed('torch'), reason='tests requires torch')
def test_build_scheduler_from_cfg():
    import torch.nn as nn
    from torch.optim import SGD
    model = nn.Conv2d(1, 1, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    cfg = dict(
        type='LinearParamScheduler',
        optimizer=optimizer,
        param_name='lr',
        begin=0,
        end=100)
    scheduler = PARAM_SCHEDULERS.build(cfg)
    assert scheduler.begin == 0
    assert scheduler.end == 100

    cfg = dict(
        type='LinearParamScheduler',
        convert_to_iter_based=True,
        optimizer=optimizer,
        param_name='lr',
        begin=0,
        end=100,
        epoch_length=10)

    scheduler = PARAM_SCHEDULERS.build(cfg)
    assert scheduler.begin == 0
    assert scheduler.end == 1000
