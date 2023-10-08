# Copyright (c) OpenMMLab. All rights reserved.
import functools
import time

import pytest

from mmengine.config import Config, ConfigDict  # type: ignore
from mmengine.registry import (DefaultScope, Registry, build_from_cfg,
                               build_model_from_cfg)
from mmengine.utils import ManagerMixin, is_installed


class TestRegistry:

    def test_init(self):
        CATS = Registry('cat')
        assert CATS.name == 'cat'
        assert CATS.module_dict == {}
        assert CATS.build_func is build_from_cfg
        assert len(CATS) == 0

        # test `build_func` parameter
        def build_func(cfg, registry, default_args):
            pass

        CATS = Registry('cat', build_func=build_func)
        assert CATS.build_func is build_func

        # test `parent` parameter
        # `parent` is either None or a `Registry` instance
        with pytest.raises(AssertionError):
            CATS = Registry('little_cat', parent='cat', scope='little_cat')

        LITTLECATS = Registry('little_cat', parent=CATS, scope='little_cat')
        assert LITTLECATS.parent is CATS
        assert CATS._children.get('little_cat') is LITTLECATS

        # test `scope` parameter
        # `scope` is either None or a string
        with pytest.raises(AssertionError):
            CATS = Registry('cat', scope=1)

        CATS = Registry('cat')
        assert CATS.scope == 'test_registry'

        CATS = Registry('cat', scope='cat')
        assert CATS.scope == 'cat'

    def test_split_scope_key(self):
        DOGS = Registry('dogs')

        scope, key = DOGS.split_scope_key('BloodHound')
        assert scope is None and key == 'BloodHound'
        scope, key = DOGS.split_scope_key('hound.BloodHound')
        assert scope == 'hound' and key == 'BloodHound'
        scope, key = DOGS.split_scope_key('hound.little_hound.Dachshund')
        assert scope == 'hound' and key == 'little_hound.Dachshund'

    def test_register_module(self):
        CATS = Registry('cat')

        @CATS.register_module()
        def muchkin(size):
            pass

        assert CATS.get('muchkin') is muchkin
        assert 'muchkin' in CATS

        # test `name` parameter which must be either of None, a string or a
        # sequence of string
        # `name` is None
        @CATS.register_module()
        class BritishShorthair:
            pass

        assert len(CATS) == 2
        assert CATS.get('BritishShorthair') is BritishShorthair

        # `name` is a string
        @CATS.register_module(name='Munchkin')
        class Munchkin:
            pass

        assert len(CATS) == 3
        assert CATS.get('Munchkin') is Munchkin
        assert 'Munchkin' in CATS

        # `name` is a sequence of string
        @CATS.register_module(name=['Siamese', 'Siamese2'])
        class SiameseCat:
            pass

        assert CATS.get('Siamese') is SiameseCat
        assert CATS.get('Siamese2') is SiameseCat
        assert len(CATS) == 5

        # `name` is an invalid type
        with pytest.raises(
                TypeError,
                match=('name must be None, an instance of str, or a sequence '
                       "of str, but got <class 'int'>")):

            @CATS.register_module(name=7474741)
            class SiameseCat:
                pass

        # test `force` parameter, which must be a boolean
        # force is not a boolean
        with pytest.raises(
                TypeError,
                match="force must be a boolean, but got <class 'int'>"):

            @CATS.register_module(force=1)
            class BritishShorthair:
                pass

        # force=False
        with pytest.raises(
                KeyError,
                match='BritishShorthair is already registered in cat '
                'at test_registry'):

            @CATS.register_module()
            class BritishShorthair:
                pass

        # force=True
        @CATS.register_module(force=True)
        class BritishShorthair:
            pass

        assert len(CATS) == 5

        # test `module` parameter, which is either None or a class
        # when the `register_module`` is called as a method rather than a
        # decorator, which must be a class
        with pytest.raises(
                TypeError,
                match='module must be Callable,'
                " but got <class 'str'>"):
            CATS.register_module(module='string')

        class SphynxCat:
            pass

        CATS.register_module(module=SphynxCat)
        assert CATS.get('SphynxCat') is SphynxCat
        assert len(CATS) == 6

        CATS.register_module(name='Sphynx1', module=SphynxCat)
        assert CATS.get('Sphynx1') is SphynxCat
        assert len(CATS) == 7

        CATS.register_module(name=['Sphynx2', 'Sphynx3'], module=SphynxCat)
        assert CATS.get('Sphynx2') is SphynxCat
        assert CATS.get('Sphynx3') is SphynxCat
        assert len(CATS) == 9

        # partial functions can be registered
        muchkin0 = functools.partial(muchkin, size=0)
        CATS.register_module('muchkin0', False, muchkin0)
        # lambda functions can be registered
        CATS.register_module(name='unknown cat', module=lambda: 'unknown')

        assert CATS.get('muchkin0') is muchkin0
        assert 'unknown cat' in CATS
        assert 'muchkin0' in CATS
        assert len(CATS) == 11

    def _build_registry(self):
        """A helper function to build a Hierarchical Registry."""
        #        Hierarchical Registry
        #                           DOGS
        #                      _______|_______
        #                     |               |
        #            HOUNDS (hound)          SAMOYEDS (samoyed)
        #           _______|_______                |
        #          |               |               |
        #     LITTLE_HOUNDS    MID_HOUNDS   LITTLE_SAMOYEDS
        #     (little_hound)   (mid_hound)  (little_samoyed)
        registries = []
        DOGS = Registry('dogs')
        registries.append(DOGS)
        HOUNDS = Registry('hounds', parent=DOGS, scope='hound')
        registries.append(HOUNDS)
        LITTLE_HOUNDS = Registry(
            'little hounds', parent=HOUNDS, scope='little_hound')
        registries.append(LITTLE_HOUNDS)
        MID_HOUNDS = Registry('mid hounds', parent=HOUNDS, scope='mid_hound')
        registries.append(MID_HOUNDS)
        SAMOYEDS = Registry('samoyeds', parent=DOGS, scope='samoyed')
        registries.append(SAMOYEDS)
        LITTLE_SAMOYEDS = Registry(
            'little samoyeds', parent=SAMOYEDS, scope='little_samoyed')
        registries.append(LITTLE_SAMOYEDS)

        return registries

    def test__get_root_registry(self):
        #        Hierarchical Registry
        #                           DOGS
        #                      _______|_______
        #                     |               |
        #            HOUNDS (hound)          SAMOYEDS (samoyed)
        #           _______|_______                |
        #          |               |               |
        #     LITTLE_HOUNDS    MID_HOUNDS   LITTLE_SAMOYEDS
        #     (little_hound)   (mid_hound)  (little_samoyed)
        registries = self._build_registry()
        DOGS, HOUNDS, LITTLE_HOUNDS, MID_HOUNDS = registries[:4]

        assert DOGS._get_root_registry() is DOGS
        assert HOUNDS._get_root_registry() is DOGS
        assert LITTLE_HOUNDS._get_root_registry() is DOGS
        assert MID_HOUNDS._get_root_registry() is DOGS

    def test_get(self):
        #        Hierarchical Registry
        #                           DOGS
        #                      _______|_______
        #                     |               |
        #            HOUNDS (hound)          SAMOYEDS (samoyed)
        #           _______|_______                |
        #          |               |               |
        #     LITTLE_HOUNDS    MID_HOUNDS   LITTLE_SAMOYEDS
        #     (little_hound)   (mid_hound)  (little_samoyed)
        registries = self._build_registry()
        DOGS, HOUNDS, LITTLE_HOUNDS = registries[:3]
        MID_HOUNDS, SAMOYEDS, LITTLE_SAMOYEDS = registries[3:]

        # error type of key
        with pytest.raises(TypeError):
            MID_HOUNDS.get(None)

        @DOGS.register_module()
        def bark(word, times):
            return [word] * times

        dog_bark = functools.partial(bark, 'woof')
        DOGS.register_module('dog_bark', False, dog_bark)

        @DOGS.register_module()
        class GoldenRetriever:
            pass

        assert len(DOGS) == 3
        assert DOGS.get('GoldenRetriever') is GoldenRetriever
        assert DOGS.get('bark') is bark
        assert DOGS.get('dog_bark') is dog_bark

        @HOUNDS.register_module()
        class BloodHound:
            pass

        assert len(HOUNDS) == 1
        # get key from current registry
        assert HOUNDS.get('BloodHound') is BloodHound
        # get key from its children
        assert DOGS.get('hound.BloodHound') is BloodHound
        # get key from current registry
        assert HOUNDS.get('hound.BloodHound') is BloodHound

        # If the key is not found in the current registry, then look for its
        # parent
        assert HOUNDS.get('GoldenRetriever') is GoldenRetriever
        assert HOUNDS.get('bark') is bark
        assert HOUNDS.get('dog_bark') is dog_bark

        @LITTLE_HOUNDS.register_module()
        class Dachshund:
            pass

        assert len(LITTLE_HOUNDS) == 1
        # get key from current registry
        assert LITTLE_HOUNDS.get('Dachshund') is Dachshund
        # get key from its parent
        assert LITTLE_HOUNDS.get('hound.BloodHound') is BloodHound
        # get key from its children
        assert HOUNDS.get('little_hound.Dachshund') is Dachshund
        # get key from its descendants
        assert DOGS.get('hound.little_hound.Dachshund') is Dachshund

        # If the key is not found in the current registry, then look for its
        # parent
        assert LITTLE_HOUNDS.get('BloodHound') is BloodHound
        assert LITTLE_HOUNDS.get('GoldenRetriever') is GoldenRetriever

        @MID_HOUNDS.register_module()
        class Beagle:
            pass

        # get key from its sibling registries
        assert LITTLE_HOUNDS.get('hound.mid_hound.Beagle') is Beagle

        @SAMOYEDS.register_module()
        class PedigreeSamoyed:
            pass

        assert len(SAMOYEDS) == 1
        # get key from its uncle
        assert LITTLE_HOUNDS.get('samoyed.PedigreeSamoyed') is PedigreeSamoyed

        @LITTLE_SAMOYEDS.register_module()
        class LittlePedigreeSamoyed:
            pass

        # get key from its cousin
        assert LITTLE_HOUNDS.get('samoyed.little_samoyed.LittlePedigreeSamoyed'
                                 ) is LittlePedigreeSamoyed

        # get key from its nephews
        assert HOUNDS.get('samoyed.little_samoyed.LittlePedigreeSamoyed'
                          ) is LittlePedigreeSamoyed

        # invalid keys
        # GoldenRetrieverererer can not be found at LITTLE_HOUNDS modules
        assert LITTLE_HOUNDS.get('GoldenRetrieverererer') is None
        # samoyedddd is not a child of DOGS
        assert DOGS.get('samoyedddd.PedigreeSamoyed') is None
        # samoyed is a child of DOGS but LittlePedigreeSamoyed can not be found
        # at SAMOYEDS modules
        assert DOGS.get('samoyed.LittlePedigreeSamoyed') is None
        assert LITTLE_HOUNDS.get('mid_hound.PedigreeSamoyedddddd') is None

        # Get mmengine.utils by string
        utils = LITTLE_HOUNDS.get('mmengine.utils')
        import mmengine.utils
        assert utils is mmengine.utils

        unknown = LITTLE_HOUNDS.get('mmengine.unknown')
        assert unknown is None

    def test__search_child(self):
        #        Hierarchical Registry
        #                           DOGS
        #                      _______|_______
        #                     |               |
        #            HOUNDS (hound)          SAMOYEDS (samoyed)
        #           _______|_______                |
        #          |               |               |
        #     LITTLE_HOUNDS    MID_HOUNDS   LITTLE_SAMOYEDS
        #     (little_hound)   (mid_hound)  (little_samoyed)
        registries = self._build_registry()
        DOGS, HOUNDS, LITTLE_HOUNDS = registries[:3]

        assert DOGS._search_child('hound') is HOUNDS
        assert DOGS._search_child('not a child') is None
        assert DOGS._search_child('little_hound') is LITTLE_HOUNDS
        assert LITTLE_HOUNDS._search_child('hound') is None
        assert LITTLE_HOUNDS._search_child('mid_hound') is None

    @pytest.mark.parametrize('cfg_type', [dict, ConfigDict, Config])
    def test_build(self, cfg_type):
        #        Hierarchical Registry
        #                           DOGS
        #                      _______|_______
        #                     |               |
        #            HOUNDS (hound)          SAMOYEDS (samoyed)
        #           _______|_______                |
        #          |               |               |
        #     LITTLE_HOUNDS    MID_HOUNDS   LITTLE_SAMOYEDS
        #     (little_hound)   (mid_hound)  (little_samoyed)
        registries = self._build_registry()
        DOGS, HOUNDS, LITTLE_HOUNDS, MID_HOUNDS, SAMOYEDS = registries[:5]

        @DOGS.register_module()
        def bark(word, times):
            return ' '.join([word] * times)

        dog_bark = functools.partial(bark, word='woof')
        DOGS.register_module('dog_bark', False, dog_bark)

        bark_cfg = cfg_type(dict(type='bark', word='meow', times=3))
        dog_bark_cfg = cfg_type(dict(type='dog_bark', times=3))

        @DOGS.register_module()
        class GoldenRetriever:
            pass

        gr_cfg = cfg_type(dict(type='GoldenRetriever'))
        assert isinstance(DOGS.build(gr_cfg), GoldenRetriever)
        assert DOGS.build(bark_cfg) == 'meow meow meow'
        assert DOGS.build(dog_bark_cfg) == 'woof woof woof'

        @HOUNDS.register_module()
        class BloodHound:
            pass

        bh_cfg = cfg_type(dict(type='BloodHound'))
        assert isinstance(HOUNDS.build(bh_cfg), BloodHound)
        assert isinstance(HOUNDS.build(gr_cfg), GoldenRetriever)
        assert HOUNDS.build(bark_cfg) == 'meow meow meow'
        assert HOUNDS.build(dog_bark_cfg) == 'woof woof woof'

        @LITTLE_HOUNDS.register_module()
        class Dachshund:
            pass

        d_cfg = cfg_type(dict(type='Dachshund'))
        assert isinstance(LITTLE_HOUNDS.build(d_cfg), Dachshund)

        @MID_HOUNDS.register_module()
        class Beagle:
            pass

        b_cfg = cfg_type(dict(type='Beagle'))
        assert isinstance(MID_HOUNDS.build(b_cfg), Beagle)

        # test `default_scope`
        # switch the current registry to another registry
        DefaultScope.get_instance(
            f'test-{time.time()}', scope_name='mid_hound')
        dog = LITTLE_HOUNDS.build(b_cfg)
        assert isinstance(dog, Beagle)

        # `default_scope` can not be found
        DefaultScope.get_instance(
            f'test2-{time.time()}', scope_name='scope-not-found')
        dog = MID_HOUNDS.build(b_cfg)
        assert isinstance(dog, Beagle)

        # test overwrite default scope with `_scope_`
        @SAMOYEDS.register_module()
        class MySamoyed:

            def __init__(self, friend):
                self.friend = DOGS.build(friend)

        @SAMOYEDS.register_module()
        class YourSamoyed:
            pass

        s_cfg = cfg_type(
            dict(
                _scope_='samoyed',
                type='MySamoyed',
                friend=dict(type='hound.BloodHound')))
        dog = DOGS.build(s_cfg)
        assert isinstance(dog, MySamoyed)
        assert isinstance(dog.friend, BloodHound)
        assert DefaultScope.get_current_instance().scope_name != 'samoyed'

        s_cfg = cfg_type(
            dict(
                _scope_='samoyed',
                type='MySamoyed',
                friend=dict(type='YourSamoyed')))
        dog = DOGS.build(s_cfg)
        assert isinstance(dog, MySamoyed)
        assert isinstance(dog.friend, YourSamoyed)
        assert DefaultScope.get_current_instance().scope_name != 'samoyed'

        # build an instance by lambda or partial function.
        lambda_dog = lambda name: name  # noqa: E731
        DOGS.register_module(name='lambda_dog', module=lambda_dog)
        lambda_cfg = cfg_type(dict(type='lambda_dog', name='unknown'))
        assert DOGS.build(lambda_cfg) == 'unknown'

        DOGS.register_module(
            name='patial dog',
            module=functools.partial(lambda_dog, name='patial'))
        unknown_cfg = cfg_type(dict(type='patial dog'))
        assert DOGS.build(unknown_cfg) == 'patial'

    def test_switch_scope_and_registry(self):
        DOGS = Registry('dogs')
        HOUNDS = Registry('hounds', scope='hound', parent=DOGS)
        SAMOYEDS = Registry('samoyeds', scope='samoyed', parent=DOGS)
        CHIHUAHUA = Registry('chihuahuas', scope='chihuahua', parent=DOGS)

        #                         Hierarchical Registry
        #                                 DOGS
        #               ___________________|___________________
        #              |                   |                   |
        #     HOUNDS (hound)         SAMOYEDS (samoyed) CHIHUAHUA (chihuahua)

        DefaultScope.get_instance(
            f'scope_{time.time()}', scope_name='chihuahua')
        assert DefaultScope.get_current_instance().scope_name == 'chihuahua'

        # Test switch scope and get target registry.
        with CHIHUAHUA.switch_scope_and_registry(scope='hound') as \
                registry:
            assert DefaultScope.get_current_instance().scope_name == 'hound'
            assert id(registry) == id(HOUNDS)

        # Test nested-ly switch scope.
        with CHIHUAHUA.switch_scope_and_registry(scope='samoyed') as \
                samoyed_registry:
            assert DefaultScope.get_current_instance().scope_name == 'samoyed'
            assert id(samoyed_registry) == id(SAMOYEDS)

            with CHIHUAHUA.switch_scope_and_registry(scope='hound') as \
                    hound_registry:
                assert DefaultScope.get_current_instance().scope_name == \
                       'hound'
                assert id(hound_registry) == id(HOUNDS)

        # Test switch to original scope
        assert DefaultScope.get_current_instance().scope_name == 'chihuahua'

        # Test get an unknown registry.
        with CHIHUAHUA.switch_scope_and_registry(scope='unknown') as \
                registry:
            assert id(registry) == id(CHIHUAHUA)
            assert DefaultScope.get_current_instance().scope_name == 'unknown'

    def test_repr(self):
        CATS = Registry('cat')

        @CATS.register_module()
        class BritishShorthair:
            pass

        @CATS.register_module()
        class Munchkin:
            pass

        assert 'Registry of cat' in repr(CATS)
        assert 'BritishShorthair' in repr(CATS)
        assert 'Munchkin' in repr(CATS)


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
