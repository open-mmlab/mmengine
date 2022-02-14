# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine.registry import Registry, build_from_cfg


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

        # can only decorate a class
        with pytest.raises(TypeError):

            @CATS.register_module()
            def some_method():
                pass

        # test `name` parameter which must be either of None, a string or a
        # sequence of string
        # `name` is None
        @CATS.register_module()
        class BritishShorthair:
            pass

        assert len(CATS) == 1
        assert CATS.get('BritishShorthair') is BritishShorthair

        # `name` is a string
        @CATS.register_module(name='Munchkin')
        class Munchkin:
            pass

        assert len(CATS) == 2
        assert CATS.get('Munchkin') is Munchkin
        assert 'Munchkin' in CATS

        # `name` is a sequence of string
        @CATS.register_module(name=['Siamese', 'Siamese2'])
        class SiameseCat:
            pass

        assert CATS.get('Siamese') is SiameseCat
        assert CATS.get('Siamese2') is SiameseCat
        assert len(CATS) == 4

        # `name` is an invalid type
        with pytest.raises(
                TypeError,
                match=('name must be either of None, an instance of str or a '
                       "sequence of str, but got <class 'int'>")):

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
                match='BritishShorthair is already registered in cat'):

            @CATS.register_module()
            class BritishShorthair:
                pass

        # force=True
        @CATS.register_module(force=True)
        class BritishShorthair:
            pass

        assert len(CATS) == 4

        # test `module` parameter, which is either None or a class
        # when the `register_module`` is called as a method rather than a
        # decorator, which must be a class
        with pytest.raises(
                TypeError,
                match="module must be a class, but got <class 'str'>"):
            CATS.register_module(module='string')

        class SphynxCat:
            pass

        CATS.register_module(module=SphynxCat)
        assert CATS.get('SphynxCat') is SphynxCat
        assert len(CATS) == 5

        CATS.register_module(name='Sphynx1', module=SphynxCat)
        assert CATS.get('Sphynx1') is SphynxCat
        assert len(CATS) == 6

        CATS.register_module(name=['Sphynx2', 'Sphynx3'], module=SphynxCat)
        assert CATS.get('Sphynx2') is SphynxCat
        assert CATS.get('Sphynx3') is SphynxCat
        assert len(CATS) == 8

    def _build_registry(self):
        r"""A helper function to build a hierarchy registry.
                                     DOGS
                                  /       \
                                 /         \
                    HOUNDS (hound)          SAMOYEDS (samoyed)
                      /        \                 |
                     /          \                |
             LITTLE_HOUNDS    MID_HOUNDS   LITTLE_SAMOYEDS
             (little_hound)   (mid_hound)  (little_samoyed)
        """
        registries = []
        DOGS = Registry('dogs')
        registries.append(DOGS)
        HOUNDS = Registry('dogs', parent=DOGS, scope='hound')
        registries.append(HOUNDS)
        LITTLE_HOUNDS = Registry('dogs', parent=HOUNDS, scope='little_hound')
        registries.append(LITTLE_HOUNDS)
        MID_HOUNDS = Registry('dogs', parent=HOUNDS, scope='mid_hound')
        registries.append(MID_HOUNDS)
        SAMOYEDS = Registry('dogs', parent=DOGS, scope='samoyed')
        registries.append(SAMOYEDS)
        LITTLE_SAMOYEDS = Registry(
            'dogs', parent=SAMOYEDS, scope='little_samoyed')
        registries.append(LITTLE_SAMOYEDS)

        return registries

    def test_get(self):
        #        Hierarchy Registry
        #
        #                             DOGS
        #                          /       \
        #                         /         \
        #            HOUNDS (hound)          SAMOYEDS (samoyed)
        #              /        \                  |
        #             /          \                 |
        #     LITTLE_HOUNDS    MID_HOUNDS   LITTLE_SAMOYEDS
        #     (little_hound)   (mid_hound)  (little_samoyed)
        registries = self._build_registry()
        DOGS, HOUNDS, LITTLE_HOUNDS = registries[:3]
        MID_HOUNDS, SAMOYEDS, LITTLE_SAMOYEDS = registries[3:]

        @DOGS.register_module()
        class GoldenRetriever:
            pass

        assert len(DOGS) == 1
        assert DOGS.get('GoldenRetriever') is GoldenRetriever

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

    def test_build(self):
        #        Hierarchy Registry
        #
        #                             DOGS
        #                          /       \
        #                         /         \
        #            HOUNDS (hound)          SAMOYEDS (samoyed)
        #              /        \                  |
        #             /          \                 |
        #     LITTLE_HOUNDS    MID_HOUNDS   LITTLE_SAMOYEDS
        #     (little_hound)   (mid_hound)  (little_samoyed)
        registries = self._build_registry()
        DOGS, HOUNDS, LITTLE_HOUNDS, MID_HOUNDS = registries[:4]

        @DOGS.register_module()
        class GoldenRetriever:
            pass

        gr_cfg = dict(type='GoldenRetriever')
        assert isinstance(DOGS.build(gr_cfg), GoldenRetriever)

        @HOUNDS.register_module()
        class BloodHound:
            pass

        bh_cfg = dict(type='BloodHound')
        assert isinstance(HOUNDS.build(bh_cfg), BloodHound)
        assert isinstance(HOUNDS.build(gr_cfg), GoldenRetriever)

        @LITTLE_HOUNDS.register_module()
        class Dachshund:
            pass

        d_cfg = dict(type='Dachshund')
        assert isinstance(LITTLE_HOUNDS.build(d_cfg), Dachshund)

        @MID_HOUNDS.register_module()
        class Beagle:
            pass

        b_cfg = dict(type='Beagle')
        assert isinstance(MID_HOUNDS.build(b_cfg), Beagle)

        # test `default_scope`
        # `default_scope` is an invalid scope
        with pytest.raises(KeyError):
            LITTLE_HOUNDS.build(b_cfg, default_scope='invalid_mid_hound')

        # switch the current registry to another registry
        dog = LITTLE_HOUNDS.build(b_cfg, default_scope='mid_hound')
        assert isinstance(dog, Beagle)

    def test_repr(self):
        CATS = Registry('cat')

        @CATS.register_module()
        class BritishShorthair:
            pass

        @CATS.register_module()
        class Munchkin:
            pass

        repr_str = 'Registry(name=cat, items={'
        repr_str += (
            "'BritishShorthair': <class 'test_registry.TestRegistry.test_repr."
            "<locals>.BritishShorthair'>, ")
        repr_str += (
            "'Munchkin': <class 'test_registry.TestRegistry.test_repr."
            "<locals>.Munchkin'>")
        repr_str += '})'
        assert repr(CATS) == repr_str


def test_build_from_cfg():
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
    cfg = dict(type='ResNet', depth=50)
    model = build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(type='ResNeXt', depth=50, stages=3)
    model = build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    cfg = dict(type=ResNet, depth=50)
    model = build_from_cfg(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # `cfg` should be a dict
    with pytest.raises(
            TypeError, match="cfg must be a dict, but got <class 'str'>"):
        cfg = 'ResNet'
        model = build_from_cfg(cfg, BACKBONES)

    # `cfg` is a dict but it does not contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = dict(depth=50, stages=4)
        model = build_from_cfg(cfg, BACKBONES)

    # cfg['type'] should be a str or class
    with pytest.raises(
            TypeError,
            match="type must be a str or valid type, but got <class 'int'>"):
        cfg = dict(type=1000)
        model = build_from_cfg(cfg, BACKBONES)

    # non-registered class
    with pytest.raises(KeyError, match='VGG is not in the backbone registry'):
        cfg = dict(type='VGG')
        model = build_from_cfg(cfg, BACKBONES)

    # `cfg` contains unexpected arguments
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', non_existing_arg=50)
        model = build_from_cfg(cfg, BACKBONES)

    # test `default_args` parameter
    cfg = dict(type='ResNet', depth=50)
    model = build_from_cfg(cfg, BACKBONES, default_args={'stages': 3})
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 3

    # default_args must be a dict or None
    with pytest.raises(TypeError):
        cfg = dict(type='ResNet', depth=50)
        model = build_from_cfg(cfg, BACKBONES, default_args=1)

    # cfg or default_args should contain the key "type"
    with pytest.raises(KeyError, match='must contain the key "type"'):
        cfg = dict(depth=50)
        model = build_from_cfg(cfg, BACKBONES, default_args=dict(stages=4))

    # "type" defined using default_args
    cfg = dict(depth=50)
    model = build_from_cfg(cfg, BACKBONES, default_args=dict(type='ResNet'))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(depth=50)
    model = build_from_cfg(cfg, BACKBONES, default_args=dict(type=ResNet))
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    # test `registry` parameter
    # incorrect registry type
    with pytest.raises(
            TypeError,
            match=('registry must be a mmengine.Registry object, but got '
                   "<class 'str'>")):
        cfg = dict(type='ResNet', depth=50)
        model = build_from_cfg(cfg, 'BACKBONES')
