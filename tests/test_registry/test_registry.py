# Copyright (c) OpenMMLab. All rights reserved.
import time

import pytest

from mmengine.config import Config, ConfigDict  # type: ignore
from mmengine.registry import DefaultScope, Registry, build_from_cfg


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
        def muchkin():
            pass

        assert CATS.get('muchkin') is muchkin
        assert 'muchkin' in CATS

        # can only decorate a class or a function
        with pytest.raises(TypeError):

            class Demo:

                def some_method(self):
                    pass

            method = Demo().some_method
            CATS.register_module(name='some_method', module=method)

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
                match='module must be a class or a function,'
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

        # invalid keys
        # GoldenRetrieverererer can not be found at LITTLE_HOUNDS modules
        assert LITTLE_HOUNDS.get('GoldenRetrieverererer') is None
        # samoyedddd is not a child of DOGS
        assert DOGS.get('samoyedddd.PedigreeSamoyed') is None
        # samoyed is a child of DOGS but LittlePedigreeSamoyed can not be found
        # at SAMOYEDS modules
        assert DOGS.get('samoyed.LittlePedigreeSamoyed') is None
        assert LITTLE_HOUNDS.get('mid_hound.PedigreeSamoyedddddd') is None

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
        class GoldenRetriever:
            pass

        gr_cfg = cfg_type(dict(type='GoldenRetriever'))
        assert isinstance(DOGS.build(gr_cfg), GoldenRetriever)

        @HOUNDS.register_module()
        class BloodHound:
            pass

        bh_cfg = cfg_type(dict(type='BloodHound'))
        assert isinstance(HOUNDS.build(bh_cfg), BloodHound)
        assert isinstance(HOUNDS.build(gr_cfg), GoldenRetriever)

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

    def test_get_registry_by_scope(self):
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

        repr_str = 'Registry(name=cat, items={'
        repr_str += (
            "'BritishShorthair': <class 'test_registry.TestRegistry.test_repr."
            "<locals>.BritishShorthair'>, ")
        repr_str += (
            "'Munchkin': <class 'test_registry.TestRegistry.test_repr."
            "<locals>.Munchkin'>")
        repr_str += '})'
        assert repr(CATS) == repr_str
