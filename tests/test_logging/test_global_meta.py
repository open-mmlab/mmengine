# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine.logging import BaseGlobalAccessible, GlobalMeta


class SubClassA(BaseGlobalAccessible):

    def __init__(self, name='', *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class SubClassB(BaseGlobalAccessible):

    def __init__(self, name='', *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class TestGlobalMeta:

    def test_init(self):
        # Subclass's constructor does not contain name arguments will raise an
        # error.
        with pytest.raises(AssertionError):

            class SubClassNoName(metaclass=GlobalMeta):

                def __init__(self, *args, **kwargs):
                    pass

        # Subclass's constructor contains arguments without default value will
        # raise an error.
        with pytest.raises(AssertionError):

            class SubClassNoDefault(metaclass=GlobalMeta):

                def __init__(self, a, name='', *args, **kwargs):
                    pass

        class GlobalAccessible(metaclass=GlobalMeta):

            def __init__(self, name=''):
                self.name = name

        assert GlobalAccessible.root.name == 'root'


class TestBaseGlobalAccessible:

    def test_init(self):
        # test get root instance.
        assert BaseGlobalAccessible.root._name == 'root'
        # test create instance by name.
        base_cls = BaseGlobalAccessible('name')
        assert base_cls._name == 'name'

    def test_create_instance(self):
        # SubClass should manage their own `_instance_dict`.
        SubClassA.create_instance('instance_a')
        SubClassB.create_instance('instance_b')
        assert SubClassB._instance_dict != SubClassA._instance_dict

        # test `message_hub` can create by name.
        message_hub = SubClassA.create_instance('name1')
        assert message_hub.instance_name == 'name1'
        # test return root message_hub
        message_hub = SubClassA.create_instance()
        assert message_hub.instance_name == 'root'
        # test default get root `message_hub`.

    def test_get_instance(self):
        message_hub = SubClassA.get_instance()
        assert message_hub.instance_name == 'root'
        # test default get latest `message_hub`.
        message_hub = SubClassA.create_instance('name2')
        message_hub = SubClassA.get_instance(current=True)
        assert message_hub.instance_name == 'name2'
        message_hub.mark = -1
        # test get latest `message_hub` repeatedly.
        message_hub = SubClassA.create_instance('name3')
        assert message_hub.instance_name == 'name3'
        message_hub = SubClassA.get_instance(current=True)
        assert message_hub.instance_name == 'name3'
        # test get root repeatedly.
        message_hub = SubClassA.get_instance()
        assert message_hub.instance_name == 'root'
        # test get name1 repeatedly
        message_hub = SubClassA.get_instance('name2')
        assert message_hub.mark == -1
        # create_instance will raise error if `name` is not specified and
        # given other arguments
        with pytest.raises(ValueError):
            SubClassA.create_instance(a=1)
