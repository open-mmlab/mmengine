# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine.utils import ManagerMeta, ManagerMixin


class SubClassA(ManagerMixin):

    def __init__(self, name='', *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class SubClassB(ManagerMixin):

    def __init__(self, name='', *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class TestGlobalMeta:

    def test_init(self):
        # Subclass's constructor does not contain name arguments will raise an
        # error.
        with pytest.raises(AssertionError):

            class SubClassNoName1(metaclass=ManagerMeta):

                def __init__(self, a, *args, **kwargs):
                    pass

        # Valid subclass.
        class GlobalAccessible1(metaclass=ManagerMeta):

            def __init__(self, name):
                self.name = name


class TestManagerMixin:

    def test_init(self):
        # test create instance by name.
        base_cls = ManagerMixin('name')
        assert base_cls.instance_name == 'name'

    def test_get_instance(self):
        # SubClass should manage their own `_instance_dict`.
        with pytest.raises(RuntimeError):
            SubClassA.get_current_instance()
        SubClassA.get_instance('instance_a')
        SubClassB.get_instance('instance_b')
        assert SubClassB._instance_dict != SubClassA._instance_dict

        # Test `message_hub` can create by name.
        message_hub = SubClassA.get_instance('name1')
        assert message_hub.instance_name == 'name1'
        # No arguments will raise an assertion error.

        SubClassA.get_instance('name2')
        message_hub = SubClassA.get_current_instance()
        message_hub.mark = -1
        assert message_hub.instance_name == 'name2'
        # Test get latest `message_hub` repeatedly.
        message_hub = SubClassA.get_instance('name3')
        assert message_hub.instance_name == 'name3'
        message_hub = SubClassA.get_current_instance()
        assert message_hub.instance_name == 'name3'
        # Test get name2 repeatedly.
        message_hub = SubClassA.get_instance('name2')
        assert message_hub.mark == -1
        # Non-string instance name will raise `AssertionError`.
        with pytest.raises(AssertionError):
            SubClassA.get_instance(name=1)
        # `get_instance` should not accept other arguments if corresponding
        # instance has been created.
        with pytest.warns(UserWarning):
            SubClassA.get_instance('name2', a=1, b=2)
