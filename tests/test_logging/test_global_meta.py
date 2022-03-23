# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine.logging import ManageMeta, ManageMixin


class SubClassA(ManageMixin):

    def __init__(self, name='', *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class SubClassB(ManageMixin):

    def __init__(self, name='', *args, **kwargs):
        super().__init__(name, *args, **kwargs)


class TestGlobalMeta:

    def test_init(self):
        # Subclass's constructor does not contain name arguments will raise an
        # error.
        with pytest.raises(AssertionError):

            class SubClassNoName1(metaclass=ManageMeta):

                def __init__(self, a, *args, **kwargs):
                    pass

        # Valid subclass.
        class GlobalAccessible1(metaclass=ManageMeta):

            def __init__(self, name):
                self.name = name


class TestManageMixin:

    def test_init(self):
        # test create instance by name.
        base_cls = ManageMixin('name')
        assert base_cls._name == 'name'

    def test_get_instance(self):
        # SubClass should manage their own `_instance_dict`.
        SubClassA.get_instance('instance_a')
        SubClassB.get_instance('instance_b')
        assert SubClassB._instance_dict != SubClassA._instance_dict

        # test `message_hub` can create by name.
        message_hub = SubClassA.get_instance('name1')
        assert message_hub.instance_name == 'name1'
        # no arguments will raise an assertion error.
        with pytest.raises(AssertionError):
            SubClassA.get_instance()

        SubClassA.get_instance('name2')
        message_hub = SubClassA.get_instance(current=True)
        message_hub.mark = -1
        assert message_hub.instance_name == 'name2'
        # test get latest `message_hub` repeatedly.
        message_hub = SubClassA.get_instance('name3')
        assert message_hub.instance_name == 'name3'
        message_hub = SubClassA.get_instance(current=True)
        assert message_hub.instance_name == 'name3'
        # test get name2 repeatedly
        message_hub = SubClassA.get_instance('name2')
        assert message_hub.mark == -1
        #
        with pytest.raises(AssertionError):
            SubClassA.get_instance(a=1)
