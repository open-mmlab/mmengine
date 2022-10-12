# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import pytest

from mmengine.registry import DefaultScope


class TestDefaultScope:

    def test_scope(self):
        default_scope = DefaultScope.get_instance('name1', scope_name='mmdet')
        assert default_scope.scope_name == 'mmdet'
        # `DefaultScope.get_instance` must have `scope_name` argument.
        with pytest.raises(TypeError):
            DefaultScope.get_instance('name2')

    def test_get_current_instance(self):
        DefaultScope._instance_dict = OrderedDict()
        assert DefaultScope.get_current_instance() is None
        DefaultScope.get_instance('instance_name', scope_name='mmengine')
        default_scope = DefaultScope.get_current_instance()
        assert default_scope.scope_name == 'mmengine'

    def test_overwrite_default_scope(self):
        origin_scope = DefaultScope.get_instance(
            'test_overwrite_default_scope', scope_name='origin_scope')
        with DefaultScope.overwrite_default_scope(scope_name=None):
            assert DefaultScope.get_current_instance(
            ).scope_name == 'origin_scope'
        with DefaultScope.overwrite_default_scope(scope_name='test_overwrite'):
            assert DefaultScope.get_current_instance(
            ).scope_name == 'test_overwrite'
        assert DefaultScope.get_current_instance(
        ).scope_name == origin_scope.scope_name == 'origin_scope'

        # Test overwrite default scope immediately.
        # Test sequentially overwrite.
        with DefaultScope.overwrite_default_scope(scope_name='test_overwrite'):
            pass
        with DefaultScope.overwrite_default_scope(scope_name='test_overwrite'):
            pass

        # Test nested overwrite.
        with DefaultScope.overwrite_default_scope(scope_name='test_overwrite'):
            with DefaultScope.overwrite_default_scope(
                    scope_name='test_overwrite'):
                pass
