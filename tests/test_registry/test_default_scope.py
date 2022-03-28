# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

from mmengine.registry import DefaultScope


class TestDefaultScope:

    def test_scope(self):
        default_scope = DefaultScope.get_instance('name1', scope_name='mmdet')
        assert default_scope.scope_name == 'mmdet'
        default_scope = DefaultScope.get_instance('name2')
        assert default_scope.scope_name == 'mmengine'

    def test_get_current_instance(self):
        DefaultScope._intance_dict = OrderedDict()
        assert DefaultScope.get_current_instance() is None
        DefaultScope.get_instance('instance_name', scope_name='mmengine')
        default_scope = DefaultScope.get_current_instance()
        assert default_scope.scope_name == 'mmengine'
