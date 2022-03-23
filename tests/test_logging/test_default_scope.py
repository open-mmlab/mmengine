# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import DefaultScope


class TestDefaultScope:

    def test_scope(self):
        default_scope = DefaultScope.get_instance('name1', scope_name='mmdet')
        assert default_scope.scope_name == 'mmdet'
        default_scope = DefaultScope.get_instance('name2')
        assert default_scope.scope_name == 'mmengine'
