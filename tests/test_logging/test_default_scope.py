# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import DefaultScope


class TestDefaultScope:

    def test_init(self):
        DefaultScope.create_instance('exp_name', scope='mmdet')
        assert DefaultScope.get_instance(current=True).scope == 'mmdet'
        assert DefaultScope.get_instance().scope == 'mmengine'
