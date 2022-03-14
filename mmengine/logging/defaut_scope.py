# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import BaseGlobalAccessible


class DefaultScope(BaseGlobalAccessible):

    def __init__(self, name, scope='mmengine'):
        super().__init__(name)
        self._scope = scope

    @property
    def scope(self):
        return self._scope
