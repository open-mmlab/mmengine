# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import BaseGlobalAccessible


class DefaultScope(BaseGlobalAccessible):
    """Scope of current task used to reset the current registry, which can be
    accessed globally.

    Args:
        name (str): Name of default scope for global access. Defaults to ''.
        scope (str): Scope of current task. Defaults to 'mmengine'.

    Examples:
        >>> from mmengine import MODELS
        >>> # Define default scope in runner.
        >>> DefaultScope.create_instance('mmdet')
        >>> # Get default scope globally.
        >>> scope = DefaultScope.get_instance('mmdet').scope
        >>> model = MODELS.build(default_scope=scope)
    """

    def __init__(self, name, scope='mmengine'):
        super().__init__(name)
        self._scope = scope

    @property
    def scope(self) -> str:
        """
        Returns:
            str: Get current scope.
        """
        return self._scope
