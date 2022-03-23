# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import ManageMixin


class DefaultScope(ManageMixin):
    """Scope of current task used to reset the current registry, which can be
    accessed globally.

    Args:
        name (str): Name of default scope for global access. Defaults to ''.
        scope_name (str): Scope of current task. Defaults to 'mmengine'.

    Examples:
        >>> from mmengine import MODELS
        >>> # Define default scope in runner.
        >>> DefaultScope.get_instance('mmdet')
        >>> # Get default scope globally.
        >>> scope = DefaultScope.get_instance('mmdet').scope
        >>> # build model from cfg.
        >>> model = MODELS.build(model_cfg, default_scope=scope)
    """

    def __init__(self, name, scope_name='mmengine'):
        super().__init__(name)
        self._scope_name = scope_name

    @property
    def scope_name(self) -> str:
        """
        Returns:
            str: Get current scope.
        """
        return self._scope_name
