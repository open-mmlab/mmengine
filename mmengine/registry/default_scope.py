# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmengine.utils.manager import ManagerMixin, _accquire_lock, _release_lock


class DefaultScope(ManagerMixin):
    """Scope of current task used to reset the current registry, which can be
    accessed globally.

    Args:
        name (str): Name of default scope for global access.
        scope_name (str): Scope of current task.

    Examples:
        >>> from mmengine import MODELS
        >>> # Define default scope in runner.
        >>> DefaultScope.get_instance('mmdet')
        >>> # Get default scope globally.
        >>> scope = DefaultScope.get_instance('mmdet').scope
        >>> # build model from cfg.
        >>> model = MODELS.build(model_cfg, default_scope=scope)
    """

    def __init__(self, name: str, scope_name: str):
        super().__init__(name)
        self._scope_name = scope_name

    @property
    def scope_name(self) -> str:
        """
        Returns:
            str: Get current scope.
        """
        return self._scope_name

    @classmethod
    def get_current_instance(cls) -> Optional['DefaultScope']:
        """Get latest created default scope.

        Since default_scope is an optional argument for ``Registry.build``.
        ``get_current_instance`` should return ``None`` if there is no
        ``DefaultScope`` created.

        Examples:
            >>> default_scope = DefaultScope.get_current_instance()
            >>> # There is no `DefaultScope` created yet,
            >>> # `get_current_instance` return `None`.
            >>> default_scope = DefaultScope.get_instance(
            >>>     'instance_name', scope_name='mmengine')
            >>> default_scope.scope_name
            mmengine
            >>> default_scope = DefaultScope.get_current_instance()
            >>> default_scope.scope_name
            mmengine

        Returns:
            Optional[DefaultScope]: Return None If there has not been
            ``DefaultScope`` instance created yet, otherwise return the
            latest created DefaultScope instance.
        """
        _accquire_lock()
        if cls._instance_dict:
            instance = super(DefaultScope, cls).get_current_instance()
        else:
            instance = None
        _release_lock()
        return instance
