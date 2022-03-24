# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import threading
from collections import OrderedDict
from typing import Any, Optional

_lock = threading.RLock()


def _accquire_lock() -> None:
    """Acquire the module-level lock for serializing access to shared data.

    This should be released with _release_lock().
    """
    if _lock:
        _lock.acquire()


def _release_lock() -> None:
    """Release the module-level lock acquired by calling _accquire_lock()."""
    if _lock:
        _lock.release()


class ManagerMeta(type):
    """The metaclass for global accessible class.

    The subclasses inheriting from ``ManagerMeta`` will manage their
    own ``_instance_dict`` and root instances. The constructors of subclasses
    must contain an optional ``name`` argument.

    Examples:
        >>> class SubClass1(metaclass=ManagerMeta):
        >>>     def __init__(self, *args, **kwargs):
        >>>         pass
        AssertionError: <class '__main__.SubClass1'>.__init__ must have the
        name argument.
        >>> class SubClass2(metaclass=ManagerMeta):
        >>>     def __init__(self, name):
        >>>         pass
        >>> # valid format.
    """

    def __init__(cls, *args):
        cls._instance_dict = OrderedDict()
        params = inspect.getfullargspec(cls)
        params_names = params[0] if params[0] else []
        assert 'name' in params_names, (f'{cls} must have the `name` argument')
        super().__init__(*args)


class ManagerMixin(metaclass=ManagerMeta):
    """``ManagerMixin`` is the base class for classes that have global access
    requirements.

    The subclasses inheriting from ``ManagerMixin`` can get their
    global instancees.

    Examples:
        >>> class GlobalAccessible(ManagerMixin):
        >>>     def __init__(self, name=''):
        >>>         super().__init__(name)
        >>>
        >>> GlobalAccessible.get_instance('name')
        >>> instance_1 = GlobalAccessible.get_instance('name')
        >>> instance_2 = GlobalAccessible.get_instance('name')
        >>> assert id(instance_1) == id(instance_2)

    Args:
        name (str): Name of the instance. Defaults to ''.
    """

    def __init__(self, name: str = '', **kwargs):
        self._name = name

    @classmethod
    def get_instance(cls, name: str = '', current: bool = False, **kwargs)\
            -> Any:
        """Get subclass instance by name if the name exists. if name is not
        specified, this method will return latest created instance.

        Examples
            >>> instance = GlobalAccessible.get_instance(current=True)
            AssertionError: At least one of name and current needs to be set
            >>> instance = GlobalAccessible.get_instance('name1')
            >>> instance.instance_name
            name1
            >>> instance = GlobalAccessible.get_instance('name2')
            >>> instance = GlobalAccessible.get_instance(current=True)
            >>> instance.instance_name
            name2

        Args:
            name (str): Name of instance. Defaults to ''.
            current(bool): Whether to return the latest created instance, if
                name is not spicified. Defaults to False.

        Returns:
            object: Corresponding name instance, the latest instance, or root
            instance.
        """
        _accquire_lock()
        instance_dict = cls._instance_dict
        # Get the instance by name.
        if name:
            assert not current, ('`current` should not be True if `name` is '
                                 'specified.')
            if name not in instance_dict:
                instance = cls(name=name, **kwargs)
                instance_dict[name] = instance
        # Get latest instantiated instance or root instance.
        else:
            assert current, 'At least one of name and current needs to be set'
            name = next(iter(reversed(cls._instance_dict)))
            assert name, (f'Before calling {cls}.get_instance, you should '
                          'call get_instance.')
        _release_lock()
        return instance_dict[name]

    @classmethod
    def check_instance_created(cls, name: str) -> bool:
        """Check if the instance was created.

        Args:
            name: Name of instance.

        Returns:
            bool: Whether instance is created.
        """
        return name in cls._instance_dict

    @property
    def instance_name(self) -> Optional[str]:
        """Get the name of instance.

        Returns:
            str: Name of instance.
        """
        return self._name
