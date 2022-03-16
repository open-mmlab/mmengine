# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import logging
from collections import OrderedDict
from typing import Any, Optional
import threading

_lock = threading.RLock()


def _accquire_lock():
    """
    Acquire the module-level lock for serializing access to shared data.

    This should be released with _releaseLock().
    """
    if _lock:
        _lock.acquire()


def _release_lock():
    """
    Release the module-level lock acquired by calling _acquireLock().
    """
    if _lock:
        _lock.release()


class MetaGlobalAccessible(type):
    """The metaclass for global accessible class.

    The subclasses inheriting from ``MetaGlobalAccessible`` will manage their
    own ``_instance_dict`` and root instances. The constructors of subclasses
    must contain an optional ``name`` argument and all other arguments must
    have default values.

    Examples:
        >>> class SubClass1(metaclass=MetaGlobalAccessible):
        >>>     def __init__(self, *args, **kwargs):
        >>>         pass
        AssertionError: <class '__main__.SubClass1'>.__init__ must have the
        name argument.
        >>> class SubClass2(metaclass=MetaGlobalAccessible):
        >>>     def __init__(self, a, name=None, **kwargs):
        >>>         pass
        AssertionError:
        In <class '__main__.SubClass2'>.__init__, Only the name argument is
        allowed to have no default values.
        >>> class SubClass3(metaclass=MetaGlobalAccessible):
        >>>     def __init__(self, name, **kwargs):
        >>>         pass  # Right format
        >>> class SubClass4(metaclass=MetaGlobalAccessible):
        >>>     def __init__(self, a=1, name='', **kwargs):
        >>>         pass  # Right format
    """

    def __init__(cls, *args):
        cls._instance_dict = OrderedDict()
        super().__init__(*args)


class BaseGlobalAccessible(metaclass=MetaGlobalAccessible):
    """``BaseGlobalAccessible`` is the base class for classes that have global
    access requirements.

    The subclasses inheriting from ``BaseGlobalAccessible`` can get their
    global instancees.

    Examples:
        >>> class GlobalAccessible(BaseGlobalAccessible):
        >>>     def __init__(self, name=''):
        >>>         super().__init__(name)
        >>>
        >>> GlobalAccessible.create_instance('name')
        >>> instance_1 = GlobalAccessible.get_instance('name')
        >>> instance_2 = GlobalAccessible.get_instance('name')
        >>> assert id(instance_1) == id(instance_2)

    Args:
        name (str): Name of the instance. Defaults to ''.
    """

    def __init__(self, name: str = '', **kwargs):
        self._name = name

    @classmethod
    def get_instance(cls, name: str = '', current: bool = False, /, **kwargs) -> Any:  # prevent subclass initiate needs `current` aurgument.
        """Get subclass instance by name if the name exists. if name is not
        specified, this method will return latest created instance.

        Examples
            >>> instance = GlobalAccessible.create_instance('name1')
            >>> instance = GlobalAccessible.get_instance('name1')
            >>> instance.instance_name
            name1
            >>> instance = GlobalAccessible.create_instance('name2')
            >>> instance = GlobalAccessible.get_instance(current=True)
            >>> instance.instance_name
            name2
            >>> instance = GlobalAccessible.get_instance()
            >>> instance.instance_name  # get root instance
            root
            >>> instance = GlobalAccessible.get_instance('name3') # error
            AssertionError: Cannot get <class '__main__.GlobalAccessible'> by
            name: name3, please make sure you have created it

        Args:
            name (str): Name of instance. Defaults to ''.
            current(bool): Whether to return the latest created instance or
                the root instance, if name is not spicified. Defaults to False.

        Returns:
            object: Corresponding name instance, the latest instance, or root
            instance.
        """
        instance_dict = cls._instance_dict
        # Get the instance by name.
        if name:
            if name not in instance_dict:
                cls(name, **kwargs)
                return instance_dict[name]
            else:
                assert current
        # Get latest instantiated instance or root instance.
        else:
            if current:
                current_name = next(iter(reversed(cls._instance_dict)))
                assert current_name, f'Before calling {cls}.get_instance, ' \
                                     'you should call create_instance.'
                return cls._instance_dict[current_name]
            else:
                return cls.root

    @property
    def instance_name(self) -> Optional[str]:
        """Get the name of instance.

        Returns:
            str: Name of instance.
        """
        return self._name
