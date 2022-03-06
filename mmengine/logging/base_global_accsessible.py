# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from collections import OrderedDict
from typing import Any, Optional


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
        params = inspect.getfullargspec(cls)
        # `inspect.getfullargspec` returns a tuple includes `(args, varargs,
        # varkw, defaults, kwonlyargs, kwonlydefaults, annotations)`.
        # To make sure `cls(name='root')` can be implemented, the
        # `args` and `defaults` should be checked.
        params_names = params[0] if params[0] else []
        default_params = params[3] if params[3] else []
        assert 'name' in params_names, f'{cls}.__init__ must have the name ' \
                                       'argument'
        if len(default_params) == len(params_names) - 2 and 'name' != \
                params[0][1]:
            raise AssertionError(f'In {cls}.__init__, Only the name argument '
                                 'is allowed to have no default values.')
        if len(default_params) < len(params_names) - 2:
            raise AssertionError('Besides name, the arguments of the '
                                 f'{cls}.__init__ must have default values')
        cls.root = cls(name='root')
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
    def create_instance(cls, name: str = '', **kwargs) -> Any:
        """Create subclass instance by name, and subclass cannot create
        instances with duplicated names. The created instance will be stored in
        ``cls._instance_dict``, and can be accessed by ``get_instance``.

        Examples:
            >>> instance_1 = GlobalAccessible.create_instance('name')
            >>> instance_2 = GlobalAccessible.create_instance('name')
            AssertionError: <class '__main__.GlobalAccessible'> cannot be
            created by name twice.
            >>> root_instance = GlobalAccessible.create_instance()
            >>> root_instance.instance_name  # get default root instance
            root

        Args:
            name (str): Name of instance. Defaults to ''.

        Returns:
            object: Subclass instance.
        """
        instance_dict = cls._instance_dict
        # Create instance and fill the instance in the `instance_dict`.
        if name:
            assert name not in instance_dict, f'{cls} cannot be created by ' \
                                              f'{name} twice.'
            instance = cls(name=name, **kwargs)
            instance_dict[name] = instance
            return instance
        # Get default root instance.
        else:
            if kwargs:
                raise ValueError('If name is not specified, create_instance '
                                 f'will return root {cls} and cannot accept '
                                 f'any arguments, but got kwargs: {kwargs}')
            return cls.root

    @classmethod
    def get_instance(cls, name: str = '', current: bool = False) -> Any:
        """Get subclass instance by name if the name exists. if name is not
        specified, this method will return latest created instance of root
        instance.

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
            assert name in instance_dict, \
                f'Cannot get {cls} by name: {name}, please make sure you ' \
                'have created it'
            return instance_dict[name]
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
