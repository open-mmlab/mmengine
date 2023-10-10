# Copyright (c) OpenMMLab. All rights reserved.
from collections import namedtuple
from importlib import import_module

import numpy as np
import pytest

from mmengine import MMLogger
from mmengine.utils import is_installed
# yapf: disable
from mmengine.utils.misc import (apply_to, concat_list, deprecated_api_warning,
                                 deprecated_function, get_object_from_string,
                                 has_method, import_modules_from_strings,
                                 is_list_of, is_method_overridden, is_seq_of,
                                 is_tuple_of, iter_cast, list_cast,
                                 requires_executable, requires_package,
                                 slice_list, to_1tuple, to_2tuple, to_3tuple,
                                 to_4tuple, to_ntuple, tuple_cast)

# yapf: enable


def test_to_ntuple():
    single_number = 2
    assert to_1tuple(single_number) == (single_number, )
    assert to_2tuple(single_number) == (single_number, single_number)
    assert to_3tuple(single_number) == (single_number, single_number,
                                        single_number)
    assert to_4tuple(single_number) == (single_number, single_number,
                                        single_number, single_number)
    assert to_ntuple(5)(single_number) == (single_number, single_number,
                                           single_number, single_number,
                                           single_number)
    assert to_ntuple(6)(single_number) == (single_number, single_number,
                                           single_number, single_number,
                                           single_number, single_number)


def test_iter_cast():
    assert list_cast([1, 2, 3], int) == [1, 2, 3]
    assert list_cast(['1.1', 2, '3'], float) == [1.1, 2.0, 3.0]
    assert list_cast([1, 2, 3], str) == ['1', '2', '3']
    assert tuple_cast((1, 2, 3), str) == ('1', '2', '3')
    assert next(iter_cast([1, 2, 3], str)) == '1'
    with pytest.raises(TypeError):
        iter_cast([1, 2, 3], '')
    with pytest.raises(TypeError):
        iter_cast(1, str)


def test_is_seq_of():
    assert is_seq_of([1.0, 2.0, 3.0], float)
    assert is_seq_of([(1, ), (2, ), (3, )], tuple)
    assert is_seq_of((1.0, 2.0, 3.0), float)
    assert is_list_of([1.0, 2.0, 3.0], float)
    assert not is_seq_of((1.0, 2.0, 3.0), float, seq_type=list)
    assert not is_tuple_of([1.0, 2.0, 3.0], float)
    assert not is_seq_of([1.0, 2, 3], int)
    assert not is_seq_of((1.0, 2, 3), int)


def test_slice_list():
    in_list = [1, 2, 3, 4, 5, 6]
    assert slice_list(in_list, [1, 2, 3]) == [[1], [2, 3], [4, 5, 6]]
    assert slice_list(in_list, [len(in_list)]) == [in_list]
    with pytest.raises(TypeError):
        slice_list(in_list, 2.0)
    with pytest.raises(ValueError):
        slice_list(in_list, [1, 2])


def test_concat_list():
    assert concat_list([[1, 2]]) == [1, 2]
    assert concat_list([[1, 2], [3, 4, 5], [6]]) == [1, 2, 3, 4, 5, 6]


def test_requires_package(capsys):

    @requires_package('nnn')
    def func_a():
        pass

    @requires_package(['numpy', 'n1', 'n2'])
    def func_b():
        pass

    @requires_package('numpy')
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "nnn" are required in method "func_a" but '
                   'not found, please install them first.\n')

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == (
        'Prerequisites "n1, n2" are required in method "func_b" but not found,'
        ' please install them first.\n')

    assert func_c() == 1


def test_requires_executable(capsys):

    @requires_executable('nnn')
    def func_a():
        pass

    @requires_executable(['ls', 'n1', 'n2'])
    def func_b():
        pass

    @requires_executable('mv')
    def func_c():
        return 1

    with pytest.raises(RuntimeError):
        func_a()
    out, _ = capsys.readouterr()
    assert out == ('Prerequisites "nnn" are required in method "func_a" but '
                   'not found, please install them first.\n')

    with pytest.raises(RuntimeError):
        func_b()
    out, _ = capsys.readouterr()
    assert out == (
        'Prerequisites "n1, n2" are required in method "func_b" but not found,'
        ' please install them first.\n')

    assert func_c() == 1


def test_import_modules_from_strings():
    # multiple imports
    import os.path as osp_
    import sys as sys_
    osp, sys = import_modules_from_strings(['os.path', 'sys'])
    assert osp == osp_
    assert sys == sys_

    # single imports
    osp = import_modules_from_strings('os.path')
    assert osp == osp_
    # No imports
    assert import_modules_from_strings(None) is None
    assert import_modules_from_strings([]) is None
    assert import_modules_from_strings('') is None
    # Unsupported types
    with pytest.raises(TypeError):
        import_modules_from_strings(1)
    with pytest.raises(TypeError):
        import_modules_from_strings([1])
    # Failed imports
    with pytest.raises(ImportError):
        import_modules_from_strings('_not_implemented_module')
    with pytest.warns(UserWarning):
        imported = import_modules_from_strings(
            '_not_implemented_module', allow_failed_imports=True)
        assert imported is None
    with pytest.warns(UserWarning):
        imported = import_modules_from_strings(['os.path', '_not_implemented'],
                                               allow_failed_imports=True)
        assert imported[0] == osp
        assert imported[1] is None


def test_is_method_overridden():

    class Base:

        def foo1():
            pass

        def foo2():
            pass

    class Sub(Base):

        def foo1():
            pass

    # test passing sub class directly
    assert is_method_overridden('foo1', Base, Sub)
    assert not is_method_overridden('foo2', Base, Sub)

    # test passing instance of sub class
    sub_instance = Sub()
    assert is_method_overridden('foo1', Base, sub_instance)
    assert not is_method_overridden('foo2', Base, sub_instance)

    # base_class should be a class, not instance
    base_instance = Base()
    with pytest.raises(AssertionError):
        is_method_overridden('foo1', base_instance, sub_instance)


def test_has_method():

    class Foo:

        def __init__(self, name):
            self.name = name

        def print_name(self):
            print(self.name)

    foo = Foo('foo')
    assert not has_method(foo, 'name')
    assert has_method(foo, 'print_name')


def test_deprecated_api_warning():

    @deprecated_api_warning(name_dict=dict(old_key='new_key'))
    def dummy_func(new_key=1):
        return new_key

    # replace `old_key` to `new_key`
    assert dummy_func(old_key=2) == 2

    # The expected behavior is to replace the
    # deprecated key `old_key` to `new_key`,
    # but got them in the arguments at the same time
    with pytest.raises(AssertionError):
        dummy_func(old_key=1, new_key=2)


def test_deprecated_function():

    @deprecated_function('0.2.0', '0.3.0', 'toy instruction')
    def deprecated_demo(arg1: int, arg2: int) -> tuple:
        """This is a long summary. This is a long summary. This is a long
        summary. This is a long summary.

        Args:
            arg1 (int): Long description with a line break. Long description
                with a line break.
            arg2 (int): short description.

        Returns:
            Long description without a line break. Long description without
            a line break.
        """

        return arg1, arg2

    MMLogger.get_instance('test_deprecated_function')
    deprecated_demo(1, 2)
    # out, _ = capsys.readouterr()
    # assert "'test_misc.deprecated_demo' is deprecated" in out
    assert (1, 2) == deprecated_demo(1, 2)

    expected_docstring = \
    """.. deprecated:: 0.2.0
    Deprecated and will be removed in version 0.3.0.
    Please toy instruction.


    This is a long summary. This is a long summary. This is a long
    summary. This is a long summary.

    Args:
        arg1 (int): Long description with a line break. Long description
            with a line break.
        arg2 (int): short description.

    Returns:
        Long description without a line break. Long description without
        a line break.
    """  # noqa: E122
    assert expected_docstring.strip(' ') == deprecated_demo.__doc__
    MMLogger._instance_dict.clear()

    # Test with short summary without args.
    @deprecated_function('0.2.0', '0.3.0', 'toy instruction')
    def deprecated_demo1():
        """Short summary."""

    expected_docstring = \
    """.. deprecated:: 0.2.0
    Deprecated and will be removed in version 0.3.0.
    Please toy instruction.


    Short summary."""  # noqa: E122
    assert expected_docstring.strip(' ') == deprecated_demo1.__doc__


@pytest.mark.skipif(not is_installed('torch'), reason='tests requires torch')
def test_apply_to():
    import torch

    # Test only apply `+1` to int object.
    data = dict(a=1, b=2.0)
    result = apply_to(data, lambda x: isinstance(x, int), lambda x: x + 1)
    assert result == dict(a=2, b=2.0)

    # Test with nested data
    data = dict(a=[dict(c=1)], b=2.0)
    result = apply_to(data, lambda x: isinstance(x, int), lambda x: x + 1)
    assert result == dict(a=[dict(c=2)], b=2.0)

    # Tensor to numpy
    data = dict(a=[dict(c=torch.tensor(1))], b=torch.tensor(2))
    result = apply_to(data, lambda x: isinstance(x, torch.Tensor),
                      lambda x: x.numpy())
    assert isinstance(result['b'], np.ndarray)
    assert isinstance(result['a'][0]['c'], np.ndarray)

    # Tuple and convert string
    data = (1, dict(a=[dict(b=2.0)]), 'test')
    result = apply_to(
        data, lambda x: isinstance(x, int) or x == 'test',
        lambda x: torch.Tensor(x) if isinstance(x, int) else 'train')
    assert isinstance(result, tuple)
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1]['a'][0]['b'], float)
    assert result[2] == 'train'

    # Named Tuple
    dataclass = namedtuple('Data', ['a', 'b'])
    data = dataclass('test', dict(a=[dict(c=1)], b=2.0))
    result = apply_to(
        data, lambda x: isinstance(x, int) or x == 'test',
        lambda x: torch.Tensor(x) if isinstance(x, int) else 'train')
    assert isinstance(result, dataclass)
    assert result[0] == 'train'
    assert isinstance(result.b['a'][0]['c'], torch.Tensor)
    assert isinstance(result.b['b'], float)


def test_locate():
    assert get_object_from_string('a.b.c') is None
    config_module = import_module('mmengine.config')
    assert get_object_from_string('mmengine.config') is config_module
    assert get_object_from_string(
        'mmengine.config.Config') is config_module.Config
    assert get_object_from_string('mmengine.config.Config.fromfile') is \
        config_module.Config.fromfile
