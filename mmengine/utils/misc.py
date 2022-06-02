# Copyright (c) OpenMMLab. All rights reserved.
import collections.abc
import functools
import glob
import itertools
import logging
import os.path as osp
import pkgutil
import subprocess
import warnings
from collections import abc
from importlib import import_module
from inspect import getfullargspec
from itertools import repeat
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parrots_wrapper import _BatchNorm, _InstanceNorm


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError(f'Failed to import {imp}')
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def iter_cast(inputs, dst_type, return_type=None):
    """Cast elements of an iterable object into some type.

    Args:
        inputs (Iterable): The input object.
        dst_type (type): Destination type.
        return_type (type, optional): If specified, the output object will be
            converted to this type, otherwise an iterator.

    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, abc.Iterable):
        raise TypeError('inputs must be an iterable object')
    if not isinstance(dst_type, type):
        raise TypeError('"dst_type" must be a valid type')

    out_iterable = map(dst_type, inputs)

    if return_type is None:
        return out_iterable
    else:
        return return_type(out_iterable)


def list_cast(inputs, dst_type):
    """Cast elements of an iterable object into a list of some type.

    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    """Cast elements of an iterable object into a tuple of some type.

    A partial method of :func:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq: Sequence,
              expected_type: Type,
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def slice_list(in_list, lens):
    """Slice a list into several sub lists by a list of given length.

    Args:
        in_list (list): The list to be sliced.
        lens(int or list): The expected length of each out list.

    Returns:
        list: A list of sliced list.
    """
    if isinstance(lens, int):
        assert len(in_list) % lens == 0
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError('"indices" must be an integer or a list of integers')
    elif sum(lens) != len(in_list):
        raise ValueError('sum of lens and list length does not '
                         f'match: {sum(lens)} != {len(in_list)}')
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    """Concatenate a list of list into a single list.

    Args:
        in_list (list): The list of list to be merged.

    Returns:
        list: The concatenated flat list.
    """
    return list(itertools.chain(*in_list))


def check_prerequisites(
        prerequisites,
        checker,
        msg_tmpl='Prerequisites "{}" are required in method "{}" but not '
                 'found, please install them first.'):  # yapf: disable
    """A decorator factory to check if prerequisites are satisfied.

    Args:
        prerequisites (str of list[str]): Prerequisites to be checked.
        checker (callable): The checker method that returns True if a
            prerequisite is meet, False otherwise.
        msg_tmpl (str): The message template with two variables.

    Returns:
        decorator: A specific decorator.
    """

    def wrap(func):

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            requirements = [prerequisites] if isinstance(
                prerequisites, str) else prerequisites
            missing = []
            for item in requirements:
                if not checker(item):
                    missing.append(item)
            if missing:
                print(msg_tmpl.format(', '.join(missing), func.__name__))
                raise RuntimeError('Prerequisites not meet.')
            else:
                return func(*args, **kwargs)

        return wrapped_func

    return wrap


def _check_py_package(package):
    try:
        import_module(package)
    except ImportError:
        return False
    else:
        return True


def _check_executable(cmd):
    if subprocess.call(f'which {cmd}', shell=True) != 0:
        return False
    else:
        return True


def requires_package(prerequisites):
    """A decorator to check if some python packages are installed.

    Example:
        >>> @requires_package('numpy')
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        array([0.])
        >>> @requires_package(['numpy', 'non_package'])
        >>> func(arg1, args):
        >>>     return numpy.zeros(1)
        ImportError
    """
    return check_prerequisites(prerequisites, checker=_check_py_package)


def requires_executable(prerequisites):
    """A decorator to check if some executable files are installed.

    Example:
        >>> @requires_executable('ffmpeg')
        >>> func(arg1, args):
        >>>     print(1)
        1
    """
    return check_prerequisites(prerequisites, checker=_check_executable)


def deprecated_api_warning(name_dict: dict,
                           cls_name: Optional[str] = None) -> Callable:
    """A decorator to check if some arguments are deprecate and try to replace
    deprecate src_arg_name to dst_arg_name.

    Args:
        name_dict(dict):
            key (str): Deprecate argument names.
            val (str): Expected argument names.

    Returns:
        func: New function.
    """

    def api_warning_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get name of the function
            func_name = old_func.__name__
            if cls_name is not None:
                func_name = f'{cls_name}.{func_name}'
            if args:
                arg_names = args_info.args[:len(args)]
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in arg_names:
                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead', DeprecationWarning)
                        arg_names[arg_names.index(src_arg_name)] = dst_arg_name
            if kwargs:
                for src_arg_name, dst_arg_name in name_dict.items():
                    if src_arg_name in kwargs:
                        assert dst_arg_name not in kwargs, (
                            f'The expected behavior is to replace '
                            f'the deprecated key `{src_arg_name}` to '
                            f'new key `{dst_arg_name}`, but got them '
                            f'in the arguments at the same time, which '
                            f'is confusing. `{src_arg_name} will be '
                            f'deprecated in the future, please '
                            f'use `{dst_arg_name}` instead.')

                        warnings.warn(
                            f'"{src_arg_name}" is deprecated in '
                            f'`{func_name}`, please use "{dst_arg_name}" '
                            'instead', DeprecationWarning)
                        kwargs[dst_arg_name] = kwargs.pop(src_arg_name)

            # apply converted arguments to the decorated method
            output = old_func(*args, **kwargs)
            return output

        return new_func

    return api_warning_wrapper


def is_method_overridden(method: str, base_class: type,
                         derived_class: Union[type, Any]) -> bool:
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method


def has_method(obj: object, method: str) -> bool:
    """Check whether the object has a method.

    Args:
        method (str): The method name to check.
        obj (object): The object to check.

    Returns:
        bool: True if the object has the method else False.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


def mmcv_full_available() -> bool:
    """Check whether mmcv-full is installed.

    Returns:
        bool: True if mmcv-full is installed else False.
    """
    try:
        import mmcv  # noqa: F401
    except ImportError:
        return False
    ext_loader = pkgutil.find_loader('mmcv._ext')
    return ext_loader is not None


def is_norm(layer: nn.Module,
            exclude: Optional[Union[type, Tuple[type]]] = None) -> bool:
    """Check if a layer is a normalization layer.

    Args:
        layer (nn.Module): The layer to be checked.
        exclude (type, tuple[type], optional): Types to be excluded.

    Returns:
        bool: Whether the layer is a norm layer.
    """
    if exclude is not None:
        if not isinstance(exclude, tuple):
            exclude = (exclude, )
        if not is_tuple_of(exclude, type):
            raise TypeError(
                f'"exclude" must be either None or type or a tuple of types, '
                f'but got {type(exclude)}: {exclude}')

    if exclude and isinstance(layer, exclude):
        return False

    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)


def tensor2imgs(tensor: torch.Tensor,
                mean: Optional[Tuple[float, float, float]] = None,
                std: Optional[Tuple[float, float, float]] = None,
                to_bgr: bool = True):
    """Convert tensor to 3-channel images or 1-channel gray images.

    Args:
        tensor (torch.Tensor): Tensor that contains multiple images, shape (
            N, C, H, W). :math:`C` can be either 3 or 1. If C is 3, the format
            should be RGB.
        mean (tuple[float], optional): Mean of images. If None,
            (0, 0, 0) will be used for tensor with 3-channel,
            while (0, ) for tensor with 1-channel. Defaults to None.
        std (tuple[float], optional): Standard deviation of images. If None,
            (1, 1, 1) will be used for tensor with 3-channel,
            while (1, ) for tensor with 1-channel. Defaults to None.
        to_bgr (bool): For the tensor with 3 channel, convert its format to
            BGR. For the tensor with 1 channel, it must be False. Defaults to
            True.

    Returns:
        list[np.ndarray]: A list that contains multiple images.
    """

    assert torch.is_tensor(tensor) and tensor.ndim == 4
    channels = tensor.size(1)
    assert channels in [1, 3]
    if mean is None:
        mean = (0, ) * channels
    if std is None:
        std = (1, ) * channels
    assert (channels == len(mean) == len(std) == 3) or \
           (channels == len(mean) == len(std) == 1 and not to_bgr)
    mean = tensor.new_tensor(mean).view(1, -1)
    std = tensor.new_tensor(std).view(1, -1)
    tensor = tensor.permute(0, 2, 3, 1) * std + mean
    imgs = tensor.detach().cpu().numpy()
    if to_bgr and channels == 3:
        imgs = imgs[:, :, :, (2, 1, 0)]  # RGB2BGR
    imgs = [np.ascontiguousarray(img) for img in imgs]
    return imgs


def find_latest_checkpoint(path: str, suffix: str = 'pth'):
    """Find the latest checkpoint from the given path.

    Refer to https://github.com/microsoft/SoftTeacher/blob/main/ssod/utils/patch.py  # noqa: E501

    Args:
        path(str): The path to find checkpoints.
        suffix(str): File extension. Defaults to 'pth'.

    Returns:
        str or None: File path of the latest checkpoint.
    """
    if not osp.exists(path):
        raise FileNotFoundError('{path} does not exist.')

    if osp.exists(osp.join(path, f'latest.{suffix}')):
        return osp.join(path, f'latest.{suffix}')

    checkpoints = glob.glob(osp.join(path, f'*.{suffix}'))
    if len(checkpoints) == 0:
        raise FileNotFoundError(f'checkpoints can not be found in {path}. '
                                'Maybe check the suffix again.')

    latest = -1
    latest_path = None
    for checkpoint in checkpoints:
        count = int(osp.basename(checkpoint).split('_')[-1].split('.')[0])
        if count > latest:
            latest = count
            latest_path = checkpoint

    return latest_path


def has_batch_norm(model: nn.Module) -> bool:
    """Detect whether model has a BatchNormalization layer.

    Args:
        model (nn.Module): training model.

    Returns:
        bool: whether model has a BatchNormalization layer
    """
    if isinstance(model, _BatchNorm):
        return True
    for m in model.children():
        if has_batch_norm(m):
            return True
    return False


def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the images to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need a divisibility of 32. Defaults to 0
        pad_value (int, float): The padding value. Defaults to 0.
        pad_first_dim (bool): Whether to pad first dim of tensor. +++

    Returns:
       Tensor: The 4D-tensor.
    """
    assert isinstance(
        tensor_list,
        list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({
        tensor.ndim
        for tensor in tensor_list
    }) == 1, (f'Expected the dimensions of all tensors must be the same, '
              f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


def detect_anomalous_params(loss: torch.Tensor, model) -> None:
    parameters_in_graph = set()
    visited = set()

    def traverse(grad_fn):
        if grad_fn is None:
            return
        if grad_fn not in visited:
            visited.add(grad_fn)
            if hasattr(grad_fn, 'variable'):
                parameters_in_graph.add(grad_fn.variable)
            parents = grad_fn.next_functions
            if parents is not None:
                for parent in parents:
                    grad_fn = parent[0]
                    traverse(grad_fn)

    traverse(loss.grad_fn)
    from mmengine import MMLogger
    logger = MMLogger.get_current_instance()
    for n, p in model.named_parameters():
        if p not in parameters_in_graph and p.requires_grad:
            logger.log(
                level=logging.ERROR,
                msg=f'{n} with shape {p.size()} is not '
                f'in the computational graph \n')


def merge_dict(*args):
    """Merge all dictionaries into one dictionary.

    If pytorch version >= 1.8, ``merge_dict`` will be wrapped
    by ``torch.fx.wrap``,  which will make ``torch.fx.symbolic_trace`` skip
    trace ``merge_dict``.

    Note:
        If a function needs to be traced by ``torch.fx.symbolic_trace``,
        but inevitably needs to use ``update`` method of ``dict``(``update``
        is not traceable). It should use ``merge_dict`` to replace
        ``xxx.update``.

    Args:
        *args: dictionary needs to be merged.

    Returns:
        dict: Merged dict from args
    """
    output = dict()
    for item in args:
        assert isinstance(
            item,
            dict), (f'all arguments of merge_dict should be a dict, but got '
                    f'{type(item)}')
        output.update(item)
    return output


# torch.fx is only available when pytorch version >= 1.8.
# If the subclass of `BaseModel` has multiple submodules, and each module
# will return a loss dict during training process, i.e., `TwoStageDetector`
# in mmdet. It should use `merge_dict` to get the total loss, rather than
# `loss.update` to keep model traceable.
try:
    import torch.fx

    # make torch.fx skip trace `merge_dict`.
    merge_dict = torch.fx.wrap(merge_dict)

except ImportError:
    warnings.warn('Cannot import torch.fx, merge_dict in a simple')
