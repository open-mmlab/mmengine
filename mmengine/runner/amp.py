# Copyright (c) OpenMMLab. All rights reserved.
import logging
from contextlib import contextmanager
from typing import Optional

import torch

from mmengine import print_log
from mmengine.utils import TORCH_VERSION, digit_version


@contextmanager
def autocast(device_type: Optional[str] = None,
             dtype: Optional[torch.dtype] = None,
             enabled: bool = True,
             cache_enabled: Optional[bool] = None):
    """A wrapper of ``torch.autocast`` and ``toch.cuda.amp.autocast``.

    Pytorch 1.5.0 provide ``torch.cuda.amp.autocast`` for running in
    mixed precision , and update it to ``torch.autocast`` in 1.10.0.
    Both interfaces have different arguments, and ``torch.autocast``
    support running with cpu additionally.

    This function provides a unified interface by wrapping
    ``torch.autocast`` and ``torch.cuda.amp.autocast``, which resolves the
    compatibility issues that ``torch.cuda.amp.autocast`` does not support
    running mixed precision with cpu, and both contexts have different
    arguments. We suggest users using this function in the code
    to achieve maximized compatibility of different PyTorch versions.

    Note:
        ``autocast`` requires pytorch version >= 1.5.0. If pytorch version
        <= 1.10.0 and cuda is not available, it will raise an error with
        ``enabled=True``, since ``torch.cuda.amp.autocast`` only support cuda
        mode.

    Examples:
         >>> # case1: 1.10 > Pytorch version >= 1.5.0
         >>> with autocast():
         >>>    # run in mixed precision context
         >>>    pass
         >>> with autocast(device_type='cpu')::
         >>>    # raise error, torch.cuda.amp.autocast only support cuda mode.
         >>>    pass
         >>> # case2: Pytorch version >= 1.10.0
         >>> with autocast():
         >>>    # default cuda mixed precision context
         >>>    pass
         >>> with autocast(device_type='cpu'):
         >>>    # cpu mixed precision context
         >>>    pass
         >>> with autocast(
         >>>     device_type='cuda', enabled=True, cache_enabled=True):
         >>>    # enable precision context with more specific arguments.
         >>>    pass

    Args:
        device_type (str, required):  Whether to use 'cuda' or 'cpu' device.
        enabled(bool):  Whether autocasting should be enabled in the region.
            Defaults to True
        dtype (torch_dtype, optional):  Whether to use ``torch.float16`` or
            ``torch.bfloat16``.
        cache_enabled(bool, optional):  Whether the weight cache inside
            autocast should be enabled.
    """
    # If `enabled` is True, enable an empty context and all calculations
    # are performed under fp32.
    assert digit_version(TORCH_VERSION) >= digit_version('1.5.0'), (
        'The minimum pytorch version requirements of mmengine is 1.5.0, but '
        f'got {TORCH_VERSION}')

    if (digit_version('1.5.0') <= digit_version(TORCH_VERSION) <
            digit_version('1.10.0')):
        # If pytorch version is between 1.5.0 and 1.10.0, the default value of
        # dtype for `torch.cuda.amp.autocast` is torch.float16.
        assert device_type == 'cuda' or device_type is None, (
            'Pytorch version under 1.5.0 only supports running automatic '
            'mixed training with cuda')
        if dtype is not None or cache_enabled is not None:
            print_log(
                f'{dtype} and {device_type} will not work for '
                '`autocast` since your Pytorch version: '
                f'{TORCH_VERSION} <= 1.10.0',
                logger='current',
                level=logging.WARNING)

        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(enabled=enabled):
                yield
        else:
            if not enabled:
                yield
            else:
                raise RuntimeError(
                    'If pytorch versions is between 1.5.0 and 1.10, '
                    '`autocast` is only available in gpu mode')

    else:
        if torch.cuda.is_available():
            device_type = 'cuda' if device_type is None else device_type
        else:
            device_type = 'cpu' if device_type is None else device_type

        if digit_version(TORCH_VERSION) < digit_version('1.11.0'):
            if dtype is not None and dtype != torch.bfloat16:
                print_log(
                    f'{dtype} must be `torch.bfloat16` with Pytorch '
                    f'version: {TORCH_VERSION}',
                    logger='current',
                    level=logging.WARNING)
            dtype = torch.bfloat16
        with torch.autocast(
                device_type=device_type,
                enabled=enabled,
                dtype=dtype,
                cache_enabled=cache_enabled):
            yield
