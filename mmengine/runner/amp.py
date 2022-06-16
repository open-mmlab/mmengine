# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from typing import Optional

import torch

from mmengine.utils import TORCH_VERSION, digit_version


@contextmanager
def auto_cast(device_type: str = 'auto',
              enabled: bool = True,
              dtype: Optional[torch.dtype] = None,
              cache_enabled: bool = True):
    """A wrapper of ``torch.autocast`` and ``toch.cuda.amp.autocast``.

    Pytorch 1.6.0 provide ``torch.cuda.amp.autocast`` for running in
    mixed precision , and update it to ``torch.autocast`` in 1.10.0.
    Both interfaces have different arguments, and ``torch.autocast``
    support running with cpu additionally.

    ``auto_cast`` is a wrapper of both them, dealing with compatibility issues
    between different versions of pytorch, and provide unified interface for
    easy to use.

    Note:
        ``auto_cast`` requires pytorch version >= 1.5.0. If pytorch version
        <= 1.10.0 and cuda is not available, it will raise an error with
        ``enabled=True``, since ``torch.cuda.amp.autocast`` only support cuda
        mode.

    Examples:
         >>> # case1: 1.10 > Pytorch version >= 1.5.0
         >>> with auto_cast():
         >>>    # run in mixed precision context
         >>>    pass
         >>> with auto_cast(device_type='cpu')::
         >>>    # raise error, torch.cuda.amp.autocast only support cuda mode.
         >>>    pass
         >>> # case2: Pytorch version >= 1.10.0
         >>> with auto_cast():
         >>>    # default cuda mixed precision context
         >>>    pass
         >>> with auto_cast(device_type='cpu'):
         >>>    # cpu mixed precision context
         >>>    pass
         >>> with auto_cast(
         >>>     device_type='cuda', enabled=True, cache_enabled=True):
         >>>    # enable precision context with more specific arguments.
         >>>    pass

    Args:
        device_type (str):  Whether to use 'cuda' or 'cpu' device. Defaults
            to 'auto'.
        enabled (bool): Whether autocasting should be enabled in the region.
            Defaults to True.
        dtype (torch.dtype, optional):  Use torch.float16 or torch.bfloat16.
        cache_enabled(bool, optional, default=True): Whether the weight cache
            inside autocast should be enabled. Defaults to True.
    """
    # If `enabled` is True, enable an empty context and all calculations
    # are performed under fp32.
    assert digit_version(TORCH_VERSION) >= digit_version('1.5.0'), (
        'The minimum pytorch version requirements of mmengine is 1.5.0, but '
        f'got {TORCH_VERSION}')

    if digit_version('1.5.0') <= digit_version(TORCH_VERSION)\
            < digit_version('1.10.0'):
        # If pytorch version is between 1.5.0 and 1.10.0, the default value of
        # dtype for `torch.cuda.amp.autocast` is torch.float16.
        if dtype is None:
            dtype = torch.float16

        assert device_type in ('auto', 'cuda'), (
            f'{TORCH_VERSION} only support using ``torch.cuda.amp.autocast` '
            'with cuda`')
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(
                    enabled=enabled, dtype=dtype, cache_enabled=cache_enabled):
                yield
        else:
            if not enabled:
                yield
            else:
                raise RuntimeError(
                    'If pytorch versions is between 1.5.0 and 1.10, '
                    '`auto_cast` is only available in gpu mode')

    elif digit_version(TORCH_VERSION) >= digit_version('1.10.0'):
        if device_type == 'auto':
            if torch.cuda.is_available():
                device_type = 'cuda'
            else:
                device_type = 'cpu'
        with torch.autocast(
                device_type=device_type,
                enabled=enabled,
                dtype=dtype,
                cache_enabled=cache_enabled):
            yield
