# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager

import torch

from mmengine.utils import TORCH_VERSION, digit_version


@contextmanager
def autocast(enabled: bool = True, **kwargs):
    """A wrapper of ``torch.autocast`` and ``toch.cuda.amp.autocast``.

    Pytorch 1.6.0 provide ``torch.cuda.amp.autocast`` for running in
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
        enabled (bool): Whether autocasting should be enabled in the region.
            Defaults to True.
        kwargs (dict): Arguments of torch.autocast except for ``enabled``.
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
        assert not kwargs, (
            f'autocast under pytorch {TORCH_VERSION} only accept `enabled` '
            'arguments.')
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

    elif (digit_version('1.11.0') > digit_version(TORCH_VERSION) >=
          digit_version('1.10.0')):
        if torch.cuda.is_available():
            kwargs.setdefault('device_type', 'cuda')
        else:
            kwargs.setdefault('device_type', 'cpu')
            # torch.autocast only support `dtype=torch.bfloat16` in
            # pytorch 1.10
            kwargs.setdefault('dtype', torch.bfloat16)

        with torch.autocast(enabled=enabled, **kwargs):
            yield

    elif digit_version(TORCH_VERSION) >= digit_version('1.11.0'):
        if torch.cuda.is_available():
            kwargs.setdefault('device_type', 'cuda')
        else:
            kwargs.setdefault('device_type', 'cpu')

        with torch.autocast(enabled=enabled, **kwargs):
            yield
