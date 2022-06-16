# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager

import torch

from mmengine.utils import TORCH_VERSION, digit_version


@contextmanager
def auto_cast(enabled: bool = True, **kwargs):
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
        enabled (bool): Whether autocasting should be enabled in the region.
            Defaults to True.
        kwargs (dict): Arguments of torch.autocast except for ``enabled``.
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
        assert not kwargs, (
            f'auto_cast under pytorch {TORCH_VERSION} only accept `enabled` '
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
                    '`auto_cast` is only available in gpu mode')

    elif digit_version(TORCH_VERSION) >= digit_version('1.10.0'):
        if torch.cuda.is_available():
            kwargs.setdefault('device_type', 'cuda')
        else:
            kwargs.setdefault('device_type', 'cpu')

        with torch.autocast(enabled=enabled, **kwargs):
            yield
