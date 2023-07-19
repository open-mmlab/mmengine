# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION
from .base import BaseStrategy
from .deepspeed import DeepSpeedStrategy
from .distributed import DDPStrategy
from .single_device import SingleDeviceStrategy

__all__ = [
    'BaseStrategy', 'DDPStrategy', 'SingleDeviceStrategy', 'DeepSpeedStrategy'
]

if digit_version(TORCH_VERSION) >= digit_version('2.0.0'):
    try:
        from .fsdp import FSDPStrategy  # noqa:F401
        __all__.append('FSDPStrategy')
    except:  # noqa: E722
        pass
