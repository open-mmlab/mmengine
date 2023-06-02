# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import is_installed
from .base import BaseStrategy
from .distributed import DDPStrategy
from .fsdp import FSDPStrategy
from .single_device import SingleDeviceStrategy

__all__ = [
    'BaseStrategy', 'DDPStrategy', 'SingleDeviceStrategy', 'FSDPStrategy'
]

if is_installed('deepspeed'):
    from .deepspeed import DeepSpeedStrategy  # noqa:F401
    __all__.append('DeepSpeedStrategy')
