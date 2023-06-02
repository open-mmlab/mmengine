# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseStrategy
from .deepspeed import DeepSpeedStrategy
from .distributed import DDPStrategy
from .single_device import SingleDeviceStrategy

__all__ = [
    'BaseStrategy', 'DDPStrategy', 'SingleDeviceStrategy', 'DeepSpeedStrategy'
]
