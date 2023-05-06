# Copyright (c) OpenMMLab. All rights reserved.
from .base_strategy import BaseStrategy
from .ddp_strategy import DDPStrategy
from .deepspeed_strategy import DeepSpeedStrategy
from .single_device_strategy import SingleDeviceStrategy

__all__ = [
    'BaseStrategy', 'DDPStrategy', 'SingleDeviceStrategy', 'DeepSpeedStrategy'
]
