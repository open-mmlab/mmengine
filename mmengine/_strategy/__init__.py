# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import is_installed
from .base_strategy import BaseStrategy
from .ddp_strategy import DDPStrategy
from .fsdp_strategy import FSDPStrategy
from .single_device_strategy import SingleDeviceStrategy

__all__ = [
    'BaseStrategy', 'DDPStrategy', 'SingleDeviceStrategy', 'FSDPStrategy'
]

if is_installed('deepspeed'):
    from .deepspeed_strategy import DeepSpeedStrategy
    __all__.append('DeepSpeedStrategy')
