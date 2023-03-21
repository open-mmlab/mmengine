# Copyright (c) OpenMMLab. All rights reserved.
from .ddp_strategy import DDPStrategy
from .native_strategy import NativeStrategy
from .strategy import Mode, Strategy

__all__ = ['Strategy', 'DDPStrategy', 'NativeStrategy', 'Mode']
