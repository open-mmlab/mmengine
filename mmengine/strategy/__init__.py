# Copyright (c) OpenMMLab. All rights reserved.
from .ddp_strategy import DDPStrategy
from .strategy import Strategy

__all__ = ['Strategy', 'DDPStrategy']
