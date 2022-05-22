# Copyright (c) OpenMMLab. All rights reserved.
from .data_parallel import (MMDataParallel, MMDistributedDataParallel,
                            MMSeporateDDPWrapper)
from .utils import is_model_wrapper

__all__ = [
    'MMDistributedDataParallel', 'MMDataParallel', 'is_model_wrapper',
    'MMSeporateDDPWrapper'
]
