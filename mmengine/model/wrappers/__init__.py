# Copyright (c) OpenMMLab. All rights reserved.
from .distributed import MMDistributedDataParallel
from .seperate_distributed import MMSeparateDistributedDataParallel
from .utils import is_model_wrapper

__all__ = [
    'MMDistributedDataParallel', 'is_model_wrapper',
    'MMSeparateDistributedDataParallel'
]
