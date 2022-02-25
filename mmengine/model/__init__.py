# Copyright (c) OpenMMLab. All rights reserved.
from .wrappers import (MMDataParallel, MMDistributedDataParallel,
                       is_model_wrapper)

__all__ = ['MMDistributedDataParallel', 'MMDataParallel', 'is_model_wrapper']
