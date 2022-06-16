# Copyright (c) OpenMMLab. All rights reserved.
from .utils import (get_device, get_max_cuda_memory, is_cuda_available,
                    is_mlu_available)

__all__ = [
    'get_max_cuda_memory', 'get_device', 'is_cuda_available',
    'is_mlu_available'
]
