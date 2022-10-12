# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version
from .distributed import MMDistributedDataParallel
from .seperate_distributed import MMSeparateDistributedDataParallel
from .test_time_aug import BaseTTAModel
from .utils import is_model_wrapper

__all__ = [
    'MMDistributedDataParallel', 'is_model_wrapper',
    'MMSeparateDistributedDataParallel', 'BaseTTAModel'
]

if digit_version(TORCH_VERSION) >= digit_version('1.11.0'):
    from .fully_sharded_distributed import \
        MMFullyShardedDataParallel  # noqa:F401
    __all__.append('MMFullyShardedDataParallel')
