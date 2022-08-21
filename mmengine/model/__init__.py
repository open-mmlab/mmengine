# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version
from .averaged_model import (BaseAveragedModel, ExponentialMovingAverage,
                             MomentumAnnealingEMA, StochasticWeightAverage)
from .base_model import BaseDataPreprocessor, BaseModel, ImgDataPreprocessor
from .base_module import BaseModule, ModuleDict, ModuleList, Sequential
from .utils import (detect_anomalous_params, merge_dict, revert_sync_batchnorm,
                    stack_batch)
from .wrappers import (MMDistributedDataParallel,
                       MMSeparateDistributedDataParallel, is_model_wrapper)

__all__ = [
    'MMDistributedDataParallel', 'is_model_wrapper', 'BaseAveragedModel',
    'StochasticWeightAverage', 'ExponentialMovingAverage',
    'MomentumAnnealingEMA', 'BaseModel', 'BaseDataPreprocessor',
    'ImgDataPreprocessor', 'MMSeparateDistributedDataParallel', 'BaseModule',
    'stack_batch', 'merge_dict', 'detect_anomalous_params', 'ModuleList',
    'ModuleDict', 'Sequential', 'revert_sync_batchnorm'
]

if digit_version(TORCH_VERSION) >= digit_version('1.11.0'):
    from .wrappers import MMFullyShardedDataParallel  # noqa:F401
    __all__.append('MMFullyShardedDataParallel')
