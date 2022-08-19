# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils.parrots_wrapper import TORCH_VERSION
from mmengine.utils.version_utils import digit_version
from .averaged_model import (BaseAveragedModel, ExponentialMovingAverage,
                             MomentumAnnealingEMA, StochasticWeightAverage)
from .base_model import BaseDataPreprocessor, BaseModel, ImgDataPreprocessor
from .base_module import BaseModule, ModuleDict, ModuleList, Sequential
from .utils import (Caffe2XavierInit, ConstantInit, KaimingInit, NormalInit,
                    PretrainedInit, TruncNormalInit, UniformInit, XavierInit,
                    bias_init_with_prob, caffe2_xavier_init, constant_init,
                    detect_anomalous_params, kaiming_init, merge_dict,
                    normal_init, stack_batch, trunc_normal_init, uniform_init,
                    xavier_init)
from .wrappers import (MMDistributedDataParallel,
                       MMSeparateDistributedDataParallel, is_model_wrapper)

__all__ = [
    'MMDistributedDataParallel', 'is_model_wrapper', 'BaseAveragedModel',
    'StochasticWeightAverage', 'ExponentialMovingAverage',
    'MomentumAnnealingEMA', 'BaseModel', 'BaseDataPreprocessor',
    'ImgDataPreprocessor', 'MMSeparateDistributedDataParallel', 'BaseModule',
    'stack_batch', 'merge_dict', 'detect_anomalous_params', 'ModuleList',
    'ModuleDict', 'Sequential', 'constant_init', 'xavier_init', 'kaiming_init',
    'normal_init', 'trunc_normal_init', 'uniform_init', 'caffe2_xavier_init',
    'bias_init_with_prob', 'bias_init_with_prob', 'XavierInit', 'NormalInit',
    'TruncNormalInit', 'UniformInit', 'KaimingInit', 'Caffe2XavierInit',
    'PretrainedInit', 'ConstantInit'
]

if digit_version(TORCH_VERSION) >= digit_version('1.11.0'):
    from .wrappers import MMFullyShardedDataParallel  # noqa:F401
    __all__.append('MMFullyShardedDataParallel')
