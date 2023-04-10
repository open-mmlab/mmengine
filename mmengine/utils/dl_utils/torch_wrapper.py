# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
import torch

TORCH_VERSION = torch.__version__

def _get_extension():
    """A wrapper to obtain extension class from PyTorch."""
    from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                               CUDAExtension)
    return BuildExtension, CppExtension, CUDAExtension


def _get_conv() -> tuple:
    """A wrapper to obtain base classes of Conv layers from PyTorch."""
    from torch.nn.modules.conv import _ConvNd, _ConvTransposeMixin
    return _ConvNd, _ConvTransposeMixin


def _get_pool() -> tuple:
    """A wrapper to obtain base classes of pooling layers from PyTorch."""
    from torch.nn.modules.pooling import (_AdaptiveAvgPoolNd,
                                          _AdaptiveMaxPoolNd, _AvgPoolNd,
                                          _MaxPoolNd)
    return _AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd


def _get_norm() -> tuple:
    """A wrapper to obtain base classes of normalization layers from PyTorch."""
    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.instancenorm import _InstanceNorm
    SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_


_ConvNd, _ConvTransposeMixin = _get_conv()
_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()
_AdaptiveAvgPoolNd, _AdaptiveMaxPoolNd, _AvgPoolNd, _MaxPoolNd = _get_pool()


class SyncBatchNorm(SyncBatchNorm_):  # type: ignore

    def _check_input_dim(self, input):
        super()._check_input_dim(input)
