# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmengine.registry import MODEL_WRAPPERS
from .data_parallel import MMDataParallel, MMDistributedDataParallel
from .utils import is_model_wrapper

MODEL_WRAPPERS.register_module(module=DataParallel)
MODEL_WRAPPERS.register_module(module=DistributedDataParallel)

__all__ = ['MMDistributedDataParallel', 'MMDataParallel', 'is_model_wrapper']
