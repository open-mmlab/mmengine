# Copyright (c) OpenMMLab. All rights reserved.
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from mmengine.registry import MODEL_WRAPPERS

MODEL_WRAPPERS.register_module(module=DataParallel)
MODEL_WRAPPERS.register_module(module=DistributedDataParallel)
