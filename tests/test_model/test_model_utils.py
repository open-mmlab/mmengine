# Copyright (c) OpenMMLab. All rights reserved.
import os

import pytest
import torch
import torch.nn as nn
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmengine.model import (MMDistributedDataParallel,
                            MMSeparateDistributedDataParallel,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.registry import MODEL_WRAPPERS, Registry


@pytest.mark.skipif(
    torch.__version__ == 'parrots', reason='not supported in parrots now')
def test_revert_syncbn():
    # conv = ConvModule(3, 8, 2, norm_cfg=dict(type='SyncBN'))
    conv = nn.Sequential(nn.Conv2d(3, 8, 2), nn.SyncBatchNorm(8))
    x = torch.randn(1, 3, 10, 10)
    # Expect a ValueError prompting that SyncBN is not supported on CPU
    with pytest.raises(ValueError):
        y = conv(x)
    conv = revert_sync_batchnorm(conv)
    y = conv(x)
    assert y.shape == (1, 8, 9, 9)


def test_is_model_wrapper():
    # Test basic module wrapper.
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29510'
    os.environ['RANK'] = str(0)
    init_process_group(backend='gloo', rank=0, world_size=1)
    model = nn.Linear(1, 1)

    for wrapper in [
            DistributedDataParallel, MMDistributedDataParallel,
            MMSeparateDistributedDataParallel, DataParallel
    ]:
        wrapper_model = wrapper(model)
        assert is_model_wrapper(wrapper_model)

    # Test `is_model_wrapper` can check model wrapper registered in custom
    # registry.
    CHILD_REGISTRY = Registry('test_is_model_wrapper', parent=MODEL_WRAPPERS)

    class CustomModelWrapper(nn.Module):

        def __init__(self, model):
            super().__init__()
            self.module = model

        pass

    CHILD_REGISTRY.register_module(module=CustomModelWrapper)

    for wrapper in [
            DistributedDataParallel, MMDistributedDataParallel,
            MMSeparateDistributedDataParallel, DataParallel, CustomModelWrapper
    ]:
        wrapper_model = wrapper(model)
        assert is_model_wrapper(wrapper_model)

    # Test `is_model_wrapper` will not check model wrapper in parent
    # registry from a child registry.
    for wrapper in [
            DistributedDataParallel, MMDistributedDataParallel,
            MMSeparateDistributedDataParallel, DataParallel
    ]:
        wrapper_model = wrapper(model)
        assert not is_model_wrapper(wrapper_model, registry=CHILD_REGISTRY)

    wrapper_model = CustomModelWrapper(model)
    assert is_model_wrapper(wrapper_model, registry=CHILD_REGISTRY)
    destroy_process_group()
