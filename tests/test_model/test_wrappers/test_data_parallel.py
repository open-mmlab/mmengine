# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from mmengine.model.wrappers import (MMDataParallel, MMDistributedDataParallel,
                                     is_model_wrapper)
from mmengine.registry import MODEL_WRAPPERS


def mock(*args, **kwargs):
    pass


@patch('torch.distributed._broadcast_coalesced', mock)
@patch('torch.distributed.broadcast', mock)
@patch('torch.nn.parallel.DistributedDataParallel._ddp_init_helper', mock)
def test_is_model_wrapper():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 2, 1)

        def forward(self, x):
            return self.conv(x)

    # _verify_model_across_ranks is added in torch1.9.0 so we should check
    # whether _verify_model_across_ranks is the member of torch.distributed
    # before mocking
    if hasattr(torch.distributed, '_verify_model_across_ranks'):
        torch.distributed._verify_model_across_ranks = mock

    # _verify_model_across_ranks is added in torch1.11.0 so we should check
    # whether _verify_params_across_processes is the member of
    # torch.distributed before mocking
    if hasattr(torch.distributed, '_verify_params_across_processes'):
        torch.distributed._verify_params_across_processes = mock

    model = Model()
    assert not is_model_wrapper(model)

    mmdp = MMDataParallel(model)
    assert is_model_wrapper(mmdp)

    mmddp = MMDistributedDataParallel(model, process_group=MagicMock())
    assert is_model_wrapper(mmddp)

    torch_dp = DataParallel(model)
    assert is_model_wrapper(torch_dp)

    torch_ddp = DistributedDataParallel(model, process_group=MagicMock())
    assert is_model_wrapper(torch_ddp)

    # test model wrapper registry
    @MODEL_WRAPPERS.register_module()
    class ModelWrapper:

        def __init__(self, module):
            self.module = module

        def forward(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    model_wrapper = ModelWrapper(model)
    assert is_model_wrapper(model_wrapper)


class TestMMDataParallel(TestCase):

    def setUp(self):
        """Setup the demo image in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 1)

            def forward(self, x):
                return self.conv(x)

            def train_step(self, x):
                return self.forward(x)

            def val_step(self, x):
                return self.forward(x)

        self.model = Model()

    def test_train_step(self):

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 1)

            def forward(self, x):
                return self.conv(x)

        model = Model()
        mmdp = MMDataParallel(model)

        # test without train_step attribute
        with pytest.raises(AssertionError):
            mmdp.train_step(torch.zeros([1, 1, 3, 3]))

        out = self.model.train_step(torch.zeros([1, 1, 3, 3]))
        assert out.shape == (1, 2, 3, 3)

    def test_val_step(self):

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 1)

            def forward(self, x):
                return self.conv(x)

        model = Model()
        mmdp = MMDataParallel(model)

        # test without val_step attribute
        with pytest.raises(AssertionError):
            mmdp.val_step(torch.zeros([1, 1, 3, 3]))

        out = self.model.val_step(torch.zeros([1, 1, 3, 3]))
        assert out.shape == (1, 2, 3, 3)
