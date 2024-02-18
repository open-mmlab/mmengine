# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from mmengine.utils.dl_utils import torch_meshgrid


def test_torch_meshgrid():
    # torch_meshgrid should not throw warning
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 6])
        grid_x, grid_y = torch_meshgrid(x, y)
