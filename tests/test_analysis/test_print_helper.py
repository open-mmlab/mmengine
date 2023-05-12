# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
import torch.nn as nn

from mmengine.analysis.complexity_analysis import FlopAnalyzer, parameter_count
from mmengine.analysis.print_helper import get_model_complexity_info
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


class NetAcceptOneTensor(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_features=5, out_features=6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.l1(x)
        return out


class NetAcceptTwoTensors(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_features=5, out_features=6)
        self.l2 = nn.Linear(in_features=7, out_features=6)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        out = self.l1(x1) + self.l2(x2)
        return out


class NetAcceptOneTensorAndOneScalar(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_features=5, out_features=6)
        self.l2 = nn.Linear(in_features=5, out_features=6)

    def forward(self, x1: torch.Tensor, r) -> torch.Tensor:
        out = r * self.l1(x1) + (1 - r) * self.l2(x1)
        return out


def test_get_model_complexity_info():
    input1 = torch.randn(1, 9, 5)
    input_shape1 = (9, 5)
    input2 = torch.randn(1, 9, 7)
    input_shape2 = (9, 7)
    scalar = 0.3

    # test a network that accepts one tensor as input
    model = NetAcceptOneTensor()
    complexity_info = get_model_complexity_info(model=model, inputs=input1)
    flops = FlopAnalyzer(model=model, inputs=input1).total()
    params = parameter_count(model=model)['']
    assert complexity_info['flops'] == flops
    assert complexity_info['params'] == params

    complexity_info = get_model_complexity_info(
        model=model, input_shape=input_shape1)
    flops = FlopAnalyzer(
        model=model, inputs=(torch.randn(1, *input_shape1), )).total()
    assert complexity_info['flops'] == flops

    # test a network that accepts two tensors as input
    model = NetAcceptTwoTensors()
    complexity_info = get_model_complexity_info(
        model=model, inputs=(input1, input2))
    flops = FlopAnalyzer(model=model, inputs=(input1, input2)).total()
    params = parameter_count(model=model)['']
    assert complexity_info['flops'] == flops
    assert complexity_info['params'] == params

    complexity_info = get_model_complexity_info(
        model=model, input_shape=(input_shape1, input_shape2))
    inputs = (torch.randn(1, *input_shape1), torch.randn(1, *input_shape2))
    flops = FlopAnalyzer(model=model, inputs=inputs).total()
    assert complexity_info['flops'] == flops

    # test a network that accepts one tensor and one scalar as input
    model = NetAcceptOneTensorAndOneScalar()
    # For pytorch<1.9, a scalar input is not acceptable for torch.jit,
    # wrap it to `torch.tensor`. See https://github.com/pytorch/pytorch/blob/cd9dd653e98534b5d3a9f2576df2feda40916f1d/torch/csrc/jit/python/python_arg_flatten.cpp#L90. # noqa: E501
    scalar = torch.tensor([
        scalar
    ]) if digit_version(TORCH_VERSION) < digit_version('1.9.0') else scalar
    complexity_info = get_model_complexity_info(
        model=model, inputs=(input1, scalar))
    flops = FlopAnalyzer(model=model, inputs=(input1, scalar)).total()
    params = parameter_count(model=model)['']
    assert complexity_info['flops'] == flops
    assert complexity_info['params'] == params

    # `get_model_complexity_info()` should throw `ValueError`
    # when neithor `inputs` nor `input_shape` is specified
    with pytest.raises(ValueError, match='should be set'):
        get_model_complexity_info(model)

    # `get_model_complexity_info()` should throw `ValueError`
    # when both `inputs` and `input_shape` are specified
    model = NetAcceptOneTensor()
    with pytest.raises(ValueError, match='cannot be both set'):
        get_model_complexity_info(
            model, inputs=input1, input_shape=input_shape1)
