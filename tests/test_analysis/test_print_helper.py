# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
import torch.nn as nn

from mmengine.analysis.complexity_analysis import FlopAnalyzer, parameter_count
from mmengine.analysis.print_helper import get_model_complexity_info


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


class NetAcceptOneTensorNOneScalar(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_features=5, out_features=6)
        self.l2 = nn.Linear(in_features=5, out_features=6)

    def forward(self, x1: torch.Tensor, r) -> torch.Tensor:
        out = r * self.l1(x1) + (1 - r) * self.l2(x1)
        return out


class TestGetModelCompexityInfo(unittest.TestCase):
    """Unittest for function get_model_complexity_info()

    Test use cases of variant `input_shape` and `input` combinations.
    """

    def setUp(self) -> None:
        """Create test elements (tensors, scalars, etc.) once for all."""
        self.t1 = torch.randn(1, 9, 5)
        self.shape1 = (9, 5)
        self.t2 = torch.randn(1, 9, 7)
        self.shape2 = (9, 7)
        self.scalar = 0.3

    def test_oneTensor(self) -> None:
        """Test a network that accept one tensor as input."""
        model = NetAcceptOneTensor()
        input = self.t1
        dict_complexity = get_model_complexity_info(model=model, inputs=input)
        self.assertEqual(dict_complexity['flops'],
                         FlopAnalyzer(model=model, inputs=input).total())
        self.assertEqual(dict_complexity['params'],
                         parameter_count(model=model)[''])

    def test_oneShape(self) -> None:
        """Test a network that accept one tensor as input."""
        model = NetAcceptOneTensor()
        input_shape = self.shape1
        dict_complexity = get_model_complexity_info(
            model=model, input_shape=input_shape)
        self.assertEqual(
            dict_complexity['flops'],
            FlopAnalyzer(model=model,
                         inputs=(torch.randn(1, *input_shape), )).total())
        self.assertEqual(dict_complexity['params'],
                         parameter_count(model=model)[''])

    def test_twoTensors(self) -> None:
        """Test a network that accept two tensors as input."""
        model = NetAcceptTwoTensors()
        input1 = self.t1
        input2 = self.t2
        dict_complexity = get_model_complexity_info(
            model=model, inputs=(input1, input2))
        self.assertEqual(
            dict_complexity['flops'],
            FlopAnalyzer(model=model, inputs=(input1, input2)).total())
        self.assertEqual(dict_complexity['params'],
                         parameter_count(model=model)[''])

    def test_twoShapes(self) -> None:
        """Test a network that accept two tensors as input."""
        model = NetAcceptTwoTensors()
        input_shape1 = self.shape1
        input_shape2 = self.shape2
        dict_complexity = get_model_complexity_info(
            model=model, input_shape=(input_shape1, input_shape2))
        self.assertEqual(
            dict_complexity['flops'],
            FlopAnalyzer(
                model=model,
                inputs=(torch.randn(1, *input_shape1),
                        torch.randn(1, *input_shape2))).total())
        self.assertEqual(dict_complexity['params'],
                         parameter_count(model=model)[''])

    def test_oneTensorNOneScalar(self) -> None:
        """Test a network that accept one tensor and one scalar as input."""
        model = NetAcceptOneTensorNOneScalar()
        input = self.t1
        scalar = self.scalar
        dict_complexity = get_model_complexity_info(
            model=model, inputs=(input, scalar))
        self.assertEqual(
            dict_complexity['flops'],
            FlopAnalyzer(model=model, inputs=(input, scalar)).total())
        self.assertEqual(dict_complexity['params'],
                         parameter_count(model=model)[''])

    def test_provideBothInputsNInputshape(self) -> None:
        """The function `get_model_complexity_info()` should throw `ValueError`
        when both `inputs` and `input_shape` are specified."""
        model = NetAcceptOneTensor()
        input = self.t1
        input_shape = self.shape1
        self.assertRaises(
            ValueError,
            get_model_complexity_info,
            model=model,
            inputs=input,
            input_shape=input_shape)

    def test_provideNoneOfInputsNInputshape(self) -> None:
        """The function `get_model_complexity_info()` should throw `ValueError`
        when neithor `inputs` nor `input_shape` is specified."""
        model = NetAcceptOneTensor()
        self.assertRaises(ValueError, get_model_complexity_info, model=model)
