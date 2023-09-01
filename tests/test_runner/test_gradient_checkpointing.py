# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch import nn

from mmengine.runner.gradient_checkpoint import turn_on_gradient_checkpoint
from mmengine.testing import assert_allclose


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 6, 6)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 6, 6)
        self.bn2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 6, 6)
        self.bn3 = nn.BatchNorm2d(6)
        self.linear = nn.Linear(6, 6)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.linear(x)
        return x


class TestGradientCheckpoint(TestCase):

    def test_gradient_checkpoint(self):
        model = Model()
        input = torch.randn(64, 6, 32, 32)
        input.requires_grad = True

        output = model(input)
        output.sum().backward()
        grad = input.grad.clone()

        turn_on_gradient_checkpoint(model, ['conv1', 'conv2', 'conv3'])
        output2 = model(input)
        output2.sum().backward()
        grad2 = input.grad.clone()

        assert_allclose(output, output2)
        assert_allclose(grad, grad2)
