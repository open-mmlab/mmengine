# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn.functional as F
from torch import nn

from mmengine.runner.activation_checkpointing import \
    turn_on_activation_checkpointing
from mmengine.testing import assert_allclose


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.bn3(self.conv3(x))
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TestActivationCheckpointing(TestCase):

    def test_activation_checkpointing(self):
        model = Model()
        input = torch.randn(16, 3, 224, 224)
        input.requires_grad = True
        output = model(input)
        output.sum().backward()
        grad = input.grad.clone()

        turn_on_activation_checkpointing(model, ['conv1', 'conv2', 'conv3'])
        output2 = model(input)
        output2.sum().backward()
        grad2 = input.grad.clone()

        assert_allclose(output, output2)
        assert_allclose(grad, grad2, rtol=1e-3, atol=1e-3)
