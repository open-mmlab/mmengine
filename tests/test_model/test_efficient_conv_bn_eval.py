# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from torch import nn

from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval_for_single_model
from mmengine.testing import assert_allclose
from mmengine.utils import is_installed
from mmengine.utils.dl_utils import TORCH_VERSION
from mmengine.utils.version_utils import digit_version

mmcv_is_installed = is_installed('mmcv')


class BackboneModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if mmcv_is_installed:
            from mmcv.cnn import ConvModule
            conv0 = nn.Conv2d(6, 6, 6)
            bn0 = nn.BatchNorm2d(6)
            self.mod1 = ConvModule.create_from_conv_bn(conv0, bn0)
        self.conv1 = nn.Conv2d(6, 6, 6)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 6, 6)
        self.bn2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 6, 6)
        self.bn3 = nn.BatchNorm2d(6)

    def forward(self, x):
        if mmcv_is_installed:
            # this ConvModule can use efficient_conv_bn_eval feature
            x = self.mod1(x)
        # this conv-bn pair can use efficient_conv_bn_eval feature
        x = self.bn1(self.conv1(x))
        # this conv-bn pair can use efficient_conv_bn_eval feature
        # only for the second `self.conv2` call.
        x = self.bn2(self.conv2(self.conv2(x)))
        # this conv-bn pair can use efficient_conv_bn_eval feature
        # just for the first forward of the `self.bn3`
        x = self.bn3(self.bn3(self.conv3(x)))
        return x


@unittest.skipIf(
    digit_version(TORCH_VERSION) < digit_version('1.8'),
    reason='torch.fx needs Pytorch 1.8 or higher')
class TestEfficientConvBNEval(TestCase):
    """Test the turn_on_efficient_conv_bn_eval function."""

    def test_efficient_conv_bn_eval(self):
        model = BackboneModel()
        model.eval()
        input = torch.randn(64, 6, 32, 32)
        output = model(input)
        turn_on_efficient_conv_bn_eval_for_single_model(model)
        output2 = model(input)
        print((output - output2).abs().max().item())
        assert_allclose(output, output2)
