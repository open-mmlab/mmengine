# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
import torch.nn as nn

import mmengine
from mmengine.device import (get_device, is_mlu_available, is_musa_available,
                             is_npu_available)
from mmengine.runner import autocast
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


class TestAmp(unittest.TestCase):

    def test_autocast(self):
        if is_npu_available():
            device = 'npu'
            with autocast(device_type=device):
                # torch.autocast support npu mode.
                layer = nn.Conv2d(1, 1, 1).to(device)
                res = layer(torch.randn(1, 1, 1, 1).to(device))
                self.assertIn(res.dtype, (torch.bfloat16, torch.float16))
                with autocast(enabled=False, device_type=device):
                    res = layer(torch.randn(1, 1, 1, 1).to(device))
                    self.assertEqual(res.dtype, torch.float32)
            # Test with fp32_enabled
            with autocast(enabled=False, device_type=device):
                layer = nn.Conv2d(1, 1, 1).to(device)
                res = layer(torch.randn(1, 1, 1, 1).to(device))
                self.assertEqual(res.dtype, torch.float32)
        elif is_mlu_available():
            device = 'mlu'
            with autocast(device_type=device):
                # torch.autocast support mlu mode.
                layer = nn.Conv2d(1, 1, 1).to(device)
                res = layer(torch.randn(1, 1, 1, 1).to(device))
                self.assertIn(res.dtype, (torch.bfloat16, torch.float16))
                with autocast(enabled=False, device_type=device):
                    res = layer(torch.randn(1, 1, 1, 1).to(device))
                    self.assertEqual(res.dtype, torch.float32)
            # Test with fp32_enabled
            with autocast(enabled=False, device_type=device):
                layer = nn.Conv2d(1, 1, 1).to(device)
                res = layer(torch.randn(1, 1, 1, 1).to(device))
                self.assertEqual(res.dtype, torch.float32)
        elif is_musa_available():
            device = 'musa'
            with autocast(device_type=device):
                # torch.autocast support mlu mode.
                layer = nn.Conv2d(1, 1, 1).to(device)
                res = layer(torch.randn(1, 1, 1, 1).to(device))
                self.assertIn(res.dtype, (torch.bfloat16, torch.float16))
                with autocast(enabled=False, device_type=device):
                    res = layer(torch.randn(1, 1, 1, 1).to(device))
                    self.assertEqual(res.dtype, torch.float32)
            # Test with fp32_enabled
            with autocast(enabled=False, device_type=device):
                layer = nn.Conv2d(1, 1, 1).to(device)
                res = layer(torch.randn(1, 1, 1, 1).to(device))
                self.assertEqual(res.dtype, torch.float32)
        elif not torch.cuda.is_available():
            if digit_version(TORCH_VERSION) < digit_version('1.10.0'):
                # `torch.cuda.amp.autocast` is only support in gpu mode, if
                # cuda is not available, it will return an empty context and
                # should not accept any arguments.
                with self.assertRaisesRegex(RuntimeError,
                                            'If pytorch versions is '):
                    with autocast():
                        pass

                with autocast(enabled=False):
                    layer = nn.Conv2d(1, 1, 1)
                    res = layer(torch.randn(1, 1, 1, 1))
                    self.assertEqual(res.dtype, torch.float32)

            else:
                with autocast(device_type='cpu'):
                    # torch.autocast support cpu mode.
                    layer = nn.Conv2d(1, 1, 1)
                    res = layer(torch.randn(1, 1, 1, 1))
                    self.assertIn(res.dtype, (torch.bfloat16, torch.float16))
                    with autocast(enabled=False):
                        res = layer(torch.randn(1, 1, 1, 1))
                        self.assertEqual(res.dtype, torch.float32)

        else:
            if digit_version(TORCH_VERSION) < digit_version('1.10.0'):
                devices = ['cuda']
            else:
                devices = ['cpu', 'cuda']
            for device in devices:
                with autocast(device_type=device):
                    # torch.autocast support cpu and cuda mode.
                    layer = nn.Conv2d(1, 1, 1).to(device)
                    res = layer(torch.randn(1, 1, 1, 1).to(device))
                    self.assertIn(res.dtype, (torch.bfloat16, torch.float16))
                    with autocast(enabled=False, device_type=device):
                        res = layer(torch.randn(1, 1, 1, 1).to(device))
                        self.assertEqual(res.dtype, torch.float32)
                # Test with fp32_enabled
                with autocast(enabled=False, device_type=device):
                    layer = nn.Conv2d(1, 1, 1).to(device)
                    res = layer(torch.randn(1, 1, 1, 1).to(device))
                    self.assertEqual(res.dtype, torch.float32)

        # Test mps
        if digit_version(TORCH_VERSION) >= digit_version('1.12.0'):
            mmengine.runner.amp.get_device = lambda: 'mps'
            with autocast(enabled=False):
                layer = nn.Conv2d(1, 1, 1)
                res = layer(torch.randn(1, 1, 1, 1))
                self.assertEqual(res.dtype, torch.float32)

            with self.assertRaisesRegex(ValueError,
                                        'User specified autocast device_type'):
                with autocast(enabled=True):
                    pass
        # Native pytorch does not support mlu, here we simply test autocast
        # will call `torch.autocast`, which will be overridden by mlu version
        # pytorch
            mmengine.runner.amp.get_device = lambda: 'mlu'
            with self.assertRaises(RuntimeError):
                with autocast(enabled=False):
                    pass
            mmengine.runner.amp.get_device = get_device
