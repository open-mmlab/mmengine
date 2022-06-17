# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
import torch.nn as nn

from mmengine.runner import auto_cast
from mmengine.utils import TORCH_VERSION, digit_version


class TestAmp(unittest.TestCase):

    def test_auto_cast(self):
        if not torch.cuda.is_available():
            if digit_version(TORCH_VERSION) < digit_version('1.10.0'):
                # `torch.cuda.amp.autocast` is only support in gpu mode, if
                # cuda is not available, it will return an empty context and
                # should not accept any arguments.
                with self.assertRaisesRegex(RuntimeError,
                                            'If pytorch versions is '):
                    with auto_cast():
                        pass

                with auto_cast(enabled=False):
                    layer = nn.Conv2d(1, 1, 1)
                    res = layer(torch.randn(1, 1, 1, 1))
                    self.assertEqual(res.dtype, torch.float32)

            else:
                with auto_cast(device_type='cpu'):
                    # torch.autocast support cpu mode.
                    layer = nn.Conv2d(1, 1, 1)
                    res = layer(torch.randn(1, 1, 1, 1))
                    self.assertIn(res.dtype, (torch.bfloat16, torch.float16))
                    with auto_cast(enabled=False):
                        res = layer(torch.randn(1, 1, 1, 1))
                        self.assertEqual(res.dtype, torch.float32)

        else:
            if digit_version(TORCH_VERSION) < digit_version('1.10.0'):
                devices = ['cuda']
            else:
                devices = ['cpu', 'cuda']
            for device in devices:
                with auto_cast():
                    # torch.autocast support cpu and cuda mode.
                    layer = nn.Conv2d(1, 1, 1).to(device)
                    res = layer(torch.randn(1, 1, 1, 1).to(device))
                    self.assertIn(res.dtype, (torch.bfloat16, torch.float16))
                    with auto_cast(enabled=False):
                        res = layer(torch.randn(1, 1, 1, 1).to(device))
                        self.assertEqual(res.dtype, torch.float32)
                # Test with fp32_enabled
                with auto_cast(enabled=False):
                    layer = nn.Conv2d(1, 1, 1).to(device)
                    res = layer(torch.randn(1, 1, 1, 1).to(device))
                    self.assertEqual(res.dtype, torch.float32)
