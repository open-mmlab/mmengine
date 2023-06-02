# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn
from torch.optim import SGD

from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmengine.utils import digit_version


class TestOptimWrapperDict(TestCase):

    def setUp(self) -> None:
        self.model1 = nn.Linear(1, 1)
        self.model2 = nn.Linear(1, 1)
        self.optim1 = SGD(self.model1.parameters(), lr=0.1, momentum=0.8)
        self.optim2 = SGD(self.model2.parameters(), lr=0.2, momentum=0.9)
        self.optim_wrapper1 = OptimWrapper(self.optim1)
        self.optim_wrapper2 = OptimWrapper(self.optim2)
        self.optimizers_wrappers = dict(
            optim1=self.optim_wrapper1, optim2=self.optim_wrapper2)

    def test_init(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.assertEqual(optim_wrapper_dict.optim_wrappers,
                         self.optimizers_wrappers)
        with self.assertRaisesRegex(AssertionError,
                                    '`OptimWrapperDict` only accept'):
            OptimWrapperDict(**dict(optim1=self.optim1, optim2=self.optim2))

    def test_update_params(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        with self.assertRaisesRegex(NotImplementedError,
                                    '`update_params` should be called'):
            optim_wrapper_dict.update_params(1)

    def test_backward(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        with self.assertRaisesRegex(NotImplementedError,
                                    '`backward` should be called'):
            optim_wrapper_dict.backward(1)

    def test_step(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        with self.assertRaisesRegex(NotImplementedError,
                                    '`step` should be called'):
            optim_wrapper_dict.step()

    def test_zero_grad(self):
        # Test clear all grad
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.model1(torch.randn(1, 1)).sum().backward()
        self.model2(torch.randn(1, 1)).sum().backward()
        self.assertTrue((self.model1.weight.grad != 0).any())
        self.assertTrue((self.model2.weight.grad != 0).any())
        optim_wrapper_dict.zero_grad()
        if digit_version(torch.__version__) < digit_version('2.0.0'):
            self.assertTrue((self.model1.weight.grad == 0).all())
            self.assertTrue((self.model2.weight.grad == 0).all())
        else:
            self.assertIsNone(self.model1.weight.grad)
            self.assertIsNone(self.model2.weight.grad)

    def test_optim_context(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        with self.assertRaisesRegex(NotImplementedError,
                                    '`optim_context` should be called'):
            with optim_wrapper_dict.optim_context(self.model1):
                yield

    def test_initialize_count_status(self):
        # Test `initialize_count_status` can be called.
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        optim_wrapper_dict.initialize_count_status(self.model1, 1, 1)

    def test_param_groups(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.assertEqual(optim_wrapper_dict.param_groups['optim1'],
                         self.optim1.param_groups)
        self.assertEqual(optim_wrapper_dict.param_groups['optim2'],
                         self.optim2.param_groups)

    def test_get_lr(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        lr = optim_wrapper_dict.get_lr()
        self.assertEqual(lr['optim1.lr'], [0.1])
        self.assertEqual(lr['optim2.lr'], [0.2])

    def test_get_momentum(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        momentum = optim_wrapper_dict.get_momentum()
        self.assertEqual(momentum['optim1.momentum'], [0.8])
        self.assertEqual(momentum['optim2.momentum'], [0.9])

    def test_state_dict(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        state_dict = optim_wrapper_dict.state_dict()
        self.assertEqual(state_dict['optim1'],
                         self.optim_wrapper1.state_dict())
        self.assertEqual(state_dict['optim2'],
                         self.optim_wrapper2.state_dict())

    def test_load_state_dict(self):
        # Test OptimWrapperDict can load from saved state dict.
        model1 = nn.Linear(1, 1)
        model2 = nn.Linear(1, 1)
        optim1 = SGD(model1.parameters(), lr=0.1)
        optim2 = SGD(model2.parameters(), lr=0.1)
        optim_wrapper_load1 = OptimWrapper(optim1)
        optim_wrapper_load2 = OptimWrapper(optim2)

        optim_wrapper_dict_save = OptimWrapperDict(**self.optimizers_wrappers)
        optim_wrapper_dict_load = OptimWrapperDict(
            optim1=optim_wrapper_load1, optim2=optim_wrapper_load2)
        state_dict = optim_wrapper_dict_save.state_dict()
        optim_wrapper_dict_load.load_state_dict(state_dict)

        self.assertDictEqual(optim_wrapper_dict_load.state_dict(),
                             optim_wrapper_dict_save.state_dict())

    def test_items(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.assertListEqual(
            list(optim_wrapper_dict.items()),
            list(self.optimizers_wrappers.items()))

    def test_values(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.assertListEqual(
            list(optim_wrapper_dict.values()),
            list(self.optimizers_wrappers.values()))

    def test_keys(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.assertListEqual(
            list(optim_wrapper_dict.keys()),
            list(self.optimizers_wrappers.keys()))

    def test_getitem(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.assertIs(self.optimizers_wrappers['optim1'],
                      optim_wrapper_dict['optim1'])
        self.assertIs(self.optimizers_wrappers['optim2'],
                      optim_wrapper_dict['optim2'])

    def test_len(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.assertEqual(len(optim_wrapper_dict), 2)

    def test_contain(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.assertIn('optim1', optim_wrapper_dict)

    def test_repr(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        desc = repr(optim_wrapper_dict)
        self.assertRegex(desc, 'name: optim1')
