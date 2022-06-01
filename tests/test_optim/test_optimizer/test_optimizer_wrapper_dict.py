# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from unittest import TestCase
from unittest.mock import patch

import torch.nn as nn
from torch.optim import SGD

from mmengine.optim import AmpOptimWrapper, OptimWrapper, OptimWrapperDict


class TestOptimWrapperDict(TestCase):

    def setUp(self) -> None:
        model1 = nn.Linear(1, 1)
        model2 = nn.Linear(1, 1)
        self.optim1 = SGD(model1.parameters(), lr=0.1, momentum=0.8)
        self.optim2 = SGD(model2.parameters(), lr=0.2, momentum=0.9)
        self.optim_wrapper1 = OptimWrapper(self.optim1)
        self.optim_wrapper2 = OptimWrapper(self.optim2)
        self.optimizers_wrappers = dict(
            optim1=self.optim_wrapper1, optim2=self.optim_wrapper2)

    @patch('torch.cuda.is_available', lambda: True)
    def test_init(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.assertEqual(optim_wrapper_dict.optim_wrappers,
                         self.optimizers_wrappers)
        # Different types of OptimWrapper will raise an error

        with self.assertRaisesRegex(
                AssertionError, 'All optimizer wrappers should have the same'):
            optim_wrapper2 = AmpOptimWrapper(optimizer=self.optim2)
            OptimWrapperDict(optim1=self.optim_wrapper1, optim2=optim_wrapper2)

        with self.assertWarnsRegex(UserWarning, 'The `accumulative_iters` of'):
            optim_wrapper2 = OptimWrapper(
                optimizer=self.optim2, accumulative_iters=2)
            OptimWrapperDict(optim1=self.optim_wrapper1, optim2=optim_wrapper2)

    def test_accumulate_grad(self):

        @contextmanager
        def context_a(a, b, *args, **kwargs):
            a[0] = 100
            yield
            a[0] = 1

        @contextmanager
        def context_b(a, b, *args, **kwargs):
            b[0] = 200
            yield
            b[0] = 2

        a = [0]
        b = [0]
        # Test enter the context both of `optim_wrapper1` and `optim_wrapper1`.
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        self.optim_wrapper1.accumulate_grad = context_a
        self.optim_wrapper2.accumulate_grad = context_b
        with optim_wrapper_dict.accumulate_grad(a, b, 0):
            self.assertEqual(a[0], 100)
            self.assertEqual(b[0], 200)

        self.assertEqual(a[0], 1)
        self.assertEqual(b[0], 2)

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

    def test_get_lr(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        lr = optim_wrapper_dict.get_lr()
        self.assertEqual(lr['optim1'], [0.1])
        self.assertEqual(lr['optim2'], [0.2])

    def test_get_momentum(self):
        optim_wrapper_dict = OptimWrapperDict(**self.optimizers_wrappers)
        momentum = optim_wrapper_dict.get_momentum()
        self.assertEqual(momentum['optim1'], [0.8])
        self.assertEqual(momentum['optim2'], [0.9])

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
