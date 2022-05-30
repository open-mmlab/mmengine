# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from contextlib import contextmanager
from unittest import TestCase

import torch.nn as nn
from torch.optim import SGD

from mmengine.optim import AmpOptimWrapper, OptimWrapper, OptimWrapperDict


class TestOptimWrapperDict(TestCase):

    def setUp(self) -> None:
        model1 = nn.Linear(1, 1)
        model2 = nn.Linear(1, 1)
        self.optim1 = SGD(model1.parameters(), lr=0.1)
        self.optim2 = SGD(model2.parameters(), lr=0.1)
        self.optim_wrapper1 = OptimWrapper(self.optim1)
        self.optim_wrapper2 = OptimWrapper(self.optim2)
        self.optimizers_wrappers = OrderedDict(
            optim1=self.optim_wrapper1, optim2=self.optim_wrapper2)

    def test_init(self):
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        self.assertIs(composed_optim_wrapper.optimizer_wrappers,
                      self.optimizers_wrappers)
        # Different types of OptimWrapper will raise an error

        with self.assertRaisesRegex(
                AssertionError, 'All optimizer wrappers should have the same'):
            optim_wrapper2 = AmpOptimWrapper(optimizer=self.optim2)
            OptimWrapperDict(
                OrderedDict(optim1=self.optim_wrapper1, optim2=optim_wrapper2))

        with self.assertWarnsRegex(UserWarning, 'The `accumulative_iters` of'):
            optim_wrapper2 = OptimWrapper(
                optimizer=self.optim2, accumulative_iters=2)
            OptimWrapperDict(
                OrderedDict(optim1=self.optim_wrapper1, optim2=optim_wrapper2))

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
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        self.optim_wrapper1.accumulate_grad = context_a
        self.optim_wrapper2.accumulate_grad = context_b
        with composed_optim_wrapper.accumulate_grad(a, b, 0):
            self.assertEqual(a[0], 100)
            self.assertEqual(b[0], 200)

        self.assertEqual(a[0], 1)
        self.assertEqual(b[0], 2)

    def test_state_dict(self):
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        state_dict = composed_optim_wrapper.state_dict()
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

        composed_optim_wrapper_save = OptimWrapperDict(
            self.optimizers_wrappers)
        composed_optim_wrapper_load = OptimWrapperDict(
            OrderedDict(
                optim1=optim_wrapper_load1, optim2=optim_wrapper_load2))
        state_dict = composed_optim_wrapper_save.state_dict()
        composed_optim_wrapper_load.load_state_dict(state_dict)

        self.assertDictEqual(composed_optim_wrapper_load.state_dict(),
                             composed_optim_wrapper_save.state_dict())

    def test_items(self):
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        self.assertListEqual(
            list(composed_optim_wrapper.items()),
            list(self.optimizers_wrappers.items()))

    def test_values(self):
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        self.assertListEqual(
            list(composed_optim_wrapper.values()),
            list(self.optimizers_wrappers.values()))

    def test_keys(self):
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        self.assertListEqual(
            list(composed_optim_wrapper.keys()),
            list(self.optimizers_wrappers.keys()))

    def test_getitem(self):
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        self.assertIs(self.optimizers_wrappers['optim1'],
                      composed_optim_wrapper['optim1'])
        self.assertIs(self.optimizers_wrappers['optim2'],
                      composed_optim_wrapper['optim2'])

    def test_len(self):
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        self.assertEqual(len(composed_optim_wrapper), 2)

    def test_contain(self):
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        self.assertIn('optim1', composed_optim_wrapper)

    def test_repr(self):
        composed_optim_wrapper = OptimWrapperDict(self.optimizers_wrappers)
        desc = repr(composed_optim_wrapper)
        self.assertRegex(desc, 'name: optim1')
