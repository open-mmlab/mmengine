# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from unittest import TestCase

import torch.nn as nn
from torch.optim import SGD

from mmengine.optim import (AmpOptimizerWrapper, OptimizerWrapper,
                            OptimizerWrapperDict)


class TestOptimizerWrapperDict(TestCase):

    def setUp(self) -> None:
        model1 = nn.Linear(1, 1)
        model2 = nn.Linear(1, 1)
        self.optim1 = SGD(model1.parameters(), lr=0.1)
        self.optim2 = SGD(model2.parameters(), lr=0.1)
        self.optim_wrapper1 = OptimizerWrapper(self.optim1)
        self.optim_wrapper2 = OptimizerWrapper(self.optim2)
        self.optimizers_wrappers = OrderedDict(
            optim1=self.optim_wrapper1, optim2=self.optim_wrapper2)

    def test_init(self):
        composed_optim_wrapper = OptimizerWrapperDict(self.optimizers_wrappers)
        self.assertEqual(composed_optim_wrapper.optimizer_wrappers,
                         self.optimizers_wrappers)
        # Different types of OptimizerWrapper will raise an error

        with self.assertRaises(AssertionError):
            optim_wrapper2 = AmpOptimizerWrapper(optimizer=self.optim2)
            OptimizerWrapperDict(
                OrderedDict(optim1=self.optim_wrapper1, optim2=optim_wrapper2))

        with self.assertWarns(UserWarning):
            optim_wrapper2 = OptimizerWrapper(
                optimizer=self.optim2, accumulative_iters=2)
            OptimizerWrapperDict(
                OrderedDict(optim1=self.optim_wrapper1, optim2=optim_wrapper2))

    def test_state_dict(self):
        composed_optim_wrapper = OptimizerWrapperDict(self.optimizers_wrappers)
        state_dict = composed_optim_wrapper.state_dict()
        self.assertEqual(state_dict['optim1'],
                         self.optim_wrapper1.state_dict())
        self.assertEqual(state_dict['optim2'],
                         self.optim_wrapper2.state_dict())

    def test_load_state_dict(self):
        # Test OptimizerWrapperDict can load from saved state dict.
        model1 = nn.Linear(1, 1)
        model2 = nn.Linear(1, 1)
        optim1 = SGD(model1.parameters(), lr=0.1)
        optim2 = SGD(model2.parameters(), lr=0.1)
        optim_wrapper_load1 = OptimizerWrapper(optim1)
        optim_wrapper_load2 = OptimizerWrapper(optim2)

        composed_optim_wrapper_save = OptimizerWrapperDict(
            self.optimizers_wrappers)
        composed_optim_wrapper_load = OptimizerWrapperDict(
            OrderedDict(
                optim1=optim_wrapper_load1, optim2=optim_wrapper_load2))
        state_dict = composed_optim_wrapper_save.state_dict()
        composed_optim_wrapper_load.load_state_dict(state_dict)

        self.assertDictEqual(composed_optim_wrapper_load.state_dict(),
                             composed_optim_wrapper_save.state_dict())

    def test_items(self):
        composed_optim_wrapper = OptimizerWrapperDict(self.optimizers_wrappers)
        self.assertListEqual(
            list(composed_optim_wrapper.items()),
            list(self.optimizers_wrappers.items()))

    def test_values(self):
        composed_optim_wrapper = OptimizerWrapperDict(self.optimizers_wrappers)
        self.assertListEqual(
            list(composed_optim_wrapper.values()),
            list(self.optimizers_wrappers.values()))

    def test_keys(self):
        composed_optim_wrapper = OptimizerWrapperDict(self.optimizers_wrappers)
        self.assertListEqual(
            list(composed_optim_wrapper.keys()),
            list(self.optimizers_wrappers.keys()))

    def test_getitem(self):
        composed_optim_wrapper = OptimizerWrapperDict(self.optimizers_wrappers)
        self.assertEqual(self.optimizers_wrappers['optim1'],
                         composed_optim_wrapper['optim1'])
        self.assertEqual(self.optimizers_wrappers['optim2'],
                         composed_optim_wrapper['optim2'])

    def test_len(self):
        composed_optim_wrapper = OptimizerWrapperDict(self.optimizers_wrappers)
        self.assertEqual(len(composed_optim_wrapper), 2)

    def test_contain(self):
        composed_optim_wrapper = OptimizerWrapperDict(self.optimizers_wrappers)
        self.assertIn('optim1', composed_optim_wrapper)

    def test_repr(self):
        composed_optim_wrapper = OptimizerWrapperDict(self.optimizers_wrappers)
        desc = repr(composed_optim_wrapper)
        self.assertRegex(desc, 'name: optim1')
