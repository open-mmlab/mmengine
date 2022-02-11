import math
import warnings
import pytest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from unittest import TestCase

from mmengine.optim.schedule import (ConstantScheduler,
                                     CosineAnnealingScheduler, CyclicScheduler,
                                     ExponentialScheduler, LinearScheduler,
                                     MultiStepScheduler, OneCycleLR,
                                     StepScheduler)

ALL_SCHEDULERS = [
    ConstantScheduler, CosineAnnealingScheduler, CyclicScheduler,
    ExponentialScheduler, LinearScheduler, MultiStepScheduler, OneCycleLR,
    StepScheduler
]


class ToyModel(torch.nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestParameterScheduler(TestCase):
    """Unit test for ParameterScheduler.
    Some of the test cases are refered from https://github.com/pytorch/pytorch/blob/master/test/test_optim.py.
    """
    def __init__(self):
        self.model = ToyModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)


    def test_scheduler_before_optim_warning(self):
        """warns if scheduler is used before optimizer."""
        
        def call_sch_before_optim():
            scheduler = StepScheduler(
                self.optimizer, gamma=0.1, step_size=3)
            scheduler.step()
            self.optimizer.step()
        # check warning doc link
        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                              call_sch_before_optim)

        # check warning when resume
        for i, group in enumerate(self.optimizer.param_groups):
            group['initial_lr'] = 0.01
        
        def call_sch_before_optim_resume():
            scheduler = StepScheduler(
                self.optimizer, gamma=0.1, step_size=3, last_epoch=10)
            scheduler.step()
            self.optimizer.step()
        # check warning doc link
        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                              call_sch_before_optim_resume)
            

    def test_resume_scheduler_before_optim_warning(self):
        """warns if scheduler is used before optimizer after resume."""
        epochs = 6
        for i, group in enumerate(self.optimizer.param_groups):
            group['initial_lr'] = 0.01

        scheduler = StepScheduler(
            self.optimizer, gamma=0.1, step_size=3, last_epoch=10)
        
        # check warning doc link
        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                               self._call_sch_before_optim)

    def test_scheduler_before_optim_warning_with_overridden_optim_step(self):
        epochs = 35
        for i, group in enumerate(self.optimizer.param_groups):
            group['initial_lr'] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')  # allow any warning to be raised
            scheduler = StepScheduler(
                self.optimizer, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, 'No warning should be raised')

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.optimizer.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.optimizer.step = types.MethodType(new_step, self.optimizer)

        def call_sch_before_optim():
            for _ in range(epochs):
                scheduler.step()
                self.optimizer.step()

        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                              call_sch_before_optim)

    def test_new_pattern_no_warning(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')  # allow any warning to be raised
            scheduler = StepScheduler(self.optimizer, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, 'No warning should be raised')

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')  # allow any warning to be raised
            for _ in range(epochs):
                self.optimizer.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, 'No warning should be raised')

    def test_new_pattern_no_warning_with_arg(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')  # allow any warning to be raised
            scheduler = StepScheduler(self.optimizer, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, 'No warning should be raised')

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')  # allow any warning to be raised
            for _ in range(epochs):
                self.optimizer.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, 'No warning should be raised')

    def test_new_pattern_no_warning_with_overridden_optim_step(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')  # allow any warning to be raised
            scheduler = StepScheduler(self.optimizer, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, 'No warning should be raised')

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.optimizer.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.optimizer.step = types.MethodType(new_step, self.optimizer)

        def new_pattern():
            for e in range(epochs):
                self.optimizer.step()
                scheduler.step()

        self.assertWarnsRegex(UserWarning,
                              r'`optimizer.step\(\)` has been overridden',
                              new_pattern)

    def _test_lr_is_constant_for_constant_epoch(self, scheduler):
        l = []

        for _ in range(10):
            scheduler.optimizer.step()
            with warnings.catch_warnings(record=True) as w:
                scheduler.step(2)
                self._check_warning_is_epoch_deprecation_warning(w)

            l.append(self.optimizer.param_groups[0]['lr'])
        self.assertEqual(min(l), max(l))

    def test_step_lr_is_constant_for_constant_epoch(self):
        scheduler = StepScheduler(self.optimizer, 2)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_exponential_lr_is_constant_for_constant_epoch(self):
        scheduler = ExponentialScheduler(self.optimizer, gamma=0.9)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_ConstantScheduler_is_constant_for_constant_epoch(self):
        scheduler = ConstantScheduler(self.optimizer)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_linear_LinearScheduler_is_constant_for_constant_epoch(self):
        scheduler = LinearScheduler(self.optimizer)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_step_lr(self):
        # lr = 0.05     if epoch < 3
        # lr = 0.005    if 30 <= epoch < 6
        # lr = 0.0005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005
                                                                    ] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = StepScheduler(self.optimizer, gamma=0.1, step_size=3)
        self._test(scheduler, targets, epochs)

    def test_get_last_lr_step_lr(self):
        from torch.nn import Parameter
        epochs = 10
        optimizer = torch.optim.SGD(
            [Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
        targets = [[0.1] * 3 + [0.01] * 3 + [0.001] * 3 + [0.0001]]
        scheduler = torch.optim.lr_scheduler.StepScheduler(
            optimizer, 3, gamma=0.1)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_get_last_lr_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if 5 <= epoch < 9
        # lr = 0.00005   if 9 <= epoch
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005
                                                                    ] * 1
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepScheduler(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005
                                                                    ] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepScheduler(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        self._test(scheduler, targets, epochs)

    def test_multi_step_lr_with_epoch(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005
                                                                    ] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepScheduler(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        self._test_with_epoch(scheduler, targets, epochs)

    def test_get_last_lr_ConstantScheduler(self):
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantScheduler(
            self.optimizer, factor=1.0 / 2, total_iters=5)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_get_last_lr_LinearScheduler(self):
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 4
        end_factor = 3. / 5
        iters = 4
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters
            for i in range(iters)
        ]
        single_targets = [x * 0.05
                          for x in interpolation] + [0.05 * end_factor] * (
                              epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearScheduler(
            self.optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_ConstantScheduler(self):
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantScheduler(
            self.optimizer, factor=1.0 / 2, total_iters=5)
        self._test(scheduler, targets, epochs)

    def test_LinearScheduler(self):
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 2
        iters = 4
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (
            epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearScheduler(
            self.optimizer, start_factor=start_factor, total_iters=iters)
        self._test(scheduler, targets, epochs)

    def test_ConstantScheduler_with_epoch(self):
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantScheduler(
            self.optimizer, factor=1.0 / 2, total_iters=5)
        self._test_with_epoch(scheduler, targets, epochs)

    def test_LinearScheduler_with_epoch(self):
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 2
        end_factor = 1.
        iters = 4
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters
            for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (
            epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearScheduler(
            self.optimizer, start_factor=start_factor, total_iters=iters)
        self._test_with_epoch(scheduler, targets, epochs)

    def test_exp_lr(self):
        epochs = 10
        single_targets = [0.05 * (0.9**x) for x in range(epochs)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ExponentialScheduler(self.optimizer, gamma=0.9)
        self._test(scheduler, targets, epochs)

    def test_cos_anneal_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) *
            (1 + math.cos(math.pi * x / epochs)) / 2 for x in range(epochs)
        ]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = CosineAnnealingScheduler(
            self.optimizer, T_max=epochs, eta_min=eta_min)
        self._test(scheduler, targets, epochs)

    def test_closed_form_step_lr(self):
        scheduler = StepScheduler(self.optimizer, gamma=0.1, step_size=3)
        closed_form_scheduler = StepScheduler(
            self.optimizer, gamma=0.1, step_size=3)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_LinearScheduler(self):
        scheduler = LinearScheduler(
            self.optimizer,
            start_factor=1.0 / 3,
            end_factor=0.7,
            total_iters=4)
        closed_form_scheduler = LinearScheduler(
            self.optimizer,
            start_factor=1.0 / 3,
            end_factor=0.7,
            total_iters=4)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_ConstantScheduler(self):
        scheduler = ConstantScheduler(
            self.optimizer, factor=1.0 / 3, total_iters=4)
        closed_form_scheduler = ConstantScheduler(
            self.optimizer, factor=1.0 / 3, total_iters=4)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_multi_step_lr(self):
        scheduler = MultiStepScheduler(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        closed_form_scheduler = MultiStepScheduler(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_exp_lr(self):
        scheduler = ExponentialScheduler(self.optimizer, gamma=0.9)
        closed_form_scheduler = ExponentialScheduler(self.optimizer, gamma=0.9)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_cos_anneal_lr(self):
        eta_min = 1e-10
        epochs = 20
        T_max = 5
        scheduler = CosineAnnealingScheduler(
            self.optimizer, T_max=T_max, eta_min=eta_min)
        closed_form_scheduler = CosineAnnealingScheduler(
            self.optimizer, T_max=T_max, eta_min=eta_min)
        self._test_against_closed_form(scheduler, closed_form_scheduler,
                                       epochs)

    def test_compound_step_and_multistep_lr(self):
        epochs = 10
        schedulers = [None] * 2
        schedulers[0] = StepScheduler(self.optimizer, gamma=0.1, step_size=3)
        schedulers[1] = MultiStepScheduler(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        targets = [[0.05] * 2 + [0.005] * 1 + [5e-4] * 2 + [5e-5] +
                   [5e-6] * 3 + [5e-8]]
        self._test(schedulers, targets, epochs)

    def test_compound_step_and_exp_lr(self):
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(3)]
        single_targets += [0.005 * (0.9**x) for x in range(3, 6)]
        single_targets += [0.0005 * (0.9**x) for x in range(6, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 12)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = StepScheduler(self.optimizer, gamma=0.1, step_size=3)
        schedulers[1] = ExponentialScheduler(self.optimizer, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_exp_and_multistep_lr(self):
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(2)]
        single_targets += [0.005 * (0.9**x) for x in range(2, 5)]
        single_targets += [0.0005 * (0.9**x) for x in range(5, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 11)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = MultiStepScheduler(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = ExponentialScheduler(self.optimizer, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_exp_and_LinearScheduler(self):
        epochs = 10
        iters = 4
        start_factor = 0.4
        end_factor = 0.9
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(11)]
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (
                end_factor - start_factor)
        for i in range(iters, 11):
            single_targets[i] *= end_factor
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = LinearScheduler(
            self.optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters)
        schedulers[1] = ExponentialScheduler(self.optimizer, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_step_and_ConstantScheduler(self):
        epochs = 10
        iters = 4
        factor = 0.4
        schedulers = [None] * 2
        single_targets = [0.05 * 0.4] * 3 + [
            0.005 * 0.4
        ] + [0.005] * 2 + [0.0005] * 3 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = StepScheduler(self.optimizer, gamma=0.1, step_size=3)
        schedulers[1] = ConstantScheduler(
            self.optimizer, factor=0.4, total_iters=4)
        self._test(schedulers, targets, epochs)

    def test_compound_LinearScheduler_and_multistep_lr(self):
        epochs = 10
        iters = 4
        start_factor = 0.4
        schedulers = [None] * 2
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005
                                                                    ] * 2
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = MultiStepScheduler(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = LinearScheduler(
            self.optimizer, start_factor=start_factor, total_iters=iters)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_step_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) *
            (1 + math.cos(math.pi * x / epochs)) / 2 for x in range(epochs)
        ]
        single_targets = [
            x * 0.1**(i // 3) for i, x in enumerate(single_targets)
        ]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingScheduler(
            self.optimizer, T_max=epochs, eta_min=eta_min)
        schedulers[1] = StepScheduler(self.optimizer, gamma=0.1, step_size=3)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_multistep_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) *
            (1 + math.cos(math.pi * x / epochs)) / 2 for x in range(epochs)
        ]
        multipliers = [1] * 2 + [0.1] * 3 + [0.01] * 4 + [0.001]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingScheduler(
            self.optimizer, T_max=epochs, eta_min=eta_min)
        schedulers[1] = MultiStepScheduler(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_LinearScheduler(self):
        epochs = 10
        iters = 4
        start_factor = 0.4
        eta_min = 1e-10
        schedulers = [None] * 2
        single_targets = [
            eta_min + (0.05 - eta_min) *
            (1 + math.cos(math.pi * x / epochs)) / 2 for x in range(epochs)
        ]
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = LinearScheduler(
            self.optimizer, start_factor=start_factor, total_iters=iters)
        schedulers[1] = CosineAnnealingScheduler(
            self.optimizer, T_max=epochs, eta_min=eta_min)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_exp_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) *
            (1 + math.cos(math.pi * x / epochs)) / 2 for x in range(epochs)
        ]
        multipliers = [0.1**i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingScheduler(
            self.optimizer, T_max=epochs, eta_min=eta_min)
        schedulers[1] = ExponentialScheduler(self.optimizer, gamma=0.1)
        self._test(schedulers, targets, epochs)

    def test_cycle_lr_invalid_mode(self):
        with self.assertRaises(ValueError):
            scheduler = CyclicScheduler(
                self.optimizer, base_lr=0, max_lr=0, mode='CATS')

    def test_cycle_lr_triangular_mode_one_lr(self):
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        momentum_target = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target))

    def test_cycle_lr_triangular_mode_one_lr_no_momentum(self):
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        momentum_target = [self.optimizer.defaults['momentum']
                           ] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=False,
            mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target))

    def test_cycle_lr_triangular2_mode_one_lr(self):
        lr_target = [
            1, 2, 3, 4, 5, 4, 3, 2, 1, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1,
            1.25, 1.50, 1.75, 2.00, 1.75
        ]
        momentum_target = [
            5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.5, 4.0, 3.5, 3.0,
            3.5, 4.0, 4.5, 5.0, 4.75, 4.5, 4.25, 4.0, 4.25
        ]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode='triangular2')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target))

    def test_cycle_lr_exp_range_mode_one_lr(self):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        lr_target = [
            base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)
        ]
        momentum_target = [
            max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)
        ]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=base_lr,
            max_momentum=max_lr,
            mode='exp_range',
            gamma=gamma)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target))

    def test_cycle_lr_triangular_mode(self):
        lr_target_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_target_2 = [x + 1 for x in lr_target_1]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        momentum_target_2 = [x + 1 for x in momentum_target_1]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=[1, 2],
            max_lr=[5, 6],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[1, 2],
            max_momentum=[5, 6],
            mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target_1))

    def test_cycle_lr_triangular2_mode(self):
        lr_target_1 = [
            1, 2, 3, 4, 5, 4, 3, 2, 1, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1,
            1.25, 1.50, 1.75, 2.00, 1.75
        ]
        lr_target_2 = [x + 2 for x in lr_target_1]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [
            5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.5, 4.0, 3.5, 3.0,
            3.5, 4.0, 4.5, 5.0, 4.75, 4.5, 4.25, 4.0, 4.25
        ]
        momentum_target_2 = [x + 2 for x in momentum_target_1]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=[1, 3],
            max_lr=[5, 7],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[1, 3],
            max_momentum=[5, 7],
            mode='triangular2')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target_1))

    def test_cycle_lr_exp_range_mode(self):
        base_lr_1, max_lr_1 = 1, 5
        base_lr_2, max_lr_2 = 5, 12

        diff_lr_1 = max_lr_1 - base_lr_1
        diff_lr_2 = max_lr_2 - base_lr_2

        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        lr_target_1 = [
            base_lr_1 + x * diff_lr_1 * gamma**i for i, x in enumerate(xs)
        ]
        lr_target_2 = [
            base_lr_2 + x * diff_lr_2 * gamma**i for i, x in enumerate(xs)
        ]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [
            max_lr_1 - x * diff_lr_1 * gamma**i for i, x in enumerate(xs)
        ]
        momentum_target_2 = [
            max_lr_2 - x * diff_lr_2 * gamma**i for i, x in enumerate(xs)
        ]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=[base_lr_1, base_lr_2],
            max_lr=[max_lr_1, max_lr_2],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[base_lr_1, base_lr_2],
            max_momentum=[max_lr_1, max_lr_2],
            mode='exp_range',
            gamma=gamma)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target_1))

    def test_cycle_lr_triangular_mode_step_size_up_down(self):
        lr_target = [
            1.0, 2.0, 3.0, 4.0, 5.0, 13.0 / 3, 11.0 / 3, 9.0 / 3, 7.0 / 3,
            5.0 / 3, 1.0
        ]
        lr_targets = [lr_target, lr_target]
        momentum_target = [
            5.0, 4.0, 3.0, 2.0, 1.0, 5.0 / 3, 7.0 / 3, 3.0, 11.0 / 3, 13.0 / 3,
            5.0
        ]
        momentum_targets = [momentum_target, momentum_target]

        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target))

    def test_cycle_lr_triangular2_mode_step_size_up_down(self):
        lr_base_target = ([
            1.0, 3.0, 5.0, 13.0 / 3, 11.0 / 3, 9.0 / 3, 7.0 / 3, 5.0 / 3, 1.0,
            2.0, 3.0, 8.0 / 3, 7.0 / 3, 6.0 / 3, 5.0 / 3, 4.0 / 3, 1.0,
            3.0 / 2, 2.0, 11.0 / 6, 10.0 / 6, 9.0 / 6, 8.0 / 6, 7.0 / 6
        ])
        momentum_base_target = ([
            5.0, 3.0, 1.0, 5.0 / 3, 7.0 / 3, 3.0, 11.0 / 3, 13.0 / 3, 5.0, 4.0,
            3.0, 10.0 / 3, 11.0 / 3, 4.0, 13.0 / 3, 14.0 / 3, 5.0, 4.5, 4.0,
            25.0 / 6, 13.0 / 3, 4.5, 14.0 / 3, 29.0 / 6
        ])
        deltas = [2 * i for i in range(0, 2)]
        base_lrs = [1 + delta for delta in deltas]
        max_lrs = [5 + delta for delta in deltas]
        lr_targets = [[x + delta for x in lr_base_target] for delta in deltas]
        momentum_targets = [[x + delta for x in momentum_base_target]
                            for delta in deltas]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=base_lrs,
            max_lr=max_lrs,
            step_size_up=2,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=base_lrs,
            max_momentum=max_lrs,
            mode='triangular2')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_base_target))

    def test_cycle_lr_exp_range_mode_step_size_up_down(self):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = ([
            0.0, 0.5, 1.0, 5.0 / 6, 4.0 / 6, 3.0 / 6, 2.0 / 6, 1.0 / 6, 0.0,
            0.5, 1.0, 5.0 / 6, 4.0 / 6
        ])
        lr_target = [
            base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)
        ]
        lr_targets = [lr_target, lr_target]
        momentum_target = [
            max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)
        ]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=2,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=base_lr,
            max_momentum=max_lr,
            mode='exp_range',
            gamma=gamma)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target))

    def test_cycle_lr_with_momentumless_optimizer(self):
        # Note [Temporarily set optimizer to Adam]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The TestLRScheduler object carries around an SGD optimizer to avoid having to
        # instantiate one for every test. This gets in the way for our very specific case
        # in which we need to use Adam (or really any optimizer that doesn't use momentum)
        # in order to test that the momentum bug in CyclicScheduler is fixed (the bug is described
        # in more detail in https://github.com/pytorch/pytorch/issues/19003 ).
        old_opt = self.optimizer
        self.optimizer = optim.Adam([{
            'params': self.model.conv1.parameters()
        }, {
            'params': self.model.conv2.parameters(),
            'lr': 0.5
        }],
                                    lr=0.05)

        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        momentum_target = [None] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicScheduler(
            self.optimizer,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=False,
            mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets,
                            len(lr_target))

        self.optimizer = old_opt  # set optimizer back to SGD

    def test_cycle_lr_cycle_momentum_fail_with_momentumless_optimizer(self):
        with self.assertRaises(ValueError):
            adam_opt = optim.Adam(self.model.parameters())
            scheduler = CyclicScheduler(
                adam_opt, base_lr=1, max_lr=5, cycle_momentum=True)

    def test_onecycle_lr_invalid_anneal_strategy(self):
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=1e-3,
                total_steps=10,
                anneal_strategy='CATS')

    def test_onecycle_lr_invalid_pct_start(self):
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(
                self.optimizer, max_lr=1e-3, total_steps=10, pct_start=1.1)

    def test_onecycle_lr_cannot_calculate_total_steps(self):
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.optimizer, max_lr=1e-3)

    def test_onecycle_lr_linear_annealing(self):
        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy='linear')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_linear_annealing_three_phases(self):
        lr_target = [1, 9, 17, 25, 17, 9, 1, 0.75, 0.5, 0.25]
        momentum_target = [22, 15, 8, 1, 8, 15, 22, 22, 22, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=25,
            div_factor=25,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy='linear',
            pct_start=0.4,
            final_div_factor=4,
            three_phase=True)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_cosine_annealing(self):

        def annealing_cos(start, end, pct):
            cos_out = math.cos(math.pi * pct) + 1
            return end + (start - end) / 2.0 * cos_out

        lr_target = [
            1, 13, 25,
            annealing_cos(25, 0.5, 1 / 7.0),
            annealing_cos(25, 0.5, 2 / 7.0),
            annealing_cos(25, 0.5, 3 / 7.0),
            annealing_cos(25, 0.5, 4 / 7.0),
            annealing_cos(25, 0.5, 5 / 7.0),
            annealing_cos(25, 0.5, 6 / 7.0), 0.5
        ]
        momentum_target = [
            22, 11.5, 1,
            annealing_cos(1, 22, 1 / 7.0),
            annealing_cos(1, 22, 2 / 7.0),
            annealing_cos(1, 22, 3 / 7.0),
            annealing_cos(1, 22, 4 / 7.0),
            annealing_cos(1, 22, 5 / 7.0),
            annealing_cos(1, 22, 6 / 7.0), 22
        ]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_cycle_lr_with_adam(self):
        old_opt = self.optimizer
        self.optimizer = optim.Adam([{
            'params': self.model.conv1.parameters()
        }, {
            'params': self.model.conv2.parameters(),
            'lr': 0.5
        }],
                                    lr=0.05)

        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy='linear')
        self._test_cycle_lr(
            scheduler, lr_targets, momentum_targets, 10, use_beta1=True)
        self.optimizer = old_opt  # set optimizer back to SGD

    def test_lambda_lr(self):
        epochs = 10
        self.optimizer.param_groups[0]['lr'] = 0.05
        self.optimizer.param_groups[1]['lr'] = 0.4
        targets = [[0.05 * (0.9**x) for x in range(epochs)],
                   [0.4 * (0.8**x) for x in range(epochs)]]
        scheduler = LambdaLR(
            self.optimizer, lr_lambda=[lambda x1: 0.9**x1, lambda x2: 0.8**x2])
        self._test(scheduler, targets, epochs)

    def test_multiplicative_lr(self):
        epochs = 10
        self.optimizer.param_groups[0]['lr'] = 0.05
        self.optimizer.param_groups[1]['lr'] = 0.4
        targets = [[0.05 * (0.9**x) for x in range(epochs)],
                   [0.4 * (0.8**x) for x in range(epochs)]]
        scheduler = MultiplicativeLR(
            self.optimizer, lr_lambda=[lambda x1: 0.9, lambda x2: 0.8])
        self._test(scheduler, targets, epochs)

    def test_step_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: StepScheduler(self.optimizer, gamma=0.1, step_size=3),
            lambda: StepScheduler(self.optimizer, gamma=0.01 / 2, step_size=1))

    def test_multi_step_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: MultiStepScheduler(
                self.optimizer, gamma=0.1, milestones=[2, 5, 9]),
            lambda: MultiStepScheduler(
                self.optimizer, gamma=0.01, milestones=[1, 4, 6]))

    def test_exp_step_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: ExponentialScheduler(self.optimizer, gamma=0.1),
            lambda: ExponentialScheduler(self.optimizer, gamma=0.01))

    def test_cosine_lr_state_dict(self):
        epochs = 10
        eta_min = 1e-10
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingScheduler(
                self.optimizer, T_max=epochs, eta_min=eta_min),
            lambda: CosineAnnealingScheduler(
                self.optimizer, T_max=epochs // 2, eta_min=eta_min / 2),
            epochs=epochs)

    def test_lambda_lr_state_dict_fn(self):
        scheduler = LambdaLR(self.optimizer, lr_lambda=lambda x: x)
        state = scheduler.state_dict()
        self.assertIsNone(state['lr_lambdas'][0])

        scheduler_copy = LambdaLR(self.optimizer, lr_lambda=lambda x: x)
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__.keys():
            if key not in {'optimizer', 'lr_lambdas'}:
                self.assertEqual(scheduler.__dict__[key],
                                 scheduler_copy.__dict__[key])

    def test_lambda_lr_state_dict_obj(self):
        scheduler = LambdaLR(self.optimizer, lr_lambda=LambdaLRTestObject(10))
        state = scheduler.state_dict()
        self.assertIsNotNone(state['lr_lambdas'][0])

        scheduler_copy = LambdaLR(
            self.optimizer, lr_lambda=LambdaLRTestObject(-1))
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__.keys():
            if key not in {'optimizer'}:
                self.assertEqual(scheduler.__dict__[key],
                                 scheduler_copy.__dict__[key])

    def _check_scheduler_state_dict(self, constr, constr2, epochs=10):
        scheduler = constr()
        for _ in range(epochs):
            scheduler.optimizer.step()
            scheduler.step()
        scheduler_copy = constr2()
        scheduler_copy.load_state_dict(scheduler.state_dict())
        for key in scheduler.__dict__.keys():
            if key != 'optimizer':
                self.assertEqual(scheduler.__dict__[key],
                                 scheduler_copy.__dict__[key])
        self.assertEqual(scheduler.get_last_lr(), scheduler_copy.get_last_lr())

    def _test_get_last_lr(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, _LRScheduler):
            schedulers = [schedulers]
        optimizers = {scheduler.optimizer for scheduler in schedulers}
        for epoch in range(epochs):
            result = [scheduler.get_last_lr() for scheduler in schedulers]
            [optimizer.step() for optimizer in optimizers]
            [scheduler.step() for scheduler in schedulers]
            target = [[t[epoch] for t in targets]] * len(schedulers)
            for t, r in zip(target, result):
                self.assertEqual(
                    target,
                    result,
                    msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                        epoch, t, r),
                    atol=1e-5,
                    rtol=0)

    def _test_with_epoch(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, _LRScheduler):
            schedulers = [schedulers]
        optimizers = {scheduler.optimizer for scheduler in schedulers}
        for epoch in range(epochs):
            [optimizer.step() for optimizer in optimizers]
            with warnings.catch_warnings(record=True) as w:
                [scheduler.step(epoch) for scheduler in schedulers
                 ]  # step before assert: skip initial lr
                self._check_warning_is_epoch_deprecation_warning(
                    w, num_warnings=len(schedulers))
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                self.assertEqual(
                    target[epoch],
                    param_group['lr'],
                    msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                        epoch, target[epoch], param_group['lr']),
                    atol=1e-5,
                    rtol=0)

    def _test(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, _LRScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                self.assertEqual(
                    target[epoch],
                    param_group['lr'],
                    msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                        epoch, target[epoch], param_group['lr']),
                    atol=1e-5,
                    rtol=0)
            [scheduler.step() for scheduler in schedulers]

    def _test_CosineAnnealingWarmRestarts(self, scheduler, targets, epochs=10):
        for index, epoch in enumerate(torch.arange(0, epochs, 0.1)):
            epoch = round(epoch.item(), 1)
            scheduler.step(epoch)
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                self.assertEqual(
                    target[index],
                    param_group['lr'],
                    msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                        epoch, target[index], param_group['lr']),
                    atol=1e-5,
                    rtol=0)

    def _test_interleaved_CosineAnnealingWarmRestarts(self, scheduler, targets,
                                                      epochs):
        for index, epoch in enumerate(epochs):
            scheduler.step(epoch)
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                self.assertEqual(
                    target[index],
                    param_group['lr'],
                    msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                        epoch, target[index], param_group['lr']),
                    atol=1e-5,
                    rtol=0)

    def _test_against_closed_form(self,
                                  scheduler,
                                  closed_form_scheduler,
                                  epochs=10):
        self.setUp()
        targets = []
        for epoch in range(epochs):
            closed_form_scheduler.optimizer.step()
            with warnings.catch_warnings(record=True) as w:
                closed_form_scheduler.step(epoch)
                self._check_warning_is_epoch_deprecation_warning(w)
            targets.append(
                [group['lr'] for group in self.optimizer.param_groups])
        self.setUp()
        for epoch in range(epochs):
            self.optimizer.step()
            scheduler.step()
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.assertEqual(
                    targets[epoch][i],
                    param_group['lr'],
                    msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                        epoch, targets[epoch][i], param_group['lr']),
                    atol=1e-5,
                    rtol=0)

    def _test_cycle_lr(self,
                       scheduler,
                       lr_targets,
                       momentum_targets,
                       batch_iterations,
                       verbose=False,
                       use_beta1=False):
        for batch_num in range(batch_iterations):
            if verbose:
                if 'momentum' in self.optimizer.param_groups[0].keys():
                    print('batch{}:\tlr={},momentum={}'.format(
                        batch_num, self.optimizer.param_groups[0]['lr'],
                        self.optimizer.param_groups[0]['momentum']))
                elif use_beta1 and 'betas' in self.optimizer.param_groups[
                        0].keys():
                    print('batch{}:\tlr={},beta1={}'.format(
                        batch_num, self.optimizer.param_groups[0]['lr'],
                        self.optimizer.param_groups[0]['betas'][0]))
                else:
                    print('batch{}:\tlr={}'.format(
                        batch_num, self.optimizer.param_groups[0]['lr']))

            for param_group, lr_target, momentum_target in zip(
                    self.optimizer.param_groups, lr_targets, momentum_targets):
                self.assertEqual(
                    lr_target[batch_num],
                    param_group['lr'],
                    msg='LR is wrong in batch_num {}: expected {}, got {}'.
                    format(batch_num, lr_target[batch_num], param_group['lr']),
                    atol=1e-5,
                    rtol=0)

                if use_beta1 and 'betas' in param_group.keys():
                    self.assertEqual(
                        momentum_target[batch_num],
                        param_group['betas'][0],
                        msg='Beta1 is wrong in batch_num {}: expected {}, got {}'
                        .format(batch_num, momentum_target[batch_num],
                                param_group['betas'][0]),
                        atol=1e-5,
                        rtol=0)
                elif 'momentum' in param_group.keys():
                    self.assertEqual(
                        momentum_target[batch_num],
                        param_group['momentum'],
                        msg=
                        'Momentum is wrong in batch_num {}: expected {}, got {}'
                        .format(batch_num, momentum_target[batch_num],
                                param_group['momentum']),
                        atol=1e-5,
                        rtol=0)
            self.optimizer.step()
            scheduler.step()

    def test_cosine_then_cyclic(self):
        # https://github.com/pytorch/pytorch/issues/21965

        max_lr = 0.3
        base_lr = 0.1
        optim_lr = 0.5

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=optim_lr)
        lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingScheduler(
            optimizer, T_max=20, eta_min=0.1)
        lr_scheduler_2 = torch.optim.lr_scheduler.CyclicScheduler(
            optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=1,
            step_size_down=3)

        for i in range(40):
            optimizer.step()
            if i <= lr_scheduler_1.T_max:
                lr_scheduler_1.step()
            else:
                lr_scheduler_2.step()
            last_lr = optimizer.param_groups[0]['lr']

        self.assertLessEqual(last_lr, max_lr)
