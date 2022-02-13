import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.testing._internal.common_utils import TestCase

from mmengine.optim.scheduler import (ConstantMomentum,
                                      CosineAnnealingMomentum,
                                      ExponentialMomentum, LinearMomentum,
                                      MultiStepMomentum, StepMomentum,
                                      _ParameterShceduler)


class ToyModel(torch.nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestParameterScheduler(TestCase):

    def __init__(self, methodName='runTest'):
        super(TestParameterScheduler, self).__init__(methodName)
        self.model = ToyModel()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.05)

    def test_invalid_optimizer(self):
        with self.assertRaisesRegex(TypeError, 'is not an Optimizer'):
            StepMomentum('invalid_optimizer', step_size=1)

    def test_wrong_resume(self):
        with self.assertRaises(KeyError):
            StepMomentum(self.optimizer, gamma=0.1, step_size=3, last_epoch=10)

    def test_scheduler_before_optim_warning(self):
        """warns if scheduler is used before optimizer."""

        def call_sch_before_optim():
            scheduler = StepMomentum(self.optimizer, gamma=0.1, step_size=3)
            scheduler.step()
            self.optimizer.step()

        # check warning doc link
        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                              call_sch_before_optim)

        # check warning when resume
        for i, group in enumerate(self.optimizer.param_groups):
            group['initial_momentum'] = 0.01

        def call_sch_before_optim_resume():
            scheduler = StepMomentum(
                self.optimizer, gamma=0.1, step_size=3, last_epoch=10)
            scheduler.step()
            self.optimizer.step()

        # check warning doc link
        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                              call_sch_before_optim_resume)

    def test_effective_interval(self):
        with self.assertRaisesRegex(ValueError,
                                    'end should be larger than begin'):
            StepMomentum(
                self.optimizer, gamma=0.1, step_size=3, begin=10, end=5)

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
        self.assertEqual(scheduler.get_last_value(),
                         scheduler_copy.get_last_value())

    def test_step_scheduler_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: StepMomentum(self.optimizer, gamma=0.1, step_size=3),
            lambda: StepMomentum(self.optimizer, gamma=0.01 / 2, step_size=1))

    def test_multi_step_scheduler_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: MultiStepMomentum(
                self.optimizer, gamma=0.1, milestones=[2, 5, 9]),
            lambda: MultiStepMomentum(
                self.optimizer, gamma=0.01, milestones=[1, 4, 6]))

    def test_exp_step_scheduler_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: ExponentialMomentum(self.optimizer, gamma=0.1),
            lambda: ExponentialMomentum(self.optimizer, gamma=0.01))

    def test_cosine_scheduler_state_dict(self):
        epochs = 10
        eta_min = 1e-10
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingMomentum(
                self.optimizer, T_max=epochs, eta_min=eta_min),
            lambda: CosineAnnealingMomentum(
                self.optimizer, T_max=epochs // 2, eta_min=eta_min / 2),
            epochs=epochs)

    def _test_scheduler_value(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, _ParameterShceduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                self.assertEqual(
                    target[epoch],
                    param_group['momentum'],
                    msg='Momentum is wrong in epoch {}: expected {}, got {}'.
                    format(epoch, target[epoch], param_group['momentum']),
                    atol=1e-5,
                    rtol=0)
            [scheduler.step() for scheduler in schedulers]

    def test_step_scheduler(self):
        # momentum = 0.05     if epoch < 3
        # momentum = 0.005    if 30 <= epoch < 6
        # momentum = 0.0005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005
                                                                    ] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = StepMomentum(self.optimizer, gamma=0.1, step_size=3)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_multi_step_scheduler(self):
        # momentum = 0.05     if epoch < 2
        # momentum = 0.005    if 2 <= epoch < 5
        # momentum = 0.0005   if epoch < 9
        # momentum = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005
                                                                    ] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepMomentum(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_constant_scheduler(self):
        with self.assertRaises(ValueError):
            ConstantMomentum(self.optimizer, factor=99, end=5)
        # momentum = 0.025     if epoch < 5
        # momentum = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 4 + [0.05] * 6
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantMomentum(self.optimizer, factor=1.0 / 2, end=5)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_linear_scheduler(self):
        with self.assertRaises(ValueError):
            LinearMomentum(self.optimizer, start_factor=10, end=900)
        with self.assertRaises(ValueError):
            LinearMomentum(self.optimizer, start_factor=-1, end=900)
        with self.assertRaises(ValueError):
            LinearMomentum(self.optimizer, end_factor=1.001, end=900)
        with self.assertRaises(ValueError):
            LinearMomentum(self.optimizer, end_factor=-0.00001, end=900)
        # momentum = 0.025     if epoch == 0
        # momentum = 0.03125   if epoch == 1
        # momentum = 0.0375    if epoch == 2
        # momentum = 0.04375   if epoch == 3
        # momentum = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 2
        iters = 4
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (
            epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearMomentum(
            self.optimizer, start_factor=start_factor, end=iters + 1)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_exp_scheduler(self):
        epochs = 10
        single_targets = [0.05 * (0.9**x) for x in range(epochs)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ExponentialMomentum(self.optimizer, gamma=0.9)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_cos_anneal_scheduler(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) *
            (1 + math.cos(math.pi * x / epochs)) / 2 for x in range(epochs)
        ]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = CosineAnnealingMomentum(
            self.optimizer, T_max=epochs, eta_min=eta_min)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_get_last_value(self):
        from torch.nn import Parameter
        epochs = 10
        optimizer = torch.optim.SGD(
            [Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
        targets = [[0.1] * 3 + [0.01] * 3 + [0.001] * 3 + [0.0001]]
        scheduler = StepMomentum(optimizer, 3, gamma=0.1)
        for epoch in range(epochs):
            result = scheduler.get_last_value()
            self.optimizer.step()
            scheduler.step()
            target = [t[epoch] for t in targets]
            for t, r in zip(target, result):
                self.assertEqual(
                    target,
                    result,
                    msg='Momentum is wrong in epoch {}: expected {}, got {}'.
                    format(epoch, t, r),
                    atol=1e-5,
                    rtol=0)
