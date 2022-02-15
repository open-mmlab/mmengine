import math
from unittest import TestCase

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.testing import assert_allclose

# yapf: disable
from mmengine.optim.scheduler import (ConstantParamScheduler,
                                      CosineAnnealingParamScheduler,
                                      ExponentialParamScheduler,
                                      LinearParamScheduler,
                                      MultiStepParamScheduler,
                                      StepParamScheduler, _ParamScheduler)

# yapf: enable


class ToyModel(torch.nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestParameterScheduler(TestCase):

    def setUp(self) -> None:
        self.model = ToyModel()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.01)

    def test_invalid_optimizer(self):
        with self.assertRaisesRegex(TypeError, 'is not an Optimizer'):
            StepParamScheduler('invalid_optimizer', 'lr', step_size=1)

    def test_wrong_resume(self):
        """test optimizer and scheduler are not both resumed."""
        with self.assertRaises(KeyError):
            StepParamScheduler(
                self.optimizer,
                param_name='lr',
                gamma=0.1,
                step_size=3,
                last_step=10)

    def test_scheduler_before_optim_warning(self):
        """warns if scheduler is used before optimizer."""

        def call_sch_before_optim():
            scheduler = StepParamScheduler(
                self.optimizer, param_name='lr', gamma=0.1, step_size=3)
            scheduler.step()
            self.optimizer.step()

        # check warning doc link
        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                              call_sch_before_optim)

        # check warning when resume
        for i, group in enumerate(self.optimizer.param_groups):
            group['initial_lr'] = 0.01

        def call_sch_before_optim_resume():
            scheduler = StepParamScheduler(
                self.optimizer,
                param_name='lr',
                gamma=0.1,
                step_size=3,
                last_step=10)
            scheduler.step()
            self.optimizer.step()

        # check warning doc link
        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                              call_sch_before_optim_resume)

    def test_get_last_value(self):
        from torch.nn import Parameter
        epochs = 10
        optimizer = torch.optim.SGD(
            [Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
        targets = [[0.1] * 3 + [0.01] * 3 + [0.001] * 3 + [0.0001]]
        scheduler = StepParamScheduler(optimizer, 'lr', 3, gamma=0.1)
        for epoch in range(epochs):
            result = scheduler.get_last_value()
            self.optimizer.step()
            scheduler.step()
            target = [t[epoch] for t in targets]
            for t, r in zip(target, result):
                assert_allclose(
                    target,
                    result,
                    msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                        epoch, t, r),
                    atol=1e-5,
                    rtol=0)

    def test_scheduler_step_count(self):
        iteration = 10
        scheduler = StepParamScheduler(
            self.optimizer, param_name='lr', gamma=0.1, step_size=3)
        self.assertEqual(scheduler.last_step, 0)
        target = [i + 1 for i in range(iteration)]
        step_counts = []
        for i in range(iteration):
            self.optimizer.step()
            scheduler.step()
            step_counts.append(scheduler.last_step)
        self.assertEqual(step_counts, target)

    def test_effective_interval(self):
        # check invalid begin end
        with self.assertRaisesRegex(ValueError,
                                    'end should be larger than begin'):
            StepParamScheduler(
                self.optimizer,
                param_name='lr',
                gamma=0.1,
                step_size=3,
                begin=10,
                end=5)

    def test_invalid_param_name(self):
        with self.assertRaises(KeyError):
            StepParamScheduler(
                self.optimizer, param_name='invalid_name', step_size=10)

    def _test_scheduler_value(self,
                              schedulers,
                              targets,
                              epochs=10,
                              param_name='lr'):
        if isinstance(schedulers, _ParamScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                assert_allclose(
                    target[epoch],
                    param_group[param_name],
                    msg='{} is wrong in epoch {}: expected {}, got {}'.format(
                        param_name, epoch, target[epoch],
                        param_group[param_name]),
                    atol=1e-5,
                    rtol=0)
            [scheduler.step() for scheduler in schedulers]

    def test_step_scheduler(self):
        # lr = 0.05     if epoch < 3
        # lr = 0.005    if 30 <= epoch < 6
        # lr = 0.0005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005
                                                                    ] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = StepParamScheduler(
            self.optimizer,
            param_name='lr',
            gamma=0.1,
            step_size=3,
            verbose=True)
        self._test_scheduler_value(scheduler, targets, epochs)

        # momentum = 0.01     if epoch < 2
        # momentum = 0.001    if 2 <= epoch < 4
        epochs = 4
        single_targets = [0.01] * 2 + [0.001] * 2
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = StepParamScheduler(
            self.optimizer, param_name='momentum', gamma=0.1, step_size=2)
        self._test_scheduler_value(
            scheduler, targets, epochs, param_name='momentum')

    def test_multi_step_scheduler(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005
                                                                    ] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepParamScheduler(
            self.optimizer, param_name='lr', gamma=0.1, milestones=[2, 5, 9])
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_constant_scheduler(self):
        # factor should between 0~1
        with self.assertRaises(ValueError):
            ConstantParamScheduler(self.optimizer, param_name='lr', factor=99)

        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 4 + [0.05] * 6
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantParamScheduler(
            self.optimizer, param_name='lr', factor=1.0 / 2, end=5)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_linear_scheduler(self):
        with self.assertRaises(ValueError):
            LinearParamScheduler(
                self.optimizer, param_name='lr', start_factor=10, end=900)
        with self.assertRaises(ValueError):
            LinearParamScheduler(
                self.optimizer, param_name='lr', start_factor=-1, end=900)
        with self.assertRaises(ValueError):
            LinearParamScheduler(
                self.optimizer, param_name='lr', end_factor=1.001, end=900)
        with self.assertRaises(ValueError):
            LinearParamScheduler(
                self.optimizer, param_name='lr', end_factor=-0.00001, end=900)
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
        scheduler = LinearParamScheduler(
            self.optimizer,
            param_name='lr',
            start_factor=start_factor,
            end=iters + 1)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_exp_scheduler(self):
        epochs = 10
        single_targets = [0.05 * (0.9**x) for x in range(epochs)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ExponentialParamScheduler(
            self.optimizer, param_name='lr', gamma=0.9)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_cos_anneal_scheduler(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) *
            (1 + math.cos(math.pi * x / epochs)) / 2 for x in range(epochs)
        ]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = CosineAnnealingParamScheduler(
            self.optimizer, param_name='lr', T_max=epochs, eta_min=eta_min)
        self._test_scheduler_value(scheduler, targets, epochs)

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
            lambda: StepParamScheduler(
                self.optimizer, param_name='lr', gamma=0.1, step_size=3),
            lambda: StepParamScheduler(
                self.optimizer, param_name='lr', gamma=0.01 / 2, step_size=1))

    def test_multi_step_scheduler_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: MultiStepParamScheduler(
                self.optimizer,
                param_name='lr',
                gamma=0.1,
                milestones=[2, 5, 9]), lambda: MultiStepParamScheduler(
                    self.optimizer,
                    param_name='lr',
                    gamma=0.01,
                    milestones=[1, 4, 6]))

    def test_exp_scheduler_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: ExponentialParamScheduler(
                self.optimizer, param_name='lr', gamma=0.1),
            lambda: ExponentialParamScheduler(
                self.optimizer, param_name='lr', gamma=0.01))

    def test_cosine_scheduler_state_dict(self):
        epochs = 10
        eta_min = 1e-10
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingParamScheduler(
                self.optimizer, param_name='lr', T_max=epochs, eta_min=eta_min
            ),
            lambda: CosineAnnealingParamScheduler(
                self.optimizer,
                param_name='lr',
                T_max=epochs // 2,
                eta_min=eta_min / 2),
            epochs=epochs)

    def test_linear_scheduler_state_dict(self):
        epochs = 10
        self._check_scheduler_state_dict(
            lambda: LinearParamScheduler(
                self.optimizer, param_name='lr', start_factor=1 / 3),
            lambda: LinearParamScheduler(
                self.optimizer,
                param_name='lr',
                start_factor=0,
                end_factor=0.3),
            epochs=epochs)

    def test_multi_scheduler_without_overlap(self):
        epochs = 12
        single_targets = [0.025, 0.03125, 0.0375, 0.04375
                          ] + [0.05] * 4 + [0.005] * 3 + [0.0005] * 1
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler1 = LinearParamScheduler(
            self.optimizer,
            param_name='lr',
            start_factor=1 / 2,
            begin=0,
            end=5)
        scheduler2 = MultiStepParamScheduler(
            self.optimizer,
            param_name='lr',
            gamma=0.1,
            milestones=[3, 6],
            begin=5,
            end=12)
        self._test_scheduler_value([scheduler1, scheduler2], targets, epochs)

    def test_multi_scheduler_with_overlap(self):
        epochs = 10
        single_targets = [0.025, 0.03125, 0.0375, 0.004375
                          ] + [0.005] * 2 + [0.0005] * 3 + [0.00005] * 1
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler1 = LinearParamScheduler(
            self.optimizer,
            param_name='lr',
            start_factor=1 / 2,
            begin=0,
            end=5)
        scheduler2 = MultiStepParamScheduler(
            self.optimizer, param_name='lr', gamma=0.1, milestones=[3, 6, 9])
        self._test_scheduler_value([scheduler1, scheduler2], targets, epochs)
