# Copyright (c) OpenMMLab. All rights reserved.
import math
from unittest import TestCase

import torch
import torch.nn.functional as F
import torch.optim as optim

from mmengine.optim.scheduler import (ConstantMomentum,
                                      CosineAnnealingMomentum,
                                      CosineRestartMomentum,
                                      ExponentialMomentum, LinearMomentum,
                                      MultiStepMomentum, PolyMomentum,
                                      StepMomentum, _ParamScheduler)
from mmengine.testing import assert_allclose


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestMomentumScheduler(TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.model = ToyModel()
        momentum = 0.05
        self.layer2_mult = 10
        self.optimizer = optim.SGD([{
            'params': self.model.conv1.parameters()
        }, {
            'params': self.model.conv2.parameters(),
            'momentum': momentum * self.layer2_mult
        }],
                                   lr=0.01,
                                   momentum=momentum,
                                   weight_decay=5e-4)
        self.optimizer_with_betas = optim.Adam(
            [{
                'params': self.model.conv1.parameters()
            }, {
                'params': self.model.conv2.parameters(),
                'betas': (momentum * self.layer2_mult, 0.999)
            }],
            lr=0.01,
            betas=(momentum, 0.999),
            weight_decay=5e-4)

    def test_invalid_optimizer(self):
        with self.assertRaisesRegex(
                ValueError,
                'optimizer must support momentum when using momentum scheduler'
        ):
            optimizer = optim.ASGD(
                self.model.parameters(),
                lr=0.01,
            )
            StepMomentum(optimizer, step_size=1)

    def test_overwrite_optimzer_step(self):
        # raise warning if the counter in optimizer.step() is overwritten
        scheduler = ExponentialMomentum(self.optimizer, gamma=0.9)

        def overwrite_fun():
            pass

        self.optimizer.step = overwrite_fun
        self.optimizer.step()
        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                              scheduler.step)

    def test_resume(self):
        # test invalid case: optimizer and scheduler are not both resumed
        with self.assertRaisesRegex(
                KeyError, "param 'initial_momentum' is not specified"):
            StepMomentum(self.optimizer, gamma=0.1, step_size=3, last_step=10)

        # test manually resume with ``last_step`` instead of load_state_dict
        epochs = 10
        targets = [0.05 * (0.9**x) for x in range(epochs)]
        scheduler = ExponentialMomentum(self.optimizer, gamma=0.9)

        results = []
        for epoch in range(5):
            results.append(self.optimizer.param_groups[0]['momentum'])
            # The order should be
            # train_epoch() -> save_checkpoint() -> scheduler.step().
            # Break at here to simulate the checkpoint is saved before
            # the scheduler.step().
            if epoch == 4:
                break
            scheduler.step()
        scheduler2 = ExponentialMomentum(
            self.optimizer, gamma=0.9, last_step=4)
        for epoch in range(6):
            results.append(self.optimizer.param_groups[0]['momentum'])
            scheduler2.step()

        for epoch in range(epochs):
            assert_allclose(
                targets[epoch],
                results[epoch],
                msg='momentum is wrong in epoch {}: expected {}, got {}'.
                format(epoch, targets[epoch], results[epoch]),
                atol=1e-5,
                rtol=0)

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
                self.optimizer, gamma=0.1, step_size=3, last_step=10)
            scheduler.step()
            self.optimizer.step()

        # check warning doc link
        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate',
                              call_sch_before_optim_resume)

    def test_get_last_value(self):
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005]
        targets = [
            single_targets, [t * self.layer2_mult for t in single_targets]
        ]
        scheduler = StepMomentum(self.optimizer, 3, gamma=0.1)
        for epoch in range(epochs):
            result = scheduler.get_last_value()
            self.optimizer.step()
            scheduler.step()
            target = [t[epoch] for t in targets]
            for t, r in zip(target, result):
                assert_allclose(
                    target,
                    result,
                    msg='momentum is wrong in epoch {}: expected {}, got {}'.
                    format(epoch, t, r),
                    atol=1e-5,
                    rtol=0)

    def test_scheduler_step_count(self):
        iteration = 10
        scheduler = StepMomentum(self.optimizer, gamma=0.1, step_size=3)
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
            StepMomentum(
                self.optimizer, gamma=0.1, step_size=3, begin=10, end=5)

        # momentum = 0.05     if epoch == 0
        # momentum = 0.025     if epoch == 1
        # momentum = 0.03125   if epoch == 2
        # momentum = 0.0375    if epoch == 3
        # momentum = 0.04375   if epoch == 4
        # momentum = 0.005     if epoch > 4
        begin = 1
        epochs = 10
        start_factor = 1.0 / 2
        iters = 4
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        single_targets = [0.05] * begin + [x * 0.05
                                           for x in interpolation] + [0.05] * (
                                               epochs - iters - begin)
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler = LinearMomentum(
            self.optimizer,
            start_factor=start_factor,
            begin=begin,
            end=begin + iters + 1)
        self._test_scheduler_value(self.optimizer, scheduler, targets, epochs)

    def _test_scheduler_value(self,
                              optimizer,
                              schedulers,
                              targets,
                              epochs=10,
                              param_name='momentum'):
        if isinstance(schedulers, _ParamScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(optimizer.param_groups, targets):
                assert_allclose(
                    target[epoch],
                    param_group[param_name],
                    msg='{} is wrong in epoch {}: expected {}, got {}'.format(
                        param_name, epoch, target[epoch],
                        param_group[param_name]),
                    atol=1e-5,
                    rtol=0)
                if 'betas' in optimizer.defaults:
                    assert_allclose(
                        target[epoch],
                        param_group['betas'][0],
                        msg='{} is wrong in epoch {}: expected {}, got {}'.
                        format('betas_0', epoch, target[epoch],
                               param_group['betas'][0]),
                        atol=1e-5,
                        rtol=0)
            [scheduler.step() for scheduler in schedulers]

    def test_step_scheduler(self):
        # momentum = 0.05     if epoch < 3
        # momentum = 0.005    if 3 <= epoch < 6
        # momentum = 0.0005   if 6 <= epoch < 9
        # momentum = 0.00005  if epoch >=9
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005
                                                                    ] * 3
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler = StepMomentum(
            self.optimizer, gamma=0.1, step_size=3, verbose=True)
        self._test_scheduler_value(self.optimizer, scheduler, targets, epochs)

        scheduler = StepMomentum(
            self.optimizer_with_betas, gamma=0.1, step_size=3, verbose=True)
        self._test_scheduler_value(self.optimizer_with_betas, scheduler,
                                   targets, epochs)

    def test_multi_step_scheduler(self):
        # momentum = 0.05     if epoch < 2
        # momentum = 0.005    if 2 <= epoch < 5
        # momentum = 0.0005   if 5 <= epoch < 9
        # momentum = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005
                                                                    ] * 3
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler = MultiStepMomentum(
            self.optimizer, gamma=0.1, milestones=[2, 5, 9])
        self._test_scheduler_value(self.optimizer, scheduler, targets, epochs)

        scheduler = MultiStepMomentum(
            self.optimizer_with_betas, gamma=0.1, milestones=[2, 5, 9])
        self._test_scheduler_value(self.optimizer_with_betas, scheduler,
                                   targets, epochs)

    def test_constant_scheduler(self):
        # factor should between 0~1
        with self.assertRaises(ValueError):
            ConstantMomentum(self.optimizer, factor=99)

        # momentum = 0.025     if epoch < 5
        # momentum = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 4 + [0.05] * 6
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler = ConstantMomentum(self.optimizer, factor=1.0 / 2, end=5)
        self._test_scheduler_value(self.optimizer, scheduler, targets, epochs)

        scheduler = ConstantMomentum(
            self.optimizer_with_betas, factor=1.0 / 2, end=5)
        self._test_scheduler_value(self.optimizer_with_betas, scheduler,
                                   targets, epochs)

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
        # momentum = 0.005     if epoch >= 4
        epochs = 10
        start_factor = 1.0 / 2
        iters = 4
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (
            epochs - iters)
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler = LinearMomentum(
            self.optimizer, start_factor=start_factor, end=iters + 1)
        self._test_scheduler_value(self.optimizer, scheduler, targets, epochs)

        scheduler = LinearMomentum(
            self.optimizer_with_betas,
            start_factor=start_factor,
            end=iters + 1)
        self._test_scheduler_value(self.optimizer_with_betas, scheduler,
                                   targets, epochs)

    def test_exp_scheduler(self):
        epochs = 10
        single_targets = [0.05 * (0.9**x) for x in range(epochs)]
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler = ExponentialMomentum(self.optimizer, gamma=0.9)
        self._test_scheduler_value(self.optimizer, scheduler, targets, epochs)

        scheduler = ExponentialMomentum(self.optimizer_with_betas, gamma=0.9)
        self._test_scheduler_value(self.optimizer_with_betas, scheduler,
                                   targets, epochs)

    def test_cos_anneal_scheduler(self):
        epochs = 12
        t = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / t)) / 2
            for x in range(epochs)
        ]
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler = CosineAnnealingMomentum(
            self.optimizer, T_max=t, eta_min=eta_min)
        self._test_scheduler_value(self.optimizer, scheduler, targets, epochs)

        scheduler = CosineAnnealingMomentum(
            self.optimizer_with_betas, T_max=t, eta_min=eta_min)
        self._test_scheduler_value(self.optimizer_with_betas, scheduler,
                                   targets, epochs)

        # Test default `T_max`
        scheduler = CosineAnnealingMomentum(
            self.optimizer, begin=5, end=100, eta_min=eta_min)
        self.assertEqual(scheduler.T_max, 100 - 5)

    def test_poly_scheduler(self):
        epochs = 10
        power = 0.9
        min_lr = 0.001
        iters = 4
        layer1_targets = [
            min_lr + (0.05 - min_lr) * (1 - i / iters)**power
            for i in range(iters)
        ] + [min_lr] * (
            epochs - iters)
        layer2_targets = [
            min_lr + (0.05 * self.layer2_mult - min_lr) *
            (1 - i / iters)**power for i in range(iters)
        ] + [min_lr] * (
            epochs - iters)
        targets = [layer1_targets, layer2_targets]
        scheduler = PolyMomentum(
            self.optimizer, power=power, eta_min=min_lr, end=iters + 1)
        self._test_scheduler_value(
            self.optimizer, scheduler, targets, epochs=10)

        scheduler = PolyMomentum(
            self.optimizer_with_betas,
            power=power,
            eta_min=min_lr,
            end=iters + 1)
        self._test_scheduler_value(
            self.optimizer_with_betas, scheduler, targets, epochs=10)

    def test_cosine_restart_scheduler(self):
        with self.assertRaises(AssertionError):
            CosineRestartMomentum(
                self.optimizer,
                periods=[4, 5],
                restart_weights=[1, 0.5],
                eta_min=0,
                eta_min_ratio=0.1)
        with self.assertRaises(AssertionError):
            CosineRestartMomentum(
                self.optimizer,
                periods=[4, 5],
                restart_weights=[1, 0.5, 0.0],
                eta_min=0)
        single_targets = [
            0.05, 0.0426776, 0.025, 0.00732233, 0.025, 0.022612712, 0.01636271,
            0.0086372, 0.0023872, 0.0023872
        ]
        targets = [
            single_targets, [t * self.layer2_mult for t in single_targets]
        ]
        scheduler = CosineRestartMomentum(
            self.optimizer,
            periods=[4, 5],
            restart_weights=[1, 0.5],
            eta_min=0)
        self._test_scheduler_value(
            self.optimizer, scheduler, targets, epochs=10)

        scheduler = CosineRestartMomentum(
            self.optimizer_with_betas,
            periods=[4, 5],
            restart_weights=[1, 0.5],
            eta_min=0)
        self._test_scheduler_value(
            self.optimizer_with_betas, scheduler, targets, epochs=10)

    def _check_scheduler_state_dict(self, construct, construct2, epochs=10):
        scheduler = construct()
        for _ in range(epochs):
            scheduler.optimizer.step()
            scheduler.step()
        scheduler_copy = construct2()
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

    def test_exp_scheduler_state_dict(self):
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

    def test_linear_scheduler_state_dict(self):
        epochs = 10
        self._check_scheduler_state_dict(
            lambda: LinearMomentum(self.optimizer, start_factor=1 / 3),
            lambda: LinearMomentum(
                self.optimizer, start_factor=0, end_factor=0.3),
            epochs=epochs)

    def test_poly_scheduler_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: PolyMomentum(self.optimizer, power=0.5, eta_min=0.001),
            lambda: PolyMomentum(self.optimizer, power=0.8, eta_min=0.002),
            epochs=10)

    def test_cosine_restart_scheduler_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: CosineRestartMomentum(
                self.optimizer,
                periods=[4, 5],
                restart_weights=[1, 0.5],
                eta_min=0),
            lambda: CosineRestartMomentum(
                self.optimizer,
                periods=[4, 6],
                restart_weights=[1, 0.5],
                eta_min=0),
            epochs=10)

    def test_multi_scheduler_without_overlap_linear_multi_step(self):
        # use Linear in the first 5 epochs and then use MultiStep
        epochs = 12
        single_targets = [0.025, 0.03125, 0.0375, 0.04375
                          ] + [0.05] * 4 + [0.005] * 3 + [0.0005] * 1
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler1 = LinearMomentum(
            self.optimizer, start_factor=1 / 2, begin=0, end=5)
        scheduler2 = MultiStepMomentum(
            self.optimizer, gamma=0.1, milestones=[3, 6], begin=5, end=12)
        self._test_scheduler_value(self.optimizer, [scheduler1, scheduler2],
                                   targets, epochs)

    def test_multi_scheduler_without_overlap_exp_cosine(self):
        # use Exp in the first 5 epochs and then use Cosine
        epochs = 10
        single_targets1 = [0.05 * (0.9**x) for x in range(5)]
        scheduler1 = ExponentialMomentum(
            self.optimizer, gamma=0.9, begin=0, end=5)

        eta_min = 1e-10
        single_targets2 = [
            eta_min + (single_targets1[-1] - eta_min) *
            (1 + math.cos(math.pi * x / 5)) / 2 for x in range(5)
        ]
        single_targets = single_targets1 + single_targets2
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler2 = CosineAnnealingMomentum(
            self.optimizer, T_max=5, eta_min=eta_min, begin=5, end=10)

        self._test_scheduler_value(self.optimizer, [scheduler1, scheduler2],
                                   targets, epochs)

    def test_multi_scheduler_with_overlap(self):
        # use Linear at first 5 epochs together with MultiStep
        epochs = 10
        single_targets = [0.025, 0.03125, 0.0375, 0.004375
                          ] + [0.005] * 2 + [0.0005] * 3 + [0.00005] * 1
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler1 = LinearMomentum(
            self.optimizer, start_factor=1 / 2, begin=0, end=5)
        scheduler2 = MultiStepMomentum(
            self.optimizer, gamma=0.1, milestones=[3, 6, 9])
        self._test_scheduler_value(self.optimizer, [scheduler1, scheduler2],
                                   targets, epochs)

    def test_multi_scheduler_with_gap(self):
        # use Exp in the first 5 epochs and the last 5 epochs use Cosine
        # no scheduler in the middle 5 epochs
        epochs = 15
        single_targets1 = [0.05 * (0.9**x) for x in range(5)]
        scheduler1 = ExponentialMomentum(
            self.optimizer, gamma=0.9, begin=0, end=5)

        eta_min = 1e-10
        single_targets2 = [
            eta_min + (single_targets1[-1] - eta_min) *
            (1 + math.cos(math.pi * x / 5)) / 2 for x in range(5)
        ]
        single_targets = single_targets1 + [single_targets1[-1]
                                            ] * 5 + single_targets2
        targets = [
            single_targets, [x * self.layer2_mult for x in single_targets]
        ]
        scheduler2 = CosineAnnealingMomentum(
            self.optimizer, T_max=5, eta_min=eta_min, begin=10, end=15)

        self._test_scheduler_value(self.optimizer, [scheduler1, scheduler2],
                                   targets, epochs)
