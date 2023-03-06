# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from unittest import TestCase

import torch

from mmengine.logging import MMLogger
from mmengine.model import (ExponentialMovingAverage, MomentumAnnealingEMA,
                            StochasticWeightAverage)
from mmengine.testing import assert_allclose


class TestAveragedModel(TestCase):
    """Test the AveragedModel class.

    Some test cases are referenced from https://github.com/pytorch/pytorch/blob/master/test/test_optim.py
    """  # noqa: E501

    def _test_swa_model(self, net_device, avg_device):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.Linear(5, 10)).to(net_device)

        averaged_model = StochasticWeightAverage(model, device=avg_device)
        averaged_params = [
            torch.zeros_like(param) for param in model.parameters()
        ]
        n_updates = 2
        for i in range(n_updates):
            for p, p_avg in zip(model.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                p_avg += p.detach() / n_updates
            averaged_model.update_parameters(model)

        for p_avg, p_swa in zip(averaged_params, averaged_model.parameters()):
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_swa.device == avg_device)
            self.assertTrue(p.device == net_device)
            assert_allclose(p_avg, p_swa.to(p_avg.device))
        self.assertTrue(averaged_model.steps.device == avg_device)

    def test_averaged_model_all_devices(self):
        cpu = torch.device('cpu')
        self._test_swa_model(cpu, cpu)
        if torch.cuda.is_available():
            cuda = torch.device(0)
            self._test_swa_model(cuda, cpu)
            self._test_swa_model(cpu, cuda)
            self._test_swa_model(cuda, cuda)

    def test_swa_mixed_device(self):
        if not torch.cuda.is_available():
            return
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10))
        model[0].cuda()
        model[1].cpu()
        averaged_model = StochasticWeightAverage(model)
        averaged_params = [
            torch.zeros_like(param) for param in model.parameters()
        ]
        n_updates = 10
        for i in range(n_updates):
            for p, p_avg in zip(model.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                p_avg += p.detach() / n_updates
            averaged_model.update_parameters(model)

        for p_avg, p_swa in zip(averaged_params, averaged_model.parameters()):
            assert_allclose(p_avg, p_swa)
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_avg.device == p_swa.device)

    def test_swa_state_dict(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10))
        averaged_model = StochasticWeightAverage(model)
        averaged_model2 = StochasticWeightAverage(model)
        n_updates = 10
        for i in range(n_updates):
            for p in model.parameters():
                p.detach().add_(torch.randn_like(p))
            averaged_model.update_parameters(model)
        averaged_model2.load_state_dict(averaged_model.state_dict())
        for p_swa, p_swa2 in zip(averaged_model.parameters(),
                                 averaged_model2.parameters()):
            assert_allclose(p_swa, p_swa2)
        self.assertTrue(averaged_model.steps == averaged_model2.steps)

    def test_ema(self):
        # test invalid momentum
        with self.assertRaisesRegex(AssertionError,
                                    'momentum must be in range'):
            model = torch.nn.Sequential(
                torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10))
            ExponentialMovingAverage(model, momentum=3)

        # Warning should be raised if the value of momentum in EMA is
        # a large number
        with self.assertLogs(MMLogger.get_current_instance(), level='WARNING'):
            model = torch.nn.Sequential(
                torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10))
            ExponentialMovingAverage(model, momentum=0.9)
        # test EMA
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3), torch.nn.Linear(5, 10))
        momentum = 0.1

        ema_model = ExponentialMovingAverage(model, momentum=momentum)
        averaged_params = [
            torch.zeros_like(param) for param in model.parameters()
        ]
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            for p, p_avg in zip(model.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    updated_averaged_params.append(
                        (p_avg * (1 - momentum) + p * momentum).clone())
            ema_model.update_parameters(model)
            averaged_params = updated_averaged_params

        for p_target, p_ema in zip(averaged_params, ema_model.parameters()):
            assert_allclose(p_target, p_ema)

    def test_ema_update_buffers(self):
        # Test EMA and update_buffers as True.
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5, momentum=0.3), torch.nn.Linear(5, 10))
        momentum = 0.1

        ema_model = ExponentialMovingAverage(
            model, momentum=momentum, update_buffers=True)
        averaged_params = [
            torch.zeros_like(param)
            for param in itertools.chain(model.parameters(), model.buffers())
            if param.size() != torch.Size([])
        ]
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            params = [
                param for param in itertools.chain(model.parameters(),
                                                   model.buffers())
                if param.size() != torch.Size([])
            ]
            for p, p_avg in zip(params, averaged_params):
                p.detach().add_(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    updated_averaged_params.append(
                        (p_avg * (1 - momentum) + p * momentum).clone())
            ema_model.update_parameters(model)
            averaged_params = updated_averaged_params

        ema_params = [
            param for param in itertools.chain(ema_model.module.parameters(),
                                               ema_model.module.buffers())
            if param.size() != torch.Size([])
        ]
        for p_target, p_ema in zip(averaged_params, ema_params):
            assert_allclose(p_target, p_ema)

    def test_momentum_annealing_ema(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5, momentum=0.3), torch.nn.Linear(5, 10))
        # Test invalid gamma
        with self.assertRaisesRegex(AssertionError,
                                    'gamma must be greater than 0'):
            MomentumAnnealingEMA(model, gamma=-1)

        # Test EMA with momentum annealing.
        momentum = 0.1
        gamma = 4

        ema_model = MomentumAnnealingEMA(
            model, gamma=gamma, momentum=momentum, update_buffers=True)
        averaged_params = [
            torch.zeros_like(param)
            for param in itertools.chain(model.parameters(), model.buffers())
            if param.size() != torch.Size([])
        ]
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            params = [
                param for param in itertools.chain(model.parameters(),
                                                   model.buffers())
                if param.size() != torch.Size([])
            ]
            for p, p_avg in zip(params, averaged_params):
                p.add(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    m = max(momentum, gamma / (gamma + i))
                    updated_averaged_params.append(
                        (p_avg * (1 - m) + p * m).clone())
            ema_model.update_parameters(model)
            averaged_params = updated_averaged_params

        ema_params = [
            param for param in itertools.chain(ema_model.module.parameters(),
                                               ema_model.module.buffers())
            if param.size() != torch.Size([])
        ]
        for p_target, p_ema in zip(averaged_params, ema_params):
            assert_allclose(p_target, p_ema)

    def test_momentum_annealing_ema_with_interval(self):
        # Test EMA with momentum annealing and interval
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.BatchNorm2d(5, momentum=0.3), torch.nn.Linear(5, 10))
        momentum = 0.1
        gamma = 4
        interval = 3

        ema_model = MomentumAnnealingEMA(
            model,
            gamma=gamma,
            momentum=momentum,
            interval=interval,
            update_buffers=True)
        averaged_params = [
            torch.zeros_like(param)
            for param in itertools.chain(model.parameters(), model.buffers())
            if param.size() != torch.Size([])
        ]
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            params = [
                param for param in itertools.chain(model.parameters(),
                                                   model.buffers())
                if param.size() != torch.Size([])
            ]
            for p, p_avg in zip(params, averaged_params):
                p.add(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                elif i % interval == 0:
                    m = max(momentum, gamma / (gamma + i))
                    updated_averaged_params.append(
                        (p_avg * (1 - m) + p * m).clone())
                else:
                    updated_averaged_params.append(p_avg.clone())
            ema_model.update_parameters(model)
            averaged_params = updated_averaged_params

        ema_params = [
            param for param in itertools.chain(ema_model.module.parameters(),
                                               ema_model.module.buffers())
            if param.size() != torch.Size([])
        ]
        for p_target, p_ema in zip(averaged_params, ema_params):
            assert_allclose(p_target, p_ema)
