# Copyright (c) OpenMMLab. All rights reserved.

from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from mmengine.dataset import DefaultSampler, InfiniteSampler


class TestDefaultSampler(TestCase):

    def setUp(self):
        self.data_length = 100
        self.dataset = list(range(self.data_length))

    @patch('mmengine.dataset.sampler.get_dist_info', return_value=(0, 1))
    def test_non_dist(self, mock):
        sampler = DefaultSampler(self.dataset)
        self.assertEqual(sampler.world_size, 1)
        self.assertEqual(sampler.rank, 0)

        # test round_up=True
        sampler = DefaultSampler(self.dataset, round_up=True, shuffle=False)
        self.assertEqual(sampler.total_size, self.data_length)
        self.assertEqual(sampler.num_samples, self.data_length)
        self.assertEqual(list(sampler), list(range(self.data_length)))

        # test round_up=False
        sampler = DefaultSampler(self.dataset, round_up=False, shuffle=False)
        self.assertEqual(sampler.total_size, self.data_length)
        self.assertEqual(sampler.num_samples, self.data_length)
        self.assertEqual(list(sampler), list(range(self.data_length)))

    @patch('mmengine.dataset.sampler.get_dist_info', return_value=(2, 3))
    def test_dist(self, mock):
        sampler = DefaultSampler(self.dataset)
        self.assertEqual(sampler.world_size, 3)
        self.assertEqual(sampler.rank, 2)

        # test round_up=True
        sampler = DefaultSampler(self.dataset, round_up=True, shuffle=False)
        self.assertEqual(sampler.num_samples, np.ceil(self.data_length / 3))
        self.assertEqual(sampler.total_size, sampler.num_samples * 3)
        self.assertEqual(len(sampler), sampler.num_samples)
        self.assertEqual(
            list(sampler),
            list(range(self.data_length))[2::3] + [1])

        # test round_up=False
        sampler = DefaultSampler(self.dataset, round_up=False, shuffle=False)
        self.assertEqual(sampler.num_samples,
                         np.ceil((self.data_length - 2) / 3))
        self.assertEqual(sampler.total_size, self.data_length)
        self.assertEqual(len(sampler), sampler.num_samples)
        self.assertEqual(list(sampler), list(range(self.data_length))[2::3])

    @patch('mmengine.dataset.sampler.get_dist_info', return_value=(0, 1))
    @patch('mmengine.dataset.sampler.sync_random_seed', return_value=7)
    def test_shuffle(self, mock1, mock2):
        # test seed=None
        sampler = DefaultSampler(self.dataset, seed=None)
        self.assertEqual(sampler.seed, 7)

        # test random seed
        sampler = DefaultSampler(self.dataset, shuffle=True, seed=0)
        sampler.set_epoch(10)
        g = torch.Generator()
        g.manual_seed(10)
        self.assertEqual(
            list(sampler),
            torch.randperm(len(self.dataset), generator=g).tolist())

        sampler = DefaultSampler(self.dataset, shuffle=True, seed=42)
        sampler.set_epoch(10)
        g = torch.Generator()
        g.manual_seed(42 + 10)
        self.assertEqual(
            list(sampler),
            torch.randperm(len(self.dataset), generator=g).tolist())


class TestInfiniteSampler(TestCase):

    def setUp(self):
        self.data_length = 100
        self.dataset = list(range(self.data_length))

    @patch('mmengine.dataset.sampler.get_dist_info', return_value=(0, 1))
    def test_non_dist(self, mock):
        sampler = InfiniteSampler(self.dataset)
        self.assertEqual(sampler.world_size, 1)
        self.assertEqual(sampler.rank, 0)

        # test iteration
        sampler = InfiniteSampler(self.dataset, shuffle=False)
        self.assertEqual(len(sampler), self.data_length)
        self.assertEqual(sampler.size, self.data_length)
        sampler_iter = iter(sampler)
        items = [next(sampler_iter) for _ in range(self.data_length * 2)]
        self.assertEqual(items, list(range(self.data_length)) * 2)

    @patch('mmengine.dataset.sampler.get_dist_info', return_value=(2, 3))
    def test_dist(self, mock):
        sampler = InfiniteSampler(self.dataset)
        self.assertEqual(sampler.world_size, 3)
        self.assertEqual(sampler.rank, 2)

        # test iteration
        sampler = InfiniteSampler(self.dataset, shuffle=False)
        self.assertEqual(len(sampler), self.data_length)
        self.assertEqual(sampler.size, self.data_length)
        targets = (list(range(self.data_length)) * 2)[2::3]
        sampler_iter = iter(sampler)
        samples = [next(sampler_iter) for _ in range(len(targets))]
        print(samples)
        self.assertEqual(samples, targets)

    @patch('mmengine.dataset.sampler.get_dist_info', return_value=(0, 1))
    @patch('mmengine.dataset.sampler.sync_random_seed', return_value=7)
    def test_shuffle(self, mock1, mock2):
        # test seed=None
        sampler = InfiniteSampler(self.dataset, seed=None)
        self.assertEqual(sampler.seed, 7)

        # test the random seed
        sampler = InfiniteSampler(self.dataset, shuffle=True, seed=42)

        sampler_iter = iter(sampler)
        samples = [next(sampler_iter) for _ in range(self.data_length)]

        g = torch.Generator()
        g.manual_seed(42)
        self.assertEqual(
            samples,
            torch.randperm(self.data_length, generator=g).tolist())

    def test_set_epoch(self):
        sampler = InfiniteSampler(self.dataset)
        sampler.set_epoch(10)
