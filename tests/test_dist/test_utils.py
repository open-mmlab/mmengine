# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from unittest import TestCase

import torch
import torch.distributed as torch_dist

import mmengine.dist as dist
from mmengine.testing._internal import MultiProcessTestCase


class TestUtils(TestCase):

    def test_get_backend(self):
        self.assertIsNone(dist.get_backend())

    def test_get_world_size(self):
        self.assertEqual(dist.get_world_size(), 1)

    def test_get_rank(self):
        self.assertEqual(dist.get_rank(), 0)

    def test_local_size(self):
        self.assertEqual(dist.get_local_size(), 1)

    def test_local_rank(self):
        self.assertEqual(dist.get_local_rank(), 0)

    def test_get_dist_info(self):
        self.assertEqual(dist.get_dist_info(), (0, 1))

    def test_is_main_process(self):
        self.assertTrue(dist.is_main_process())

    def test_master_only(self):

        @dist.master_only
        def fun():
            assert dist.get_rank() == 0

        fun()

    def test_barrier(self):
        dist.barrier()  # nothing is done


class TestUtilsWithGLOOBackend(MultiProcessTestCase):

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29505'
        os.environ['RANK'] = str(rank)
        dist.init_dist('pytorch', 'gloo', rank=rank, world_size=world_size)
        dist.init_local_group(0, world_size)

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def test_get_backend(self):
        self._init_dist_env(self.rank, self.world_size)
        self.assertEqual(dist.get_backend(), torch_dist.get_backend())

    def test_get_world_size(self):
        self._init_dist_env(self.rank, self.world_size)
        self.assertEqual(dist.get_world_size(), 2)

    def test_get_rank(self):
        self._init_dist_env(self.rank, self.world_size)
        if torch_dist.get_rank() == 0:
            self.assertEqual(dist.get_rank(), 0)
        else:
            self.assertEqual(dist.get_rank(), 1)

    def test_local_size(self):
        self._init_dist_env(self.rank, self.world_size)
        self.assertEqual(dist.get_local_size(), 2)

    def test_local_rank(self):
        self._init_dist_env(self.rank, self.world_size)
        self.assertEqual(
            torch_dist.get_rank(dist.get_local_group()), dist.get_local_rank())

    def test_get_dist_info(self):
        self._init_dist_env(self.rank, self.world_size)
        if dist.get_rank() == 0:
            self.assertEqual(dist.get_dist_info(), (0, 2))
        else:
            self.assertEqual(dist.get_dist_info(), (1, 2))

    def test_is_main_process(self):
        self._init_dist_env(self.rank, self.world_size)
        if dist.get_rank() == 0:
            self.assertTrue(dist.is_main_process())
        else:
            self.assertFalse(dist.is_main_process())

    def test_master_only(self):
        self._init_dist_env(self.rank, self.world_size)

        @dist.master_only
        def fun():
            assert dist.get_rank() == 0

        fun()


@unittest.skipIf(
    torch.cuda.device_count() < 2, reason='need 2 gpu to test nccl')
class TestUtilsWithNCCLBackend(MultiProcessTestCase):

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29505'
        os.environ['RANK'] = str(rank)
        dist.init_dist('pytorch', 'nccl', rank=rank, world_size=world_size)
        dist.init_local_group(0, world_size)

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def test_get_backend(self):
        self._init_dist_env(self.rank, self.world_size)
        self.assertEqual(dist.get_backend(), torch_dist.get_backend())

    def test_get_world_size(self):
        self._init_dist_env(self.rank, self.world_size)
        self.assertEqual(dist.get_world_size(), 2)

    def test_get_rank(self):
        self._init_dist_env(self.rank, self.world_size)
        if torch_dist.get_rank() == 0:
            self.assertEqual(dist.get_rank(), 0)
        else:
            self.assertEqual(dist.get_rank(), 1)

    def test_local_size(self):
        self._init_dist_env(self.rank, self.world_size)
        self.assertEqual(dist.get_local_size(), 2)

    def test_local_rank(self):
        self._init_dist_env(self.rank, self.world_size)
        self.assertEqual(
            torch_dist.get_rank(dist.get_local_group()), dist.get_local_rank())

    def test_get_dist_info(self):
        self._init_dist_env(self.rank, self.world_size)
        if dist.get_rank() == 0:
            self.assertEqual(dist.get_dist_info(), (0, 2))
        else:
            self.assertEqual(dist.get_dist_info(), (1, 2))

    def test_is_main_process(self):
        self._init_dist_env(self.rank, self.world_size)
        if dist.get_rank() == 0:
            self.assertTrue(dist.is_main_process())
        else:
            self.assertFalse(dist.is_main_process())

    def test_master_only(self):
        self._init_dist_env(self.rank, self.world_size)

        @dist.master_only
        def fun():
            assert dist.get_rank() == 0

        fun()
