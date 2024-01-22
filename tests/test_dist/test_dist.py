# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
import unittest
from itertools import product
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.distributed as torch_dist

import mmengine.dist as dist
from mmengine.device import is_musa_available
from mmengine.dist.dist import sync_random_seed
from mmengine.testing._internal import MultiProcessTestCase
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION


class TestDist(TestCase):
    """Test dist module in non-distributed environment."""

    def test_all_reduce(self):
        data = torch.arange(2, dtype=torch.int64)
        expected = torch.arange(2, dtype=torch.int64)
        dist.all_reduce(data)
        self.assertTrue(torch.allclose(data, expected))

    def test_all_gather(self):
        data = torch.arange(2, dtype=torch.int64)
        expected = torch.arange(2, dtype=torch.int64)
        output = dist.all_gather(data)
        self.assertTrue(torch.allclose(output[0], expected))

    def test_gather(self):
        data = torch.arange(2, dtype=torch.int64)
        expected = torch.arange(2, dtype=torch.int64)
        output = dist.gather(data)
        self.assertTrue(torch.allclose(output[0], expected))

    def test_broadcast(self):
        data = torch.arange(2, dtype=torch.int64)
        expected = torch.arange(2, dtype=torch.int64)
        dist.broadcast(data)
        self.assertTrue(torch.allclose(data, expected))

    @patch('numpy.random.randint', return_value=10)
    def test_sync_random_seed(self, mock):
        self.assertEqual(sync_random_seed(), 10)

    def test_broadcast_object_list(self):
        with self.assertRaises(AssertionError):
            # input should be list of object
            dist.broadcast_object_list('foo')

        data = ['foo', 12, {1: 2}]
        expected = ['foo', 12, {1: 2}]
        dist.broadcast_object_list(data)
        self.assertEqual(data, expected)

    def test_all_reduce_dict(self):
        with self.assertRaises(AssertionError):
            # input should be dict
            dist.all_reduce_dict('foo')

        data = {
            'key1': torch.arange(2, dtype=torch.int64),
            'key2': torch.arange(3, dtype=torch.int64)
        }
        expected = {
            'key1': torch.arange(2, dtype=torch.int64),
            'key2': torch.arange(3, dtype=torch.int64)
        }
        dist.all_reduce_dict(data)
        for key in data:
            self.assertTrue(torch.allclose(data[key], expected[key]))

    def test_all_gather_object(self):
        data = 'foo'
        expected = 'foo'
        gather_objects = dist.all_gather_object(data)
        self.assertEqual(gather_objects[0], expected)

    def test_gather_object(self):
        data = 'foo'
        expected = 'foo'
        gather_objects = dist.gather_object(data)
        self.assertEqual(gather_objects[0], expected)

    def test_collect_results(self):
        data = ['foo', {1: 2}]
        size = 2
        expected = ['foo', {1: 2}]

        # test `device=cpu`
        output = dist.collect_results(data, size, device='cpu')
        self.assertEqual(output, expected)

        # test `device=gpu`
        output = dist.collect_results(data, size, device='gpu')
        self.assertEqual(output, expected)

    def test_all_reduce_params(self):
        for tensor_type, reduce_op in zip([torch.int64, torch.float32],
                                          ['sum', 'mean']):
            data = [
                torch.tensor([0, 1], dtype=tensor_type) for _ in range(100)
            ]
            data_gen = (item for item in data)
            expected = [
                torch.tensor([0, 1], dtype=tensor_type) for _ in range(100)
            ]

            dist.all_reduce_params(data_gen, op=reduce_op)

            for item1, item2 in zip(data, expected):
                self.assertTrue(torch.allclose(item1, item2))


@unittest.skipIf(is_musa_available(), reason='musa do not support gloo yet')
class TestDistWithGLOOBackend(MultiProcessTestCase):

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29505'
        os.environ['RANK'] = str(rank)
        torch_dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def test_all_reduce(self):
        self._init_dist_env(self.rank, self.world_size)
        tensor_types = [torch.int64, torch.float32, torch.int64]
        reduce_ops = ['sum', 'mean', 'mean']
        for tensor_type, reduce_op in zip(tensor_types, reduce_ops):
            if dist.get_rank() == 0:
                data = torch.tensor([1, 2], dtype=tensor_type)
            else:
                data = torch.tensor([3, 4], dtype=tensor_type)

            if reduce_op == 'sum':
                expected = torch.tensor([4, 6], dtype=tensor_type)
            else:
                expected = torch.tensor([2, 3], dtype=tensor_type)

            dist.all_reduce(data, reduce_op)
            self.assertTrue(torch.allclose(data, expected))

    def test_all_gather(self):
        self._init_dist_env(self.rank, self.world_size)
        if dist.get_rank() == 0:
            data = torch.tensor([0, 1])
        else:
            data = torch.tensor([1, 2])

        expected = [torch.tensor([0, 1]), torch.tensor([1, 2])]

        output = dist.all_gather(data)
        self.assertTrue(
            torch.allclose(output[dist.get_rank()], expected[dist.get_rank()]))

    def test_gather(self):
        self._init_dist_env(self.rank, self.world_size)
        if dist.get_rank() == 0:
            data = torch.tensor([0, 1])
        else:
            data = torch.tensor([1, 2])

        output = dist.gather(data)

        if dist.get_rank() == 0:
            expected = [torch.tensor([0, 1]), torch.tensor([1, 2])]
            for i in range(2):
                assert torch.allclose(output[i], expected[i])
        else:
            assert output == []

    def test_broadcast_dist(self):
        self._init_dist_env(self.rank, self.world_size)
        if dist.get_rank() == 0:
            data = torch.tensor([0, 1])
        else:
            data = torch.tensor([1, 2])

        expected = torch.tensor([0, 1])
        dist.broadcast(data, 0)
        assert torch.allclose(data, expected)

    def test_sync_random_seed(self):
        self._init_dist_env(self.rank, self.world_size)
        with patch.object(
                torch, 'tensor',
                return_value=torch.tensor(1024)) as mock_tensor:
            output = dist.sync_random_seed()
            assert output == 1024
        mock_tensor.assert_called()

    def test_broadcast_object_list(self):
        self._init_dist_env(self.rank, self.world_size)
        if dist.get_rank() == 0:
            data = ['foo', 12, {1: 2}]
        else:
            data = [None, None, None]

        expected = ['foo', 12, {1: 2}]
        dist.broadcast_object_list(data)
        self.assertEqual(data, expected)

    def test_all_reduce_dict(self):
        self._init_dist_env(self.rank, self.world_size)
        for tensor_type, reduce_op in zip([torch.int64, torch.float32],
                                          ['sum', 'mean']):
            if dist.get_rank() == 0:
                data = {
                    'key1': torch.tensor([0, 1], dtype=tensor_type),
                    'key2': torch.tensor([1, 2], dtype=tensor_type),
                }
            else:
                data = {
                    'key1': torch.tensor([2, 3], dtype=tensor_type),
                    'key2': torch.tensor([3, 4], dtype=tensor_type),
                }

            if reduce_op == 'sum':
                expected = {
                    'key1': torch.tensor([2, 4], dtype=tensor_type),
                    'key2': torch.tensor([4, 6], dtype=tensor_type),
                }
            else:
                expected = {
                    'key1': torch.tensor([1, 2], dtype=tensor_type),
                    'key2': torch.tensor([2, 3], dtype=tensor_type),
                }

            dist.all_reduce_dict(data, reduce_op)

            for key in data:
                assert torch.allclose(data[key], expected[key])

        # `torch.cat` in torch1.5 can not concatenate different types so we
        # fallback to convert them all to float type.
        if digit_version(TORCH_VERSION) == digit_version('1.5.0'):
            if dist.get_rank() == 0:
                data = {
                    'key1': torch.tensor([0, 1], dtype=torch.float32),
                    'key2': torch.tensor([1, 2], dtype=torch.int32)
                }
            else:
                data = {
                    'key1': torch.tensor([2, 3], dtype=torch.float32),
                    'key2': torch.tensor([3, 4], dtype=torch.int32),
                }

            expected = {
                'key1': torch.tensor([2, 4], dtype=torch.float32),
                'key2': torch.tensor([4, 6], dtype=torch.float32),
            }

            dist.all_reduce_dict(data, 'sum')

            for key in data:
                assert torch.allclose(data[key], expected[key])

    def test_all_gather_object(self):
        self._init_dist_env(self.rank, self.world_size)

        # data is a pickable python object
        if dist.get_rank() == 0:
            data = 'foo'
        else:
            data = {1: 2}

        expected = ['foo', {1: 2}]
        output = dist.all_gather_object(data)

        self.assertEqual(output, expected)

        # data is a list of pickable python object
        if dist.get_rank() == 0:
            data = ['foo', {1: 2}]
        else:
            data = {2: 3}

        expected = [['foo', {1: 2}], {2: 3}]
        output = dist.all_gather_object(data)

        self.assertEqual(output, expected)

    def test_gather_object(self):
        self._init_dist_env(self.rank, self.world_size)

        # data is a pickable python object
        if dist.get_rank() == 0:
            data = 'foo'
        else:
            data = {1: 2}

        output = dist.gather_object(data, dst=0)

        if dist.get_rank() == 0:
            self.assertEqual(output, ['foo', {1: 2}])
        else:
            self.assertIsNone(output)

        # data is a list of pickable python object
        if dist.get_rank() == 0:
            data = ['foo', {1: 2}]
        else:
            data = {2: 3}

        output = dist.gather_object(data, dst=0)

        if dist.get_rank() == 0:
            self.assertEqual(output, [['foo', {1: 2}], {2: 3}])
        else:
            self.assertIsNone(output)

    def test_all_reduce_params(self):
        self._init_dist_env(self.rank, self.world_size)

        tensor_types = [torch.int64, torch.float32]
        reduce_ops = ['sum', 'mean']
        coalesces = [True, False]
        for tensor_type, reduce_op, coalesce in zip(tensor_types, reduce_ops,
                                                    coalesces):
            if dist.get_rank() == 0:
                data = [
                    torch.tensor([0, 1], dtype=tensor_type) for _ in range(100)
                ]
            else:
                data = (
                    torch.tensor([2, 3], dtype=tensor_type)
                    for _ in range(100))

            data_gen = (item for item in data)

            if reduce_op == 'sum':
                expected = (
                    torch.tensor([2, 4], dtype=tensor_type)
                    for _ in range(100))
            else:
                expected = (
                    torch.tensor([1, 2], dtype=tensor_type)
                    for _ in range(100))

            dist.all_reduce_params(data_gen, coalesce=coalesce, op=reduce_op)

            for item1, item2 in zip(data, expected):
                self.assertTrue(torch.allclose(item1, item2))


@unittest.skipIf(
    torch.cuda.device_count() < 2, reason='need 2 gpu to test nccl')
class TestDistWithNCCLBackend(MultiProcessTestCase):

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29505'
        os.environ['RANK'] = str(rank)

        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        torch_dist.init_process_group(
            backend='nccl', rank=rank, world_size=world_size)

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def test_all_reduce(self):
        self._init_dist_env(self.rank, self.world_size)
        tensor_types = [torch.int64, torch.float32]
        reduce_ops = ['sum', 'mean']
        device_types = ['cpu', 'cuda']
        for tensor_type, reduce_op, device_type in product(
                tensor_types, reduce_ops, device_types):
            # 'mean' op does not support torch.int64
            if tensor_type == torch.int64 and reduce_op == 'mean':
                continue

            if dist.get_rank() == 0:
                data = torch.tensor([1, 2], dtype=tensor_type).to(device_type)
            else:
                data = torch.tensor([3, 4], dtype=tensor_type).to(device_type)

            if reduce_op == 'sum':
                expected = torch.tensor([4, 6],
                                        dtype=tensor_type).to(device_type)
            else:
                expected = torch.tensor([2, 3],
                                        dtype=tensor_type).to(device_type)

            dist.all_reduce(data, reduce_op)
            self.assertTrue(torch.allclose(data, expected))

    def test_all_gather(self):
        self._init_dist_env(self.rank, self.world_size)
        for device_type in ('cpu', 'cuda'):
            if dist.get_rank() == 0:
                data = torch.tensor([0, 1]).to(device_type)
            else:
                data = torch.tensor([1, 2]).to(device_type)

            expected = [
                torch.tensor([0, 1]).to(device_type),
                torch.tensor([1, 2]).to(device_type)
            ]

            output = dist.all_gather(data)
            self.assertTrue(
                torch.allclose(output[dist.get_rank()],
                               expected[dist.get_rank()]))

    def test_broadcast_dist(self):
        self._init_dist_env(self.rank, self.world_size)
        for device_type in ('cpu', 'cuda'):
            if dist.get_rank() == 0:
                data = torch.tensor([0, 1]).to(device_type)
            else:
                data = torch.tensor([1, 2]).to(device_type)

            expected = torch.tensor([0, 1]).to(device_type)
            dist.broadcast(data, 0)
            assert torch.allclose(data, expected)

    def test_sync_random_seed(self):
        self._init_dist_env(self.rank, self.world_size)
        with patch.object(
                torch, 'tensor',
                return_value=torch.tensor(1024)) as mock_tensor:
            output = dist.sync_random_seed()
            assert output == 1024
        mock_tensor.assert_called()

    def test_broadcast_object_list(self):
        self._init_dist_env(self.rank, self.world_size)
        if dist.get_rank() == 0:
            data = ['foo', 12, {1: 2}]
        else:
            data = [None, None, None]

        expected = ['foo', 12, {1: 2}]
        dist.broadcast_object_list(data)
        self.assertEqual(data, expected)

    def test_all_reduce_dict(self):
        self._init_dist_env(self.rank, self.world_size)
        tensor_types = [torch.int64, torch.float32]
        reduce_ops = ['sum', 'mean']
        device_types = ['cpu', 'cuda']
        for tensor_type, reduce_op, device_type in product(
                tensor_types, reduce_ops, device_types):
            # 'mean' op does not support torch.int64
            if tensor_type == torch.int64 and reduce_op == 'mean':
                continue

            if dist.get_rank() == 0:
                data = {
                    'key1':
                    torch.tensor([0, 1], dtype=tensor_type).to(device_type),
                    'key2':
                    torch.tensor([1, 2], dtype=tensor_type).to(device_type),
                }
            else:
                data = {
                    'key1':
                    torch.tensor([2, 3], dtype=tensor_type).to(device_type),
                    'key2':
                    torch.tensor([3, 4], dtype=tensor_type).to(device_type),
                }

            if reduce_op == 'sum':
                expected = {
                    'key1':
                    torch.tensor([2, 4], dtype=tensor_type).to(device_type),
                    'key2':
                    torch.tensor([4, 6], dtype=tensor_type).to(device_type),
                }
            else:
                expected = {
                    'key1':
                    torch.tensor([1, 2], dtype=tensor_type).to(device_type),
                    'key2':
                    torch.tensor([2, 3], dtype=tensor_type).to(device_type),
                }

            dist.all_reduce_dict(data, reduce_op)

            for key in data:
                assert torch.allclose(data[key], expected[key])

        # `torch.cat` in torch1.5 can not concatenate different types so we
        # fallback to convert them all to float type.
        for device_type in ('cpu', 'cuda'):
            if digit_version(TORCH_VERSION) == digit_version('1.5.0'):
                if dist.get_rank() == 0:
                    data = {
                        'key1':
                        torch.tensor([0, 1],
                                     dtype=torch.float32).to(device_type),
                        'key2':
                        torch.tensor([1, 2],
                                     dtype=torch.int32).to(device_type),
                    }
                else:
                    data = {
                        'key1':
                        torch.tensor([2, 3],
                                     dtype=torch.float32).to(device_type),
                        'key2':
                        torch.tensor([3, 4],
                                     dtype=torch.int32).to(device_type),
                    }

                expected = {
                    'key1':
                    torch.tensor([2, 4], dtype=torch.float32).to(device_type),
                    'key2':
                    torch.tensor([4, 6], dtype=torch.float32).to(device_type),
                }

                dist.all_reduce_dict(data, 'sum')

                for key in data:
                    assert torch.allclose(data[key], expected[key])

    def test_all_gather_object(self):
        self._init_dist_env(self.rank, self.world_size)

        # data is a pickable python object
        if dist.get_rank() == 0:
            data = 'foo'
        else:
            data = {1: 2}

        expected = ['foo', {1: 2}]
        output = dist.all_gather_object(data)

        self.assertEqual(output, expected)

        # data is a list of pickable python object
        if dist.get_rank() == 0:
            data = ['foo', {1: 2}]
        else:
            data = {2: 3}

        expected = [['foo', {1: 2}], {2: 3}]
        output = dist.all_gather_object(data)

        self.assertEqual(output, expected)

    def test_collect_results(self):
        self._init_dist_env(self.rank, self.world_size)

        # 1. test `device` and `tmpdir` parameters
        if dist.get_rank() == 0:
            data = ['foo', {1: 2}]
        else:
            data = [24, {'a': 'b'}]

        size = 4

        expected = ['foo', 24, {1: 2}, {'a': 'b'}]

        # 1.1 test `device=cpu` and `tmpdir` is None
        output = dist.collect_results(data, size, device='cpu')
        if dist.get_rank() == 0:
            self.assertEqual(output, expected)
        else:
            self.assertIsNone(output)

        # 1.2 test `device=cpu` and `tmpdir` is not None
        tmpdir = tempfile.mkdtemp()
        # broadcast tmpdir to all ranks to make it consistent
        object_list = [tmpdir]
        dist.broadcast_object_list(object_list)
        output = dist.collect_results(
            data, size, device='cpu', tmpdir=object_list[0])
        if dist.get_rank() == 0:
            self.assertEqual(output, expected)
        else:
            self.assertIsNone(output)

        if dist.get_rank() == 0:
            # object_list[0] will be removed by `dist.collect_results`
            self.assertFalse(osp.exists(object_list[0]))

        # 1.3 test `device=gpu`
        output = dist.collect_results(data, size, device='gpu')
        if dist.get_rank() == 0:
            self.assertEqual(output, expected)
        else:
            self.assertIsNone(output)

        # 2. test `size` parameter
        if dist.get_rank() == 0:
            data = ['foo', {1: 2}]
        else:
            data = [24, {'a': 'b'}]

        size = 3

        expected = ['foo', 24, {1: 2}]

        # 2.1 test `device=cpu` and `tmpdir` is None
        output = dist.collect_results(data, size, device='cpu')
        if dist.get_rank() == 0:
            self.assertEqual(output, expected)
        else:
            self.assertIsNone(output)

        # 2.2 test `device=gpu`
        output = dist.collect_results(data, size, device='gpu')
        if dist.get_rank() == 0:
            self.assertEqual(output, expected)
        else:
            self.assertIsNone(output)

    def test_all_reduce_params(self):
        self._init_dist_env(self.rank, self.world_size)

        tensor_types = [torch.int64, torch.float32]
        reduce_ops = ['sum', 'mean']
        coalesces = [True, False]
        device_types = ['cpu', 'cuda']
        for tensor_type, reduce_op, coalesce, device_type in zip(
                tensor_types, reduce_ops, coalesces, device_types):
            if dist.get_rank() == 0:
                data = [
                    torch.tensor([0, 1], dtype=tensor_type).to(device_type)
                    for _ in range(100)
                ]
            else:
                data = [
                    torch.tensor([2, 3], dtype=tensor_type).to(device_type)
                    for _ in range(100)
                ]

            data_gen = (item for item in data)
            dist.all_reduce_params(data_gen, coalesce=coalesce, op=reduce_op)

            if reduce_op == 'sum':
                expected = (
                    torch.tensor([2, 4], dtype=tensor_type).to(device_type)
                    for _ in range(100))
            else:
                expected = (
                    torch.tensor([1, 2], dtype=tensor_type).to(device_type)
                    for _ in range(100))

            for item1, item2 in zip(data_gen, expected):
                self.assertTrue(torch.allclose(item1, item2))
