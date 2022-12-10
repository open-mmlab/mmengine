# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from unittest import TestCase

import numpy as np
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

    def test_get_data_device(self):
        # data is a Tensor
        data = torch.tensor([0, 1])
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a list of Tensor
        data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a list but not all items are Tensor
        data = [torch.tensor([0, 1]), 123]
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a list containing Tensor and a dict
        data = [torch.tensor([0, 1]), {'key': torch.tensor([2, 3])}]
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a list containing Tensor and a dict but the dict contains
        # invalid type
        data = [torch.tensor([0, 1]), {'key': '123'}]
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a empty list
        with self.assertRaises(ValueError):
            dist.get_data_device([])

        # data is a dict
        data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([0, 1])}
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a dict but not all values are Tensor
        data = {'key1': torch.tensor([0, 1]), 'key2': 123}
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a dict and one of values is list of Tensor
        data = {'key1': torch.tensor([0, 1]), 'key2': [torch.tensor([0, 1])]}
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a dict and one of values is an invalid type
        data = {'key1': torch.tensor([0, 1]), 'key2': ['123']}
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a empty dict
        with self.assertRaises(ValueError):
            dist.get_data_device({})

        # data is not a valid type
        with self.assertRaisesRegex(
                TypeError,
                'data should be a Tensor, sequence of tensor or dict'):
            dist.get_data_device('123')

    @unittest.skipIf(
        torch.cuda.device_count() == 0, reason='at lest need 1 gpu to test')
    def test_cast_data_device(self):
        expected_device = torch.device('cuda', torch.cuda.current_device())
        # data is a Tensor
        data = torch.tensor([0, 1])
        output = dist.cast_data_device(data, expected_device)
        self.assertEqual(output.device, expected_device)

        # data is a Tensor and out is also a Tensor
        data = torch.tensor([0, 1])
        out = torch.tensor([1, 2])
        output = dist.cast_data_device(data, expected_device, out=out)
        self.assertEqual(output.device, expected_device)
        self.assertTrue(torch.allclose(output.cpu(), out))

        # data is a list of Tensor
        data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        for item in dist.cast_data_device(data, expected_device):
            self.assertEqual(item.device, expected_device)

        # both data and out are list of tensor
        data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        out = [torch.tensor([3, 4]), torch.tensor([5, 6])]
        output = dist.cast_data_device(data, expected_device, out=out)
        for item1, item2 in zip(output, out):
            self.assertEqual(item1.device, expected_device)
            self.assertTrue(torch.allclose(item1.cpu(), item2))

        # data is a list containing a Tensor and a dict
        data = [torch.tensor([0, 1]), {'key': torch.tensor([2, 3])}]
        output = dist.cast_data_device(data, expected_device)
        self.assertEqual(output[0].device, expected_device)
        self.assertEqual(output[1]['key'].device, expected_device)

        # data is a list containing a Tensor and a dict, so does out
        data = [torch.tensor([0, 1]), {'key': torch.tensor([2, 3])}]
        out = [torch.tensor([3, 4]), {'key': torch.tensor([5, 6])}]
        output = dist.cast_data_device(data, expected_device, out=out)
        self.assertEqual(output[0].device, expected_device)
        self.assertTrue(torch.allclose(output[0].cpu(), out[0]))
        self.assertEqual(output[1]['key'].device, expected_device)
        self.assertTrue(torch.allclose(output[1]['key'].cpu(), out[1]['key']))

        # data is an empty list
        with self.assertRaisesRegex(ValueError, 'data should not be empty'):
            dist.cast_data_device([], expected_device)

        # data is a dict
        data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([2, 3])}
        output = dist.cast_data_device(data, expected_device)
        for k, v in output.items():
            self.assertEqual(v.device, expected_device)

        # data is a dict, so does out
        data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([2, 3])}
        out = {'key1': torch.tensor([3, 4]), 'key2': torch.tensor([5, 6])}
        output = dist.cast_data_device(data, expected_device, out=out)
        for k, v in output.items():
            self.assertEqual(v.device, expected_device)
            self.assertTrue(torch.allclose(v.cpu(), out[k]))

        # the length of data and out should be same
        data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([2, 3])}
        out = {'key1': torch.tensor([3, 4])}
        with self.assertRaisesRegex(ValueError,
                                    'length of data and out should be same'):
            dist.cast_data_device(data, expected_device, out=out)

        # data is an empty dict
        with self.assertRaisesRegex(ValueError, 'data should not be empty'):
            dist.cast_data_device({}, expected_device)

        # data is a dict and one of values is list
        data = {'key1': torch.tensor([0, 1]), 'key2': [torch.tensor([2, 3])]}
        out = {'key1': torch.tensor([3, 4]), 'key2': [torch.tensor([5, 6])]}
        output = dist.cast_data_device(data, expected_device, out=out)
        self.assertEqual(output['key1'].device, expected_device)
        self.assertTrue(torch.allclose(output['key1'].cpu(), out['key1']))
        self.assertEqual(output['key2'][0].device, expected_device)
        self.assertTrue(
            torch.allclose(output['key2'][0].cpu(), out['key2'][0]))

        # data is not a valid type
        with self.assertRaisesRegex(
                TypeError, 'data should be a Tensor, list of tensor or dict'):
            dist.cast_data_device(123, expected_device)

        with self.assertRaisesRegex(
                TypeError, 'data should be a Tensor, list of tensor or dict'):
            dist.cast_data_device('123', expected_device)

        with self.assertRaisesRegex(
                TypeError, 'data should be a Tensor, list of tensor or dict'):
            dist.cast_data_device(np.array([0, 1]), expected_device)

        # data and out are not the same type
        data = torch.tensor([0, 1])
        out = '123'
        with self.assertRaisesRegex(TypeError,
                                    'out should be the same type with data'):
            dist.cast_data_device(data, expected_device, out=out)

        data = {0, 1}
        out = {2, 3}
        with self.assertRaisesRegex(TypeError, 'out should not be a set'):
            dist.cast_data_device(data, expected_device, out=out)


class TestUtilsWithGLOOBackend(MultiProcessTestCase):

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29505'
        os.environ['RANK'] = str(rank)

        torch_dist.init_process_group(
            backend='gloo', rank=rank, world_size=world_size)
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

    def test_get_data_device(self):
        self._init_dist_env(self.rank, self.world_size)

        # data is a Tensor
        data = torch.tensor([0, 1])
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a list of Tensor
        data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a list but not all items are Tensor
        data = [torch.tensor([0, 1]), 123]
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a list containing Tensor and a dict
        data = [torch.tensor([0, 1]), {'key': torch.tensor([2, 3])}]
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a list containing Tensor and a dict but the dict contains
        # invalid type
        data = [torch.tensor([0, 1]), {'key': '123'}]
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a empty list
        with self.assertRaises(ValueError):
            dist.get_data_device([])

        # data is a dict
        data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([0, 1])}
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a dict but not all values are Tensor
        data = {'key1': torch.tensor([0, 1]), 'key2': 123}
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a dict and one of values is list of Tensor
        data = {'key1': torch.tensor([0, 1]), 'key2': [torch.tensor([0, 1])]}
        self.assertEqual(dist.get_data_device(data), torch.device('cpu'))

        # data is a dict and one of values is an invalid type
        data = {'key1': torch.tensor([0, 1]), 'key2': ['123']}
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a empty dict
        with self.assertRaises(ValueError):
            dist.get_data_device({})

        # data is not a valid type
        with self.assertRaisesRegex(
                TypeError,
                'data should be a Tensor, sequence of tensor or dict'):
            dist.get_data_device('123')

    def test_get_comm_device(self):
        self._init_dist_env(self.rank, self.world_size)
        group = dist.get_default_group()
        assert dist.get_comm_device(group) == torch.device('cpu')


@unittest.skipIf(
    torch.cuda.device_count() < 2, reason='need 2 gpu to test nccl')
class TestUtilsWithNCCLBackend(MultiProcessTestCase):

    def _init_dist_env(self, rank, world_size):
        """Initialize the distributed environment."""
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29505'
        os.environ['RANK'] = str(rank)

        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        torch_dist.init_process_group(
            backend='nccl', rank=rank, world_size=world_size)
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

    def test_get_data_device(self):
        self._init_dist_env(self.rank, self.world_size)

        expected_device = torch.device('cuda', torch.cuda.current_device())

        # data is a Tensor
        data = torch.tensor([0, 1]).to(expected_device)
        self.assertEqual(dist.get_data_device(data), expected_device)

        # data is a list of Tensor
        data = [
            torch.tensor([0, 1]).to(expected_device),
            torch.tensor([2, 3]).to(expected_device)
        ]
        self.assertEqual(dist.get_data_device(data), expected_device)

        # data is a list but not all items are Tensor
        data = [torch.tensor([0, 1]).to(expected_device), 123]
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a list of Tensor but not all items have the same device type
        data = [torch.tensor([0, 1]), torch.tensor([2, 3]).to(expected_device)]
        with self.assertRaises(ValueError):
            dist.get_data_device(data)

        # data is a list containing Tensor and a dict
        data = [
            torch.tensor([0, 1]).to(expected_device), {
                'key': torch.tensor([2, 3]).to(expected_device)
            }
        ]
        self.assertEqual(dist.get_data_device(data), expected_device)

        # data is a list containing Tensor and a dict but the dict contains
        # invalid type
        data = [torch.tensor([0, 1]).to(expected_device), {'key': '123'}]
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a empty list
        with self.assertRaises(ValueError):
            dist.get_data_device([])

        # data is a dict
        data = {
            'key1': torch.tensor([0, 1]).to(expected_device),
            'key2': torch.tensor([0, 1]).to(expected_device)
        }
        self.assertEqual(dist.get_data_device(data), expected_device)

        # data is a dict but not all values are Tensor
        data = {'key1': torch.tensor([0, 1]).to(expected_device), 'key2': 123}
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a dict but not all values have the same device type
        data = {
            'key1': torch.tensor([0, 1]),
            'key2': torch.tensor([0, 1]).to(expected_device)
        }
        with self.assertRaises(ValueError):
            dist.get_data_device(data)

        # data is a dict and one of values is list of Tensor
        data = {
            'key1': torch.tensor([0, 1]).to(expected_device),
            'key2': [torch.tensor([0, 1]).to(expected_device)]
        }
        self.assertEqual(dist.get_data_device(data), expected_device)

        # data is a dict and one of values is an invalid type
        data = {
            'key1': torch.tensor([0, 1]).to(expected_device),
            'key2': ['123']
        }
        with self.assertRaises(TypeError):
            dist.get_data_device(data)

        # data is a empty dict
        with self.assertRaises(ValueError):
            dist.get_data_device({})

        # data is not a valid type
        with self.assertRaisesRegex(
                TypeError,
                'data should be a Tensor, sequence of tensor or dict'):
            dist.get_data_device('123')

    def test_get_comm_device(self):
        self._init_dist_env(self.rank, self.world_size)
        group = dist.get_default_group()
        expected = torch.device('cuda', torch.cuda.current_device())
        self.assertEqual(dist.get_comm_device(group), expected)

    def test_cast_data_device(self):
        self._init_dist_env(self.rank, self.world_size)

        expected_device = torch.device('cuda', torch.cuda.current_device())
        # data is a Tensor
        data = torch.tensor([0, 1])
        output = dist.cast_data_device(data, expected_device)
        self.assertEqual(output.device, expected_device)

        # data is a Tensor and out is also a Tensor
        data = torch.tensor([0, 1])
        out = torch.tensor([1, 2])
        output = dist.cast_data_device(data, expected_device, out=out)
        self.assertEqual(output.device, expected_device)
        self.assertTrue(torch.allclose(output.cpu(), out))

        # data is a list of Tensor
        data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        for item in dist.cast_data_device(data, expected_device):
            self.assertEqual(item.device, expected_device)

        # both data and out are list of tensor
        data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        out = [torch.tensor([3, 4]), torch.tensor([5, 6])]
        output = dist.cast_data_device(data, expected_device, out=out)
        for item1, item2 in zip(output, out):
            self.assertEqual(item1.device, expected_device)
            self.assertTrue(torch.allclose(item1.cpu(), item2))

        # data is a list containing a Tensor and a dict
        data = [torch.tensor([0, 1]), {'key': torch.tensor([2, 3])}]
        output = dist.cast_data_device(data, expected_device)
        self.assertEqual(output[0].device, expected_device)
        self.assertEqual(output[1]['key'].device, expected_device)

        # data is a list containing a Tensor and a dict, so does out
        data = [torch.tensor([0, 1]), {'key': torch.tensor([2, 3])}]
        out = [torch.tensor([3, 4]), {'key': torch.tensor([5, 6])}]
        output = dist.cast_data_device(data, expected_device, out=out)
        self.assertEqual(output[0].device, expected_device)
        self.assertTrue(torch.allclose(output[0].cpu(), out[0]))
        self.assertEqual(output[1]['key'].device, expected_device)
        self.assertTrue(torch.allclose(output[1]['key'].cpu(), out[1]['key']))

        # data is an empty list
        with self.assertRaisesRegex(ValueError, 'data should not be empty'):
            dist.cast_data_device([], expected_device)

        # data is a dict
        data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([2, 3])}
        output = dist.cast_data_device(data, expected_device)
        for k, v in output.items():
            self.assertEqual(v.device, expected_device)

        # data is a dict, so does out
        data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([2, 3])}
        out = {'key1': torch.tensor([3, 4]), 'key2': torch.tensor([5, 6])}
        output = dist.cast_data_device(data, expected_device, out=out)
        for k, v in output.items():
            self.assertEqual(v.device, expected_device)
            self.assertTrue(torch.allclose(v.cpu(), out[k]))

        # the length of data and out should be same
        data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([2, 3])}
        out = {'key1': torch.tensor([3, 4])}
        with self.assertRaisesRegex(ValueError,
                                    'length of data and out should be same'):
            dist.cast_data_device(data, expected_device, out=out)

        # data is an empty dict
        with self.assertRaisesRegex(ValueError, 'data should not be empty'):
            dist.cast_data_device({}, expected_device)

        # data is a dict and one of values is list
        data = {'key1': torch.tensor([0, 1]), 'key2': [torch.tensor([2, 3])]}
        out = {'key1': torch.tensor([3, 4]), 'key2': [torch.tensor([5, 6])]}
        output = dist.cast_data_device(data, expected_device, out=out)
        self.assertEqual(output['key1'].device, expected_device)
        self.assertTrue(torch.allclose(output['key1'].cpu(), out['key1']))
        self.assertEqual(output['key2'][0].device, expected_device)
        self.assertTrue(
            torch.allclose(output['key2'][0].cpu(), out['key2'][0]))

        # data is not a valid type
        with self.assertRaisesRegex(
                TypeError, 'data should be a Tensor, list of tensor or dict'):
            dist.cast_data_device(123, expected_device)

        with self.assertRaisesRegex(
                TypeError, 'data should be a Tensor, list of tensor or dict'):
            dist.cast_data_device('123', expected_device)

        with self.assertRaisesRegex(
                TypeError, 'data should be a Tensor, list of tensor or dict'):
            dist.cast_data_device(np.array([0, 1]), expected_device)

        # data and out are not the same type
        data = torch.tensor([0, 1])
        out = '123'
        with self.assertRaisesRegex(TypeError,
                                    'out should be the same type with data'):
            dist.cast_data_device(data, expected_device, out=out)

        data = {0, 1}
        out = {2, 3}
        with self.assertRaisesRegex(TypeError, 'out should not be a set'):
            dist.cast_data_device(data, expected_device, out=out)
