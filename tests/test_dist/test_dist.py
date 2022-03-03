# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp

import mmengine.dist as dist
from mmengine.dist.dist import sync_random_seed
from mmengine.utils import TORCH_VERSION, digit_version


def _test_all_reduce_non_dist():
    data = torch.arange(2, dtype=torch.int64)
    expected = torch.arange(2, dtype=torch.int64)
    dist.all_reduce(data)
    assert torch.allclose(data, expected)


def _test_all_gather_non_dist():
    data = torch.arange(2, dtype=torch.int64)
    expected = torch.arange(2, dtype=torch.int64)
    output = dist.all_gather(data)
    assert torch.allclose(output[0], expected)


def _test_gather_non_dist():
    data = torch.arange(2, dtype=torch.int64)
    expected = torch.arange(2, dtype=torch.int64)
    output = dist.gather(data)
    assert torch.allclose(output[0], expected)


def _test_broadcast_non_dist():
    data = torch.arange(2, dtype=torch.int64)
    expected = torch.arange(2, dtype=torch.int64)
    dist.broadcast(data)
    assert torch.allclose(data, expected)


@patch('numpy.random.randint', return_value=10)
def _test_sync_random_seed_no_dist(mock):
    assert sync_random_seed() == 10


def _test_broadcast_object_list_no_dist():
    with pytest.raises(AssertionError):
        # input should be list of object
        dist.broadcast_object_list('foo')

    data = ['foo', 12, {1: 2}]
    expected = ['foo', 12, {1: 2}]
    dist.broadcast_object_list(data)
    assert data == expected


def _test_all_reduce_dict_no_dist():
    with pytest.raises(AssertionError):
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
        assert torch.allclose(data[key], expected[key])


def _test_all_gather_object_no_dist():
    data = 'foo'
    expected = 'foo'
    gather_objects = dist.all_gather_object(data)
    assert gather_objects[0] == expected


def _test_gather_object_no_dist():
    data = 'foo'
    expected = 'foo'
    gather_objects = dist.gather_object(data)
    assert gather_objects[0] == expected


def _test_collect_results_non_dist():
    data = ['foo', {1: 2}]
    size = 2
    expected = ['foo', {1: 2}]

    # test `device=cpu`
    output = dist.collect_results(data, size, device='cpu')
    assert output == expected

    # test `device=gpu`
    output = dist.collect_results(data, size, device='cpu')
    assert output == expected


def init_process(rank, world_size, functions, backend='gloo'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29505'
    os.environ['RANK'] = str(rank)
    dist.init_dist('pytorch', backend, rank=rank, world_size=world_size)

    device = 'cpu' if backend == 'gloo' else 'cuda'

    for func in functions:
        func(device)


def main(functions, world_size=2, backend='gloo'):
    try:
        mp.spawn(
            init_process,
            args=(world_size, functions, backend),
            nprocs=world_size)
    except Exception:
        pytest.fail(f'{backend} failed')


def _test_all_reduce_dist(device):
    for tensor_type, reduce_op in zip([torch.int64, torch.float32],
                                      ['sum', 'mean']):
        if dist.get_rank() == 0:
            data = torch.tensor([1, 2], dtype=tensor_type).to(device)
        else:
            data = torch.tensor([3, 4], dtype=tensor_type).to(device)

        if reduce_op == 'sum':
            expected = torch.tensor([4, 6], dtype=tensor_type).to(device)
        else:
            expected = torch.tensor([2, 3], dtype=tensor_type).to(device)

        dist.all_reduce(data, reduce_op)
        assert torch.allclose(data, expected)


def _test_all_gather_dist(device):
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1]).to(device)
    else:
        data = torch.tensor([1, 2]).to(device)

    expected = [
        torch.tensor([0, 1]).to(device),
        torch.tensor([1, 2]).to(device)
    ]

    output = dist.all_gather(data)
    assert torch.allclose(output[dist.get_rank()], expected[dist.get_rank()])


def _test_gather_dist(device):
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1]).to(device)
    else:
        data = torch.tensor([1, 2]).to(device)

    output = dist.gather(data)

    if dist.get_rank() == 0:
        expected = [
            torch.tensor([0, 1]).to(device),
            torch.tensor([1, 2]).to(device)
        ]
        for i in range(2):
            assert torch.allclose(output[i], expected[i])
    else:
        assert output == []


def _test_broadcast_dist(device):
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1]).to(device)
    else:
        data = torch.tensor([1, 2]).to(device)

    expected = torch.tensor([0, 1]).to(device)
    dist.broadcast(data, 0)
    assert torch.allclose(data, expected)


def _test_sync_random_seed_dist(device):
    with patch.object(
            torch, 'tensor',
            return_value=torch.tensor(1024).to(device)) as mock_tensor:
        output = dist.sync_random_seed()
        assert output == 1024
    mock_tensor.assert_called()


def _test_broadcast_object_list_dist(device):
    if dist.get_rank() == 0:
        data = ['foo', 12, {1: 2}]
    else:
        data = [None, None, None]

    expected = ['foo', 12, {1: 2}]

    dist.broadcast_object_list(data)

    assert data == expected


def _test_all_reduce_dict_dist(device):
    for tensor_type, reduce_op in zip([torch.int64, torch.float32],
                                      ['sum', 'mean']):
        if dist.get_rank() == 0:
            data = {
                'key1': torch.tensor([0, 1], dtype=tensor_type).to(device),
                'key2': torch.tensor([1, 2], dtype=tensor_type).to(device)
            }
        else:
            data = {
                'key1': torch.tensor([2, 3], dtype=tensor_type).to(device),
                'key2': torch.tensor([3, 4], dtype=tensor_type).to(device)
            }

        if reduce_op == 'sum':
            expected = {
                'key1': torch.tensor([2, 4], dtype=tensor_type).to(device),
                'key2': torch.tensor([4, 6], dtype=tensor_type).to(device)
            }
        else:
            expected = {
                'key1': torch.tensor([1, 2], dtype=tensor_type).to(device),
                'key2': torch.tensor([2, 3], dtype=tensor_type).to(device)
            }

        dist.all_reduce_dict(data, reduce_op)

        for key in data:
            assert torch.allclose(data[key], expected[key])

    # `torch.cat` in torch1.5 can not concatenate different types so we
    # fallback to convert them all to float type.
    if digit_version(TORCH_VERSION) == digit_version('1.5.0'):
        if dist.get_rank() == 0:
            data = {
                'key1': torch.tensor([0, 1], dtype=torch.float32).to(device),
                'key2': torch.tensor([1, 2], dtype=torch.int32).to(device)
            }
        else:
            data = {
                'key1': torch.tensor([2, 3], dtype=torch.float32).to(device),
                'key2': torch.tensor([3, 4], dtype=torch.int32).to(device)
            }

        expected = {
            'key1': torch.tensor([2, 4], dtype=torch.float32).to(device),
            'key2': torch.tensor([4, 6], dtype=torch.float32).to(device)
        }

        dist.all_reduce_dict(data, 'sum')

        for key in data:
            assert torch.allclose(data[key], expected[key])


def _test_all_gather_object_dist(device):
    if dist.get_rank() == 0:
        data = 'foo'
    else:
        data = {1: 2}

    expected = ['foo', {1: 2}]
    output = dist.all_gather_object(data)

    assert output == expected


def _test_gather_object_dist(device):
    if dist.get_rank() == 0:
        data = 'foo'
    else:
        data = {1: 2}

    output = dist.gather_object(data, dst=0)

    if dist.get_rank() == 0:
        assert output == ['foo', {1: 2}]
    else:
        assert output is None


def _test_collect_results_dist(device):
    if dist.get_rank() == 0:
        data = ['foo', {1: 2}]
    else:
        data = [24, {'a': 'b'}]

    size = 4

    expected = ['foo', 24, {1: 2}, {'a': 'b'}]

    # test `device=cpu`
    output = dist.collect_results(data, size, device='cpu')
    if dist.get_rank() == 0:
        assert output == expected
    else:
        assert output is None

    # test `device=cpu` and `tmpdir is not None`
    tmpdir = tempfile.mkdtemp()
    # broadcast tmpdir to all ranks to make it consistent
    object_list = [tmpdir]
    dist.broadcast_object_list(object_list)
    output = dist.collect_results(
        data, size, device='cpu', tmpdir=object_list[0])
    if dist.get_rank() == 0:
        assert output == expected
    else:
        assert output is None

    if dist.get_rank() == 0:
        # object_list[0] will be removed by `dist.collect_results`
        assert not osp.exists(object_list[0])

    # test `device=gpu`
    output = dist.collect_results(data, size, device='gpu')
    if dist.get_rank() == 0:
        assert output == expected
    else:
        assert output is None


def test_non_distributed_env():
    _test_all_reduce_non_dist()
    _test_all_gather_non_dist()
    _test_gather_non_dist()
    _test_broadcast_non_dist()
    _test_sync_random_seed_no_dist()
    _test_broadcast_object_list_no_dist()
    _test_all_reduce_dict_no_dist()
    _test_all_gather_object_no_dist()
    _test_gather_object_no_dist()
    _test_collect_results_non_dist()


def test_gloo_backend():
    functions_to_test = [
        _test_all_reduce_dist,
        _test_all_gather_dist,
        _test_gather_dist,
        _test_broadcast_dist,
        _test_sync_random_seed_dist,
        _test_broadcast_object_list_dist,
        _test_all_reduce_dict_dist,
        _test_all_gather_object_dist,
        _test_gather_object_dist,
    ]
    main(functions_to_test, backend='gloo')


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason='need 2 gpu to test nccl')
def test_nccl_backend():
    functions_to_test = [
        _test_all_reduce_dist,
        _test_all_gather_dist,
        _test_broadcast_dist,
        _test_sync_random_seed_dist,
        _test_broadcast_object_list_dist,
        _test_all_reduce_dict_dist,
        _test_all_gather_object_dist,
        _test_collect_results_dist,
    ]
    main(functions_to_test, backend='nccl')
