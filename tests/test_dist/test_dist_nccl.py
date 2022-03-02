# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp

import mmengine.dist as dist


def init_process(rank, world_size, functions, backend='gloo'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = str(rank)
    dist.init_dist('pytorch', backend, rank=rank, world_size=world_size)

    for func in functions:
        func()


def main(functions, world_size=2, backend='gloo'):
    try:
        mp.spawn(
            init_process,
            args=(world_size, functions, backend),
            nprocs=world_size)
    except mp.ProcessRaisedException:
        pytest.fail(f'{backend} failed')


def _test_all_reduce():
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1]).to('cuda')
    else:
        data = torch.tensor([1, 2]).to('cuda')

    expected = torch.tensor([1, 3]).to('cuda')

    dist.all_reduce(data)
    assert torch.allclose(data, expected)


def _test_all_gather():
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1]).to('cuda')
    else:
        data = torch.tensor([1, 2]).to('cuda')

    expected = [
        torch.tensor([0, 1]).to('cuda'),
        torch.tensor([1, 2]).to('cuda')
    ]

    output = dist.all_gather(data)
    assert torch.allclose(output[dist.get_rank()], expected[dist.get_rank()])


def _test_broadcast():
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1]).to('cuda')
    else:
        data = torch.tensor([1, 2]).to('cuda')

    expected = torch.tensor([0, 1]).to('cuda')
    dist.broadcast(data, 0)
    assert torch.allclose(data, expected)


def _test_sync_random_seed():
    with patch.object(
            torch, 'tensor',
            return_value=torch.tensor(1024).to('cuda')) as mock_tensor:
        output = dist.sync_random_seed()
        assert output == 1024
    mock_tensor.assert_called()


def _test_broadcast_object_list():
    if dist.get_rank() == 0:
        data = ['foo', 12, {1: 2}]
    else:
        data = [None, None, None]

    expected = ['foo', 12, {1: 2}]

    dist.broadcast_object_list(data)

    assert data == expected


def _test_all_reduce_dict():
    if dist.get_rank() == 0:
        data = {
            'key1': torch.tensor([0, 1]).to('cuda'),
            'key2': torch.tensor([1, 2]).to('cuda')
        }
    else:
        data = {
            'key1': torch.tensor([2, 3]).to('cuda'),
            'key2': torch.tensor([3, 4]).to('cuda')
        }

    expected = {
        'key1': torch.tensor([2, 4]).to('cuda'),
        'key2': torch.tensor([4, 6]).to('cuda')
    }

    dist.all_reduce_dict(data)

    for key in data:
        assert torch.allclose(data[key], expected[key])


def _test_collect_results_cpu():
    pass


def test_all_functions():

    if torch.cuda.device_count() < 2:
        pytest.skip('need 2 gpu to test nccl')

    functions_to_test = [
        _test_all_reduce,
        _test_all_gather,
        _test_broadcast,
        _test_sync_random_seed,
        _test_broadcast_object_list,
        _test_all_reduce_dict,
    ]
    main(functions_to_test, backend='nccl')
