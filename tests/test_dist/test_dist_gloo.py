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


def main(functions, world_size=2):
    try:
        mp.spawn(init_process, args=(world_size, functions), nprocs=world_size)
    except mp.ProcessRaisedException:
        pytest.fail('error')


def _test_all_reduce():
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1])
    else:
        data = torch.tensor([1, 2])

    expected = torch.tensor([1, 3])

    dist.all_reduce(data)
    assert torch.allclose(data, expected)


def _test_all_gather():
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1])
    else:
        data = torch.tensor([1, 2])

    expected = [torch.tensor([0, 1]), torch.tensor([1, 2])]

    output = dist.all_gather(data)
    assert torch.allclose(output[dist.get_rank()], expected[dist.get_rank()])


def _test_gather():
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


def _test_broadcast():
    if dist.get_rank() == 0:
        data = torch.tensor([0, 1])
    else:
        data = torch.tensor([1, 2])

    expected = torch.tensor([0, 1])
    dist.broadcast(data, 0)
    assert torch.allclose(data, expected)


def _test_sync_random_seed():
    with patch.object(
            torch, 'tensor', return_value=torch.tensor(1024)) as mock_tensor:
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
        data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([1, 2])}
    else:
        data = {'key1': torch.tensor([2, 3]), 'key2': torch.tensor([3, 4])}

    expected = {'key1': torch.tensor([2, 4]), 'key2': torch.tensor([4, 6])}

    dist.all_reduce_dict(data)

    for key in data:
        assert torch.allclose(data[key], expected[key])


def _test_all_gather_object():
    if dist.get_rank() == 0:
        data = 'foo'
    else:
        data = {1: 2}

    expected = ['foo', {1: 2}]
    output = dist.all_gather_object(data)

    assert output == expected


def _test_gather_object():
    if dist.get_rank() == 0:
        data = 'foo'
    else:
        data = {1: 2}

    output = dist.gather_object(data, dst=0)

    if dist.get_rank() == 0:
        assert output == ['foo', {1: 2}]
    else:
        assert output is None


def _test_collect_results_cpu():
    pass


def test_all_functions():
    functions_to_test = [
        _test_all_reduce,
        _test_all_gather,
        # _test_gather,
        _test_broadcast,
        _test_sync_random_seed,
        _test_broadcast_object_list,
        _test_all_reduce_dict,
        _test_all_gather_object,
        _test_gather_object,
    ]
    main(functions_to_test)
