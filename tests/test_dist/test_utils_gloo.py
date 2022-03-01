# Copyright (c) OpenMMLab. All rights reserved.
import os

import pytest
import torch.distributed as torch_dist
import torch.multiprocessing as mp

import mmengine.dist as dist


def init_process(rank, world_size, functions, backend='gloo'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = str(rank)
    dist.init_dist('pytorch', backend, rank=rank, world_size=world_size)
    dist.init_local_group(0, world_size)

    for func in functions:
        func()


def main(functions, world_size=2):
    try:
        mp.spawn(init_process, args=(world_size, functions), nprocs=world_size)
    except mp.ProcessRaisedException:
        pytest.fail('error')


def _test_get_backend():
    assert dist.get_backend() == 'gloo'


def _test_get_world_size():
    assert dist.get_world_size() == 2


def _test_get_rank():
    if torch_dist.get_rank() == 0:
        assert dist.get_rank() == 0
    else:
        assert dist.get_rank() == 1


def _test_local_size():
    assert dist.get_local_size() == 2


def _test_local_rank():
    torch_dist.get_rank(dist.get_local_group()) == dist.get_local_rank()


def _test_get_dist_info():
    if dist.get_rank() == 0:
        assert dist.get_dist_info() == (0, 2)
    else:
        assert dist.get_dist_info() == (1, 2)


def _test_is_main_process():
    if dist.get_rank() == 0:
        assert dist.is_main_process()
    else:
        assert not dist.is_main_process()


def _test_master_only():

    @dist.master_only
    def fun():
        assert dist.get_rank() == 0

    fun()


def test_all_functions():
    functions_to_test = [
        _test_get_backend,
        _test_get_world_size,
        _test_get_rank,
        _test_local_size,
        _test_local_rank,
        _test_get_dist_info,
        _test_is_main_process,
        _test_master_only,
    ]
    main(functions_to_test)
