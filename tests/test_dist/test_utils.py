# Copyright (c) OpenMMLab. All rights reserved.
import os

import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as mp

import mmengine.dist as dist


def test_get_backend():
    assert dist.get_backend() is None


def test_get_world_size():
    assert dist.get_world_size() == 1


def test_get_rank():
    assert dist.get_rank() == 0


def test_local_size():
    assert dist.get_local_size() == 1


def test_local_rank():
    assert dist.get_local_rank() == 0


def test_get_dist_info():
    assert dist.get_dist_info() == (0, 1)


def test_is_main_process():
    assert dist.is_main_process()


def test_master_only():

    @dist.master_only
    def fun():
        assert dist.get_rank() == 0

    fun()


def test_barrier():
    dist.barrier()  # nothing is done


def init_process(rank, world_size, functions, backend='gloo'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = str(rank)
    dist.init_dist('pytorch', backend, rank=rank, world_size=world_size)
    dist.init_local_group(0, world_size)

    for func in functions:
        func()


def main(functions, world_size=2, backend='gloo'):
    try:
        mp.spawn(
            init_process,
            args=(world_size, functions, backend),
            nprocs=world_size)
    except mp.Exception:
        pytest.fail('error')


def _test_get_backend():
    assert dist.get_backend() == torch_dist.get_backend()


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


def test_gloo():
    main(functions_to_test)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason='need 2 gpu to test nccl')
def test_nccl():
    main(functions_to_test, backend='nccl')
