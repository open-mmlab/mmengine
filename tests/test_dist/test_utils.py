# Copyright (c) OpenMMLab. All rights reserved.
import os

import pytest
import torch
import torch.distributed as torch_dist
import torch.multiprocessing as mp

import mmengine.dist as dist


def _test_get_backend_non_dist():
    assert dist.get_backend() is None


def _test_get_world_size_non_dist():
    assert dist.get_world_size() == 1


def _test_get_rank_non_dist():
    assert dist.get_rank() == 0


def _test_local_size_non_dist():
    assert dist.get_local_size() == 1


def _test_local_rank_non_dist():
    assert dist.get_local_rank() == 0


def _test_get_dist_info_non_dist():
    assert dist.get_dist_info() == (0, 1)


def _test_is_main_process_non_dist():
    assert dist.is_main_process()


def _test_master_only_non_dist():

    @dist.master_only
    def fun():
        assert dist.get_rank() == 0

    fun()


def _test_barrier_non_dist():
    dist.barrier()  # nothing is done


def init_process(rank, world_size, functions, backend='gloo'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = str(rank)

    if backend == 'nccl':
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)

    torch_dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size)
    dist.init_local_group(0, world_size)

    for func in functions:
        func()


def main(functions, world_size=2, backend='gloo'):
    try:
        mp.spawn(
            init_process,
            args=(world_size, functions, backend),
            nprocs=world_size)
    except Exception:
        pytest.fail('error')


def _test_get_backend_dist():
    assert dist.get_backend() == torch_dist.get_backend()


def _test_get_world_size_dist():
    assert dist.get_world_size() == 2


def _test_get_rank_dist():
    if torch_dist.get_rank() == 0:
        assert dist.get_rank() == 0
    else:
        assert dist.get_rank() == 1


def _test_local_size_dist():
    assert dist.get_local_size() == 2


def _test_local_rank_dist():
    torch_dist.get_rank(dist.get_local_group()) == dist.get_local_rank()


def _test_get_dist_info_dist():
    if dist.get_rank() == 0:
        assert dist.get_dist_info() == (0, 2)
    else:
        assert dist.get_dist_info() == (1, 2)


def _test_is_main_process_dist():
    if dist.get_rank() == 0:
        assert dist.is_main_process()
    else:
        assert not dist.is_main_process()


def _test_master_only_dist():

    @dist.master_only
    def fun():
        assert dist.get_rank() == 0

    fun()


def test_non_distributed_env():
    _test_get_backend_non_dist()
    _test_get_world_size_non_dist()
    _test_get_rank_non_dist()
    _test_local_size_non_dist()
    _test_local_rank_non_dist()
    _test_get_dist_info_non_dist()
    _test_is_main_process_non_dist()
    _test_master_only_non_dist()
    _test_barrier_non_dist()


functions_to_test = [
    _test_get_backend_dist,
    _test_get_world_size_dist,
    _test_get_rank_dist,
    _test_local_size_dist,
    _test_local_rank_dist,
    _test_get_dist_info_dist,
    _test_is_main_process_dist,
    _test_master_only_dist,
]


def test_gloo_backend():
    main(functions_to_test)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason='need 2 gpu to test nccl')
def test_nccl_backend():
    main(functions_to_test, backend='nccl')
