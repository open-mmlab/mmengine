# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch
import torch.multiprocessing as mp

import mmengine.dist as dist


def init_process(rank, size, func, backend='gloo'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    func()


def main(func, world_size=2):
    processes = []
    mp.set_start_method('spawn')
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def _test_all_reduce():
    cpu_tensors = [torch.zeros(128).uniform_() for i in range(2)]
    expected = torch.zeros(128)
    for t in cpu_tensors:
        expected.add_(t)

    dist.all_reduce(cpu_tensors[dist.get_rank()])
    for tensor in cpu_tensors:
        torch.allclose(tensor, expected)


def test_all_reduce():
    main(_test_all_reduce)
