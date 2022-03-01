# Copyright (c) OpenMMLab. All rights reserved.
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
    pass


def test_barrier():
    dist.barrier()  # nothing is done
