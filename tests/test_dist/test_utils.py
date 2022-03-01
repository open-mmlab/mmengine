# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dist import get_backend
from mmengine.dist.utils import (barrier, get_dist_info, get_local_rank,
                                 get_local_size, get_rank, get_world_size,
                                 is_main_process)


def test_get_backend():
    assert get_backend() is None


def test_get_world_size():
    assert get_world_size() == 1


def test_get_rank():
    assert get_rank() == 0


def test_local_size():
    assert get_local_size() == 1


def test_local_rank():
    assert get_local_rank() == 0


def test_get_dist_info():
    assert get_dist_info() == (0, 1)


def test_is_main_process():
    assert is_main_process()


def test_master_only():
    pass


def test_barrier():
    barrier()  # nothing is done
