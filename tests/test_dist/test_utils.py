# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.dist import get_backend
from mmengine.dist.utils import (barrier, get_dist_info, get_local_rank,
                                 get_local_size, get_rank, get_world_size,
                                 is_main_process)


class TestNonDistributedEnv:

    def test_get_backend(self):
        assert get_backend() is None

    def test_get_world_size(self):
        assert get_world_size() == 1

    def test_get_rank(self):
        assert get_rank() == 0

    def test_local_size(self):
        assert get_local_size() == 1

    def test_local_rank(self):
        assert get_local_rank() == 0

    def test_get_dist_info(self):
        assert get_dist_info() == (0, 1)

    def test_is_main_process(self):
        assert is_main_process()

    def test_master_only(self):
        pass

    def test_barrier(self):
        barrier()  # nothing is done
