# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.device import (get_device, is_cuda_available, is_mlu_available,
                             is_mps_available, is_musa_available,
                             is_npu_available)


def test_get_device():
    device = get_device()
    if is_npu_available():
        assert device == 'npu'
    elif is_cuda_available():
        assert device == 'cuda'
    elif is_mlu_available():
        assert device == 'mlu'
    elif is_mps_available():
        assert device == 'mps'
    elif is_musa_available():
        assert device == 'musa'
    else:
        assert device == 'cpu'
