# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.device import get_device, is_cuda_available, is_mlu_available


def test_get_device():
    device = get_device()
    if is_cuda_available():
        assert device == 'cuda'
    elif is_mlu_available():
        assert device == 'mlu'
    else:
        assert device == 'cpu'
