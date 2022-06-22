# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn as nn

from mmengine.utils import revert_sync_batchnorm


@pytest.mark.skipif(
    torch.__version__ == 'parrots', reason='not supported in parrots now')
def test_revert_syncbn():
    # conv = ConvModule(3, 8, 2, norm_cfg=dict(type='SyncBN'))
    conv = nn.Sequential(nn.Conv2d(3, 8, 2), nn.SyncBatchNorm(8))
    x = torch.randn(1, 3, 10, 10)
    # Expect a ValueError prompting that SyncBN is not supported on CPU
    with pytest.raises(ValueError):
        y = conv(x)
    conv = revert_sync_batchnorm(conv)
    y = conv(x)
    assert y.shape == (1, 8, 9, 9)
