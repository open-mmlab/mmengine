# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmengine.model import convert_sync_batchnorm


def test_convert_sync_batchnorm_keeps_training_state():
    bn = nn.BatchNorm2d(4)
    bn.eval()

    sync_bn = convert_sync_batchnorm(bn)

    assert isinstance(sync_bn, nn.SyncBatchNorm)
    assert sync_bn.training is False
