# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import torch

from mmengine.data import BaseDataSample
from mmengine.hooks import NaiveVisualizationHook


class TestNaiveVisualizationHook:

    def test_after_train_iter(self):
        naive_visualization_hook = NaiveVisualizationHook()
        Runner = Mock(iter=1)
        Runner.writer.add_image = Mock()
        inputs = torch.randn(2, 3, 15, 15)
        gt_datasamples = [
            BaseDataSample(
                metainfo=dict(
                    img_norm_cfg=dict(
                        mean=(0, 0, 0), std=(0.5, 0.5, 0.5), to_rgb=True),
                    scale=(10, 10),
                    pad_shape=(15, 15, 3),
                    ori_height=5,
                    ori_width=5,
                    img_path='tmp.jpg')),
            BaseDataSample(
                metainfo=dict(
                    img_norm_cfg=dict(
                        mean=(0, 0, 0), std=(0.5, 0.5, 0.5), to_rgb=True),
                    scale=(10, 10),
                    pad_shape=(15, 15, 3),
                    ori_height=5,
                    ori_width=5,
                    img_path='tmp.jpg'))
        ]
        pred_datasamples = [BaseDataSample(), BaseDataSample()]
        data_batch = (inputs, gt_datasamples)
        naive_visualization_hook.after_test_iter(Runner, data_batch,
                                                 pred_datasamples)
