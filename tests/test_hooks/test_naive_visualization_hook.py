# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import torch

from mmengine.hooks import NaiveVisualizationHook
from mmengine.structures import BaseDataElement


class TestNaiveVisualizationHook:

    def test_after_train_iter(self):
        naive_visualization_hook = NaiveVisualizationHook()
        runner = Mock(iter=1)
        runner.visualizer.add_image = Mock()
        inputs = torch.randn(1, 3, 15, 15)
        batch_idx = 10
        # test with normalize, resize, pad
        gt_datasamples = BaseDataElement(
            metainfo=dict(
                img_norm_cfg=dict(
                    mean=(0, 0, 0), std=(0.5, 0.5, 0.5), to_bgr=True),
                scale=(10, 10),
                pad_shape=(15, 15, 3),
                ori_height=5,
                ori_width=5,
                img_path='tmp.jpg'))
        pred_datasamples = [BaseDataElement()]
        data_batch = [dict(inputs=inputs, data_sample=gt_datasamples)]
        naive_visualization_hook.after_test_iter(runner, batch_idx, data_batch,
                                                 pred_datasamples)
        # test with resize, pad
        gt_datasamples = BaseDataElement(
            metainfo=dict(
                scale=(10, 10),
                pad_shape=(15, 15, 3),
                ori_height=5,
                ori_width=5,
                img_path='tmp.jpg'))
        pred_datasamples = [BaseDataElement()]
        data_batch = [dict(inputs=inputs, data_sample=gt_datasamples)]
        naive_visualization_hook.after_test_iter(runner, batch_idx, data_batch,
                                                 pred_datasamples)
        # test with only resize
        gt_datasamples = BaseDataElement(
            metainfo=dict(
                scale=(15, 15), ori_height=5, ori_width=5, img_path='tmp.jpg'))
        pred_datasamples = [BaseDataElement()]
        data_batch = [dict(inputs=inputs, data_sample=gt_datasamples)]
        naive_visualization_hook.after_test_iter(runner, batch_idx, data_batch,
                                                 pred_datasamples)

        # test with only pad
        gt_datasamples = BaseDataElement(
            metainfo=dict(
                pad_shape=(15, 15, 3),
                ori_height=5,
                ori_width=5,
                img_path='tmp.jpg'))
        pred_datasamples = [BaseDataElement()]
        data_batch = [dict(inputs=inputs, data_sample=gt_datasamples)]
        naive_visualization_hook.after_test_iter(runner, batch_idx, data_batch,
                                                 pred_datasamples)

        # test no transform
        gt_datasamples = BaseDataElement(
            metainfo=dict(ori_height=15, ori_width=15, img_path='tmp.jpg'))
        pred_datasamples = [BaseDataElement()]
        data_batch = [dict(inputs=inputs, data_sample=gt_datasamples)]
        naive_visualization_hook.after_test_iter(runner, batch_idx, data_batch,
                                                 pred_datasamples)
