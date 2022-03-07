# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Optional, Sequence

import cv2

from mmengine.data import BaseDataSample
from mmengine.hooks import Hook
from mmengine.registry import HOOKS  # type: ignore
from mmengine.utils.misc import tensor2imgs  # type: ignore


@HOOKS.register_module()
class TestVisualizationHook(Hook):

    def __init__(self, show=False, draw_gt=False, draw_pred=True):
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self.show = show

    def after_test_iter(
            self,
            runner: object,
            data_batch: Optional[Sequence[BaseDataSample]] = None,
            outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        inputs, data_samples = data_batch  # type: ignore
        for img, data_sample, output in zip(
                inputs,  # type: ignore
                data_samples,  # type: ignore
                outputs):  # type: ignore
            img = tensor2imgs(img, **data_samples.img_norm_cfg)  # type: ignore
            img = cv2.resize(
                img,
                (
                    data_samples.ori_width,  # type: ignore
                    data_samples.ori_height))  # type: ignore
            name = osp.basename(data_sample.img_path)
            runner.writer.add_image(  # type: ignore
                name,
                img,
                data_sample,
                output,
                self.draw_gt,
                self.draw_pred,
            )
