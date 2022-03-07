# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Optional, Sequence, Tuple

import cv2

from mmengine.data import BaseDataSample
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.utils.misc import tensor2imgs


@HOOKS.register_module()
class NaiveVisualizationHook(Hook):
    """Show or Write the predicted results. during the process of testing.

    Args:
        interval (int): Visualization interval. Default: 1.
        draw_gt (bool): Whether to draw the ground truth. Default to True.
        draw_pred (bool): Whether to draw the predicted result.
            Default to True.
    """
    priority = 'NORMAL'

    def __init__(self, interval: int = 1, draw_gt=True, draw_pred=True):
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self._interval = interval

    def after_test_iter(
            self,
            runner,
            data_batch: Optional[Sequence[Tuple[Any, BaseDataSample]]] = None,
            outputs: Optional[Sequence[BaseDataSample]] = None) -> None:
        """Show or Write the predicted results.

        Args:
            runner (Runner): The runner of the training process.
            data_batch (Sequence[Tuple[Any, BaseDataSample]], optional): Data
                from dataloader. Defaults to None.
            outputs (Sequence[BaseDataSample], optional): Outputs from model.
                Defaults to None.
        """
        if self.every_n_iters(runner, self._interval):
            inputs, data_samples = data_batch  # type: ignore
            inputs = tensor2imgs(inputs, **data_samples[0].img_norm_cfg)
            for input, data_sample, output in zip(
                    inputs,
                    data_samples,  # type: ignore
                    outputs):  # type: ignore

                input = cv2.resize(
                    input, (data_sample.ori_width, data_sample.ori_height))
                name = osp.basename(data_sample.img_path)
                runner.writer.add_image(name, input, data_sample, output,
                                        self.draw_gt, self.draw_pred)
