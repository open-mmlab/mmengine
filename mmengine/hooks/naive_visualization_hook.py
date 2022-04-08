# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from mmengine.data import BaseDataElement
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.utils.misc import tensor2imgs


@HOOKS.register_module()
class NaiveVisualizationHook(Hook):
    """Show or Write the predicted results during the process of testing.

    Args:
        interval (int): Visualization interval. Default: 1.
        draw_gt (bool): Whether to draw the ground truth. Default to True.
        draw_pred (bool): Whether to draw the predicted result.
            Default to True.
    """
    priority = 'NORMAL'

    def __init__(self,
                 interval: int = 1,
                 draw_gt: bool = True,
                 draw_pred: bool = True):
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self._interval = interval

    def _unpad(self, input: np.ndarray, unpad_shape: Tuple[int,
                                                           int]) -> np.ndarray:
        unpad_width, unpad_height = unpad_shape
        unpad_image = input[:unpad_height, :unpad_width]
        return unpad_image

    def after_test_iter(
            self,
            runner,
            batch_idx: int,
            data_batch: Optional[Sequence[Tuple[Any, BaseDataElement]]] = None,
            outputs: Optional[Sequence[BaseDataElement]] = None) -> None:
        """Show or Write the predicted results.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (Sequence[Tuple[Any, BaseDataElement]], optional): Data
                from dataloader. Defaults to None.
            outputs (Sequence[BaseDataElement], optional): Outputs from model.
                Defaults to None.
        """
        if self.every_n_iters(runner, self._interval):
            inputs, data_samples = data_batch  # type: ignore
            inputs = tensor2imgs(inputs,
                                 **data_samples[0].get('img_norm_cfg', dict()))
            for input, data_sample, output in zip(
                    inputs,
                    data_samples,  # type: ignore
                    outputs):  # type: ignore
                # TODO We will implement a function to revert the augmentation
                # in the future.
                ori_shape = (data_sample.ori_width, data_sample.ori_height)
                if 'pad_shape' in data_sample:
                    input = self._unpad(input,
                                        data_sample.get('scale', ori_shape))
                origin_image = cv2.resize(input, ori_shape)
                name = osp.basename(data_sample.img_path)
                runner.writer.add_image(name, origin_image, data_sample,
                                        output, self.draw_gt, self.draw_pred)
