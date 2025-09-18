# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.utils.dl_utils import tensor2imgs

DATA_BATCH = Optional[Union[dict, tuple, list]]


# TODO: Due to interface changes, the current class
#  functions incorrectly
@HOOKS.register_module()
class NaiveVisualizationHook(Hook):
    """Show or Write the predicted results during the process of testing.

    Args:
        interval (int): Visualization interval. Defaults to 1.
        draw_gt (bool): Whether to draw the ground truth. Defaults to True.
        draw_pred (bool): Whether to draw the predicted result.
            Defaults to True.
    """
    priority = 'NORMAL'

    def __init__(self,
                 interval: int = 1,
                 draw_gt: bool = True,
                 draw_pred: bool = True):
        assert isinstance(interval, int) and interval > 0, (
            f'`interval` must be a positive integer, but got {interval}')
        assert isinstance(
            draw_gt, bool), (f'`draw_gt` must be a boolean, but got {draw_gt}')
        assert isinstance(
            draw_pred,
            bool), (f'`draw_pred` must be a boolean, but got {draw_pred}')
        self.draw_gt = draw_gt
        self.draw_pred = draw_pred
        self._interval = interval

    def before_train(self, runner) -> None:
        """Call add_graph method of visualizer.

        Args:
            runner (Runner): The runner of the training process.
        """
        runner.visualizer.add_graph(runner.model, None)

    def after_test_iter(self,
                        runner,
                        batch_idx: int,
                        data_batch: DATA_BATCH = None,
                        outputs: Optional[Sequence] = None) -> None:
        """Show or Write the predicted results.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict or tuple or list, optional): Data from dataloader.
            outputs (Sequence, optional): Outputs from model.
        """
        if self.every_n_inner_iters(batch_idx, self._interval):
            for idx, output in enumerate(outputs):  # type: ignore
                inputs = data_batch['inputs'][idx]  # type: ignore
                inputs = torch.stack([inputs])
                data_sample = data_batch['data_samples'][idx]  # type: ignore
                # TODO We will implement a function to revert the augmentation
                # in the future.
                inputs = tensor2imgs(inputs)[0]
                name = f'{batch_idx}_{idx}'
                runner.visualizer.add_datasample(name, inputs, data_sample,
                                                 output, self.draw_gt,
                                                 self.draw_pred)
