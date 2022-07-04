# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import logging
from typing import Dict, Optional

from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, MODELS
from .hook import DATA_BATCH, Hook


@HOOKS.register_module()
class EMAHook(Hook):
    """A Hook to apply Exponential Moving Average (EMA) on the model during
    training.

    Note:
        - EMAHook takes priority over CheckpointHook.
        - The original model parameters are actually saved in ema field after
          train.

    Args:
        ema_type (str): The type of EMA strategy to use. You can find the
            supported strategies in ``mmengine.model.averaged_model``.
            Defaults to 'ExponentialMovingAverage'
    """

    priority = 'NORMAL'

    def __init__(self, ema_type: str = 'ExponentialMovingAverage', **kwargs):
        self.ema_cfg = dict(type=ema_type, **kwargs)

    def before_run(self, runner) -> None:
        """Create an ema copy of the model."""
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model
        self.ema_model = MODELS.build(
            self.ema_cfg, default_args=dict(model=self.src_model))

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        """Update ema parameter."""
        self.ema_model.update_parameters(self.src_model)

    def before_val_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before
        validation."""
        self._swap_ema_parameters()

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """We recover source model's parameter from ema model after
        validation."""
        self._swap_ema_parameters()

    def before_test_epoch(self, runner) -> None:
        """We load parameter values from ema model to source model before
        test."""
        self._swap_ema_parameters()

    def after_test_epoch(self,
                         runner,
                         metrics: Optional[Dict[str, float]] = None) -> None:
        """We recover source model's parameter from ema model after test."""
        self._swap_ema_parameters()

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        """Save ema parameters to checkpoint."""
        checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        # Save ema parameters to the source model's state dict so that we can
        # directly load the averaged model weights for deployment.
        # Swapping the state_dict key-values instead of swapping model
        # parameters because the state_dict is a shallow copy of model
        # parameters.
        self._swap_ema_state_dict(checkpoint)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """Resume ema parameters from checkpoint."""

        if 'ema_state_dict' in checkpoint:
            # The original model parameters are actually saved in ema field.
            # swap the weights back to resume ema state.
            self._swap_ema_state_dict(checkpoint)
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])

        # Support load checkpoint without ema state dict.
        else:
            print_log(
                'There is no `ema_state_dict` in checkpoint. '
                '`EMAHook` will make a copy of `state_dict` as the '
                'initial `ema_state_dict`', 'current', logging.WARNING)
            self.ema_model.module.load_state_dict(
                copy.deepcopy(checkpoint['state_dict']))

    def _swap_ema_parameters(self) -> None:
        """Swap the parameter of model with ema_model."""
        avg_param = (
            itertools.chain(self.ema_model.module.parameters(),
                            self.ema_model.module.buffers())
            if self.ema_model.update_buffers else
            self.ema_model.module.parameters())
        src_param = (
            itertools.chain(self.src_model.parameters(),
                            self.src_model.buffers())
            if self.ema_model.update_buffers else self.src_model.parameters())
        for p_avg, p_src in zip(avg_param, src_param):
            tmp = p_avg.data.clone()
            p_avg.data.copy_(p_src.data)
            p_src.data.copy_(tmp)

    def _swap_ema_state_dict(self, checkpoint):
        """Swap the state dict values of model with ema_model."""
        model_state = checkpoint['state_dict']
        ema_state = checkpoint['ema_state_dict']
        for k in ema_state:
            if k[:7] == 'module.':
                tmp = ema_state[k]
                ema_state[k] = model_state[k[7:]]
                model_state[k[7:]] = tmp
