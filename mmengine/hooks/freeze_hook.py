# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from mmengine.hooks import Hook
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS


@HOOKS.register_module()
class FreezeHook(Hook):
    """FreezeHook is used to freeze or unfreeze network layers when
    training to a specified epoch.

    Args:
        freeze_epoch (int): The epoch number to start freezing layers.
        freeze_layers (tuple[str]): Model layers containing the keyword in
            freeze_layers will freeze the gradient.
        unfreeze_epoch (int): The epoch number to start unfreezing layers.
        unfreeze_layers (tuple[str]): Model layers containing the keyword in
            unfreeze_layers will unfreeze the gradient.
        log_grad (bool): Whether to log the requires_grad of each layer.

    Notes:
        The GPU memory usage shown in the "nvidia-smi" command does not change
        when you freeze model layers.
        https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/4

    Examples:
        >>> # The simplest FreezeHook config.
        >>> freeze_hook_cfg = dict(
                type="FreezeHook",
                freeze_epoch=1,
                freeze_layers=("backbone",)
            )

    """

    def __init__(
        self,
        freeze_epoch: int,
        freeze_layers: Union[Sequence[str], str],
        unfreeze_epoch: Optional[int] = None,
        unfreeze_layers: Optional[Union[Sequence[str], str]] = None,
        log_grad: bool = False,
    ) -> None:
        # check arguments type
        if not isinstance(freeze_epoch, int):
            raise TypeError(f"freeze_epoch must be an integer")
        if not isinstance(freeze_layers, (tuple, list, str)):
            raise TypeError(f"freeze_layers must be a tuple, list or str")
        if not isinstance(unfreeze_epoch, (int, type(None))):
            raise TypeError(f"unfreeze_epoch must be an integer or None")
        if not isinstance(unfreeze_layers, (tuple, list, str, type(None))):
            raise TypeError(f"unfreeze_layers must be a tuple, list, str or None")
        if not isinstance(log_grad, bool):
            raise TypeError(f"log_grad must be a boolean")

        # check arguments value
        if freeze_epoch <= 0:
            raise ValueError(f"freeze_epoch must be greater than 0")
        if len(freeze_layers) == 0:
            raise ValueError(f"freeze_layers must not be empty")
        if unfreeze_epoch is not None and unfreeze_epoch <= 0:
            raise ValueError(f"unfreeze_epoch must be greater than 0")
        if unfreeze_epoch is not None and unfreeze_epoch <= freeze_epoch:
            raise ValueError(f"unfreeze_epoch must be greater than freeze_epoch")
        if unfreeze_epoch is not None and unfreeze_layers is None:
            raise ValueError(f"unfreeze_layers must not be None when unfreeze_epoch is not None.")
        if (unfreeze_epoch is None and unfreeze_layers is not None) or (
            unfreeze_epoch is not None and unfreeze_layers is None
        ):
            raise ValueError(f"unfreeze_epoch and unfreeze_layers must be both None or not None")

        self.freeze_epoch = freeze_epoch
        if isinstance(freeze_layers, str):
            freeze_layers = (freeze_layers,)
        self.freeze_layers = freeze_layers
        self.unfreeze_epoch = unfreeze_epoch
        if isinstance(unfreeze_layers, str):
            unfreeze_layers = (unfreeze_layers,)
        self.unfreeze_layers = unfreeze_layers
        self.log_grad = log_grad

    def modify_layers_grad(self, model, layers, requires_grad):
        if is_model_wrapper(model):
            model = model.module
        for k, v in model.named_parameters():
            for layer in layers:
                if layer in k:
                    v.requires_grad = requires_grad
                    break

    def log_model_grad(self, model, log_grad=False):
        if log_grad:
            for k, v in model.named_parameters():
                print_log(f"{k} requires_grad: {v.requires_grad}", logger="current")

    def before_train_epoch(self, runner) -> None:
        if (runner.epoch + 1) == self.freeze_epoch:
            self.modify_layers_grad(runner.model, self.freeze_layers, requires_grad=False)
            self.log_model_grad(runner.model, self.log_grad)
            print_log(f"Freeze {self.freeze_layers} at epoch {runner.epoch + 1}", logger="current")
            # if you want to release GPU memory cache:
            # import torch; torch.cuda.empty_cache()

        if (runner.epoch + 1) == self.unfreeze_epoch:
            self.modify_layers_grad(runner.model, self.unfreeze_layers, requires_grad=True)
            self.log_model_grad(runner.model, self.log_grad)
            print_log(
                f"Unfreeze {self.unfreeze_layers} at epoch {runner.epoch + 1}", logger="current"
            )
