# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Optional, Sequence

from mmengine.logging import print_log
from mmengine.model import BaseModel, is_model_wrapper
from mmengine.registry import HOOKS
from .hook import DATA_BATCH, Hook


@HOOKS.register_module()
class FreezeHook(Hook):
    """FreezeHook is used to freeze/unfreeze network layers when training to a
    specified epoch/iter.

    Args:
        freeze_layers (str): Layers of the network that will be freeze. \
            Match by regular expression.
        freeze_iter (int): The iter number to start freezing layers.
        freeze_epoch (int): The epoch number to start freezing layers.
        unfreeze_layers (str): Layers of the network that will be unfreeze. \
            Match by regular expression.
        unfreeze_iter (int): The iter number to start unfreezing layers.
        unfreeze_epoch (int): The epoch number to start unfreezing layers.
        verbose  (bool): Whether to log the requires_grad of each layer.

    Notes:
        The GPU memory usage shown in the "nvidia-smi" command does not change
        when you freeze model layers.
        https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/4

    Examples:
        >>> # The simplest FreezeHook config.
        >>> freeze_hook_cfg = dict(
                type="FreezeHook",
                freeze_layers="backbone.*|neck.*",
                freeze_epoch=0
            )
    """

    def __init__(
        self,
        freeze_layers: str,
        freeze_iter: Optional[int] = None,
        freeze_epoch: Optional[int] = None,
        unfreeze_layers: Optional[str] = None,
        unfreeze_iter: Optional[int] = None,
        unfreeze_epoch: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        # check arguments type
        if not isinstance(freeze_layers, str):
            raise TypeError('`freeze_layers` must be a str')
        if not isinstance(freeze_iter, int) and freeze_iter is not None:
            raise TypeError('`freeze_iter` must be a int or None')
        if not isinstance(freeze_epoch, int) and freeze_epoch is not None:
            raise TypeError('`freeze_epoch` must be a int or None')
        if not isinstance(unfreeze_layers,
                          str) and unfreeze_layers is not None:
            raise TypeError('`unfreeze_layers` must be a str or None')
        if not isinstance(unfreeze_iter, int) and unfreeze_iter is not None:
            raise TypeError('`unfreeze_iter` must be a int or None')
        if not isinstance(unfreeze_epoch, int) and unfreeze_epoch is not None:
            raise TypeError('`unfreeze_epoch` must be a int or None')
        if not isinstance(verbose, bool):
            raise TypeError('`verbose`  must be a boolean')
        # check arguments value
        if freeze_iter is None and freeze_epoch is None:
            raise ValueError(
                '`freeze_iter` and `freeze_epoch` should not be both None.')
        if freeze_iter is not None and freeze_epoch is not None:
            raise ValueError(
                '`freeze_iter` and `freeze_epoch` should not be both set.')
        if freeze_iter is not None and freeze_iter < 0:
            raise ValueError(
                '`freeze_iter` must be greater than or equal to 0')
        if freeze_epoch is not None and freeze_epoch < 0:
            raise ValueError(
                '`freeze_epoch` must be greater than or equal to 0')
        if unfreeze_layers is not None:
            if unfreeze_iter is None and unfreeze_epoch is None:
                raise ValueError('`unfreeze_iter` and `unfreeze_epoch` \
                                 should not be both None.')
            if unfreeze_iter is not None and unfreeze_epoch is not None:
                raise ValueError('`unfreeze_iter` and `unfreeze_epoch` \
                        should not be both set.')
        if unfreeze_iter is not None:
            if unfreeze_iter < 0:
                raise ValueError(
                    '`unfreeze_iter` must be greater than or equal to 0')
            if freeze_iter is None:
                raise ValueError('`unfreeze_iter` and `freeze_iter` \
                        should be set at the same time.')
            if unfreeze_iter < freeze_iter:
                raise ValueError('`unfreeze_iter` must be greater than \
                        or equal to `freeze_iter`')
        if unfreeze_epoch is not None:
            if unfreeze_epoch < 0:
                raise ValueError(
                    '`unfreeze_epoch` must be greater than or equal to 0')
            if freeze_epoch is None:
                raise ValueError('`unfreeze_epoch` and `freeze_epoch` \
                        should be set at the same time.')
            if unfreeze_epoch < freeze_epoch:
                raise ValueError('`unfreeze_epoch` must be greater than \
                        or equal to `freeze_epoch`')

        self.freeze_layers = freeze_layers
        self.freeze_iter = freeze_iter
        self.freeze_epoch = freeze_epoch
        self.unfreeze_layers = unfreeze_layers
        self.unfreeze_iter = unfreeze_iter
        self.unfreeze_epoch = unfreeze_epoch
        self.verbose = verbose
        self.freeze_layer_names: list = []
        self.unfreeze_layer_names: list = []

    def _modify_layers_grad(self, model: BaseModel, layers: Sequence[str],
                            requires_grad: bool):
        """Modify the ``requires_grad`` of the specified layers.

        Args:
            model (BaseModel): A BaseModel of mmengine.
            layers (Sequence[str]): Network layers to be modified.
            requires_grad (bool): Whether to enable gradient.
        """
        if is_model_wrapper(model):
            model = model.module
        for k, v in model.named_parameters():
            if k in layers:
                v.requires_grad = requires_grad

    def _log_model_grad(self, model: BaseModel):
        """Print `requires_grad` for all network layers.

        Args:
            model (BaseModel): A BaseModel of mmengine.
        """
        for k, v in model.named_parameters():
            print_log(
                f'{k} requires_grad: {v.requires_grad}', logger='current')

    def _freeze(self,
                runner,
                idx: int,
                freeze_idx: int,
                unfreeze_idx: Optional[int] = None) -> None:
        """The main function of FreezeHook.

        Args:
            runner (Runner): The runner of the training process.
            idx (int): The index in the train loop.
            freeze_idx (int): The index to start freezing layers.
            unfreeze_idx (int): The index to start unfreezing layers.
        """
        if idx in (freeze_idx, unfreeze_idx):
            model = runner.model
            if is_model_wrapper(model):
                model = model.module

            if idx == freeze_idx:
                print_log('Start freezing layers.', logger='current')
                self._modify_layers_grad(model, self.freeze_layer_names, False)
                # if you want to release GPU memory cache:
                # import torch; torch.cuda.empty_cache()

            if idx == unfreeze_idx:
                print_log('Start unfreezing layers.', logger='current')
                self._modify_layers_grad(model, self.unfreeze_layer_names,
                                         True)

            if self.verbose:
                self._log_model_grad(model)

    def before_train(self, runner) -> None:
        """Collect network layers that will be freeze or unfreeze.

        Args:
            runner (Runner): The runner of the training process.
        """
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        layer_names = ''
        for k, _ in model.named_parameters():
            layer_names += f'{k}\n'

        freeze_names = re.findall(self.freeze_layers, layer_names)
        self.freeze_layer_names = freeze_names

        if self.unfreeze_layers is not None:
            unfreeze_names = re.findall(self.unfreeze_layers, layer_names)
            self.unfreeze_layer_names = unfreeze_names

    def before_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch: DATA_BATCH = None,
        outputs: Optional[dict] = None,
    ) -> None:
        """Update `requires_grad` before the start of `freeze_iter`

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (Sequence[dict], optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if self.freeze_iter is not None:
            self._freeze(runner, runner.iter, self.freeze_iter,
                         self.unfreeze_iter)

    def before_train_epoch(self, runner) -> None:
        """Update `requires_grad` before the start of `freeze_epoch`

        Args:
            runner (Runner): The runner of the training process.
        """
        if self.freeze_epoch is not None:
            self._freeze(runner, runner.epoch, self.freeze_epoch,
                         self.unfreeze_epoch)
