# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from typing import Optional, Union

import torch
import torch.nn as nn

# a circular import will be caused by
# from mmengine.model.wrappers import is_model_wrapper
import mmengine
from mmengine.registry import OPTIM_WRAPPERS
from .optimizer_wrapper import OptimWrapper

try:
    import apex.amp as apex_amp
except ImportError:
    apex_amp = None


@OPTIM_WRAPPERS.register_module()
class ApexOptimWrapper(OptimWrapper):
    """A subclass of :class:`OptimWrapper` that supports automatic mixed
    precision training based on apex.amp.

    ``ApexOptimWrapper`` provides a unified interface with
    ``OptimWrapper``, so ``ApexOptimWrapper`` can be used in the same way
    as ``OptimWrapper``.

    Warning:
        ``ApexOptimWrapper`` requires `nvidia apex <https://github.com/NVIDIA/apex>`_

    Args:
        opt_level (str, default="O1"): Pure or mixed precision
            optimization level. Accepted values are "O0", "O1", "O2",
            and "O3".
        loss_scale (float or str, default=None): If passed as
            a string, must be a string representing a number,
            e.g., "128.0", or the string "dynamic".
        enabled (bool, default=True): If False, renders all Amp calls no-ops,
            so your script should run as if Amp were not present.
        cast_model_type (torch.dtype, default=None): Model's parameters and
            buffers to the desired type.
        patch_torch_functions (bool, default=None): Patch all Torch functions
            and Tensor methods to perform Tensor Core-friendly ops like GEMMs
            and convolutions in FP16,
            and any ops that benefit from FP32 precision in FP32.
        keep_batchnorm_fp32 (bool or str, default=None): To enhance precision
            and enable cudnn batchnorm (which improves performance),
            it's often beneficial to keep batchnorm weights in FP32
            even if the rest of the model is FP16.
            If passed as a string, must be the string "True" or "False".
        master_weights (bool, default=None): Maintain FP32 master weights to
            accompany any FP16 model weights. FP32 master weights are stepped
            by the optimizer to enhance precision and capture small gradients.
        cast_model_outputs (torch.dtype, default=None): Option to ensure that
            the outputs of your model(s) are always cast to a particular type
            regardless of ``opt_level``.
        num_losses (int, default=1): Option to tell Amp in advance how many
            losses/backward passes you plan to use.
        verbosity (int, default=1): Set to 0 to suppress Amp-related output.
        min_loss_scale (float, default=None): Sets a floor for the loss scale
            values that can be chosen by dynamic loss scaling.
            The default value of None means that no floor is imposed.
            If dynamic loss scaling is not used, `min_loss_scale` is ignored.
        max_loss_scale (float, default=2.**24): Sets a ceiling for the
            loss scale values that can be chosen by dynamic loss scaling.
            If dynamic loss scaling is not used, `max_loss_scale` is ignored.
        **kwargs: Keyword arguments passed to OptimWrapper.

    Note:
        If you use ``IterBasedRunner`` and enable gradient accumulation,
        the original `max_iters` should be multiplied by
        ``accumulative_counts``.
    """  # noqa: E501

    def __init__(self,
                 opt_level: str = 'O1',
                 loss_scale: Union[float, str] = 'dynamic',
                 enabled: Optional[bool] = True,
                 cast_model_type: Optional[torch.dtype] = None,
                 patch_torch_functions: Optional[bool] = None,
                 keep_batchnorm_fp32: Optional[Union[bool, str]] = None,
                 master_weights: Optional[bool] = None,
                 cast_model_outputs: Optional[torch.dtype] = None,
                 num_losses: Optional[int] = 1,
                 verbosity: Optional[int] = 1,
                 min_loss_scale: Optional[float] = None,
                 max_loss_scale: Optional[float] = 2.**24,
                 **kwargs):
        assert apex_amp is not None, \
            'Apex is not installed. Please check ' \
            'https://github.com/NVIDIA/apex#linux.'
        super().__init__(**kwargs)
        self.opt_level = opt_level
        self.loss_scale = loss_scale
        self.enabled = enabled
        self.cast_model_type = cast_model_type
        self.patch_torch_functions = patch_torch_functions
        self.keep_batchnorm_fp32 = keep_batchnorm_fp32
        self.master_weights = master_weights
        self.cast_model_outputs = cast_model_outputs
        self.num_losses = num_losses
        self.verbosity = verbosity
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale
        self._apex_amp_state_dict = None

    def backward(self, loss: torch.Tensor, **kwargs) -> None:
        """Perform gradient back propagation with :attr:`loss_scaler`.

        Args:
            loss (torch.Tensor): The loss of current iteration.
            kwargs: Keyword arguments passed to :meth:`torch.Tensor.backward`
        """
        with apex_amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
        self._inner_count += 1

    def state_dict(self) -> dict:
        """Get the state dictionary of :attr:`optimizer` and
        :attr:`apex_amp`.

        Based on the state dictionary of the optimizer, the returned state
        dictionary will add a key named "apex_amp".

        Returns:
            dict: The merged state dict of :attr:`apex_amp` and
            :attr:`optimizer`.
        """
        state_dict = self.optimizer.state_dict()
        state_dict['apex_amp'] = apex_amp.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """Load and parse the state dictionary of :attr:`optimizer` and
        :attr:`apex_amp`.

        If state_dict contains "apex_amp", the :attr:`apex_amp` will
        load the corresponding keys. Otherwise, only the :attr:`optimizer`
        will load the state dictionary.

        Note:
            :meth:`load_state_dict` shuold be called after
            `apex_amp.initialize` is called.
        Args:
            state_dict (dict): The state dict of :attr:`optimizer` and
                :attr:`apex_amp`
        """
        if 'apex_amp' in state_dict:
            if hasattr(self.optimizer, '_amp_stash'):
                apex_amp.load_state_dict(state_dict.pop('apex_amp'))
            else:
                self._apex_amp_state_dict = state_dict.pop('apex_amp')
        self.optimizer.load_state_dict(state_dict)

    @contextmanager
    def optim_context(self, model: nn.Module):
        """Enables the context for mixed precision training, and enables the
        context for disabling gradient synchronization during gradient
        accumulation context.

        Args:
            model (nn.Module): The training model.
        """
        with super().optim_context(model):
            # when a given optimizer be passed through apex_amp.initialize,
            # the "_amp_stash" property will be added
            if not hasattr(self.optimizer, '_amp_stash'):
                if mmengine.model.wrappers.is_model_wrapper(model):
                    model = model.module
                model, self.optimizer = apex_amp.initialize(
                    model,
                    self.optimizer,
                    opt_level=self.opt_level,
                    loss_scale=self.loss_scale,
                    enabled=self.enabled,
                    cast_model_type=self.cast_model_type,
                    patch_torch_functions=self.patch_torch_functions,
                    keep_batchnorm_fp32=self.keep_batchnorm_fp32,
                    master_weights=self.master_weights,
                    cast_model_outputs=self.cast_model_outputs,
                    num_losses=self.num_losses,
                    verbosity=self.verbosity,
                    min_loss_scale=self.min_loss_scale,
                    max_loss_scale=self.max_loss_scale)
                # loading apex_amp state_dict after initialization of apex_amp
                if self._apex_amp_state_dict:
                    apex_amp.load_state_dict(self._apex_amp_state_dict)
                    self._apex_amp_state_dict = None
            yield
