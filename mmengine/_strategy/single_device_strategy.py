from .base_strategy import BaseStrategy
import torch.nn as nn
from typing import Union, Dict, Optional, List

from mmengine.registry import STRATEGIES
from mmengine.device import get_device
from mmengine.model import revert_sync_batchnorm
from mmengine.optim import OptimWrapper, _ParamScheduler


@STRATEGIES.register_module()
class SingleDeviceStrategy(BaseStrategy):
    """Strategy for single device training."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare(self,
                model: Union[nn.Module, dict],
                *,
                optim_wrapper: Optional[Union[OptimWrapper, dict]] = None,
                param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
                compile_target: Union[str, bool] = False,
                checkpoint: Optional[dict] = None,
                num_batches_per_epoch: Optional[int] = None,
                max_epochs: Optional[int] = None,
                max_iters: Optional[int] = None,
                cur_iter: Optional[int] = None,
                **kwargs):
        """Prepare model and some components.
        
        Args:
            model (:obj:`torch.nn.Module` or dict): The model to be run. It can be
                a dict used for build a model.

        Kwargs:
            optim_wrapper (OptimWrapper or dict, optional):
                Computing gradient of model parameters. If specified,
                :attr:`train_dataloader` should also be specified. If automatic
                mixed precision or gradient accmulation
                training is required. The type of ``optim_wrapper`` should be
                AmpOptimizerWrapper. See :meth:`build_optim_wrapper` for
                examples. Defaults to None.
            param_scheduler (_ParamScheduler or dict or list, optional):
                Parameter scheduler for updating optimizer parameters. If
                specified, :attr:`optimizer` should also be specified.
                Defaults to None.
                See :meth:`build_param_scheduler` for examples.
            compile_target (str, optional): The method of model to be compiled.
                Defaults to 'forward'.
            checkpoint (dict, optional): Checkpoint to load strategy state.
                Defaults to None.
            num_batches_per_epoch (int, optional): Number of batches per epoch.
                Defaults to None.
            max_epochs (int, optional): Number of epochs. Defaults to None.
            max_iters (int, optional): Number of iterations. Defaults to None.
            cur_iter (int, optional): Current iteration. Defaults to None.
        """
        result = []
        model = self.build_model(model)
        model = self._init_model_weights(model)
        model = self.wrap_model(model)
        self.model = self.compile_model(model, target=compile_target)
        result.append(self.model)

        if optim_wrapper is not None:
            self.optim_wrapper = self.build_optim_wrapper(optim_wrapper)
            result.append(self.optim_wrapper)

        if param_scheduler is not None:
            _default_args = {}
            if num_batches_per_epoch is not None:
                _default_args['epoch_length'] = num_batches_per_epoch
            if max_epochs is not None:
                _default_args['max_epochs'] = max_epochs
            if max_iters is not None:
                _default_args['max_iters'] = max_iters

            self.param_schedulers = self.build_param_scheduler(param_scheduler, _default_args)
            result.append(self.param_schedulers)

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

        if optim_wrapper is not None:
            # Initiate inner count of `optim_wrapper`.
            self.optim_wrapper.initialize_count_status(self.model, cur_iter, max_iters)

        return result

    def wrap_model(self, model: nn.Module) -> nn.Module:
        self.logger.info(
            'Distributed training is not used, all SyncBatchNorm (SyncBN) '
            'layers in the model will be automatically reverted to '
            'BatchNormXd layers if they are used.')
        model = self.convert_model(model)
        current_device = get_device()
        return model.to(current_device)
