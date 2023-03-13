# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Union

import torch.nn as nn

from mmengine.config import Config, ConfigDict
from mmengine.optim import OptimWrapper, OptimWrapperDict, _ParamScheduler

# used for type hints
ConfigType = Union[Config, ConfigDict, Dict]
ModelType = Union[nn.Module]
OptimType = Union[OptimWrapper, OptimWrapperDict]
SchedulerType = Union[_ParamScheduler, List[_ParamScheduler],
                      Dict[str, List[_ParamScheduler]]]
# used for runtime type checks
_ConfigType = (Config, ConfigDict, dict)
_ModelType = nn.Module  # include BaseModel and all model wrapper types
_OptimType = (OptimWrapper, OptimWrapperDict)
_SchedulerType = (_ParamScheduler, list, dict)


def copy_or_to_dict(config: ConfigType) -> Dict:
    if isinstance(config, dict):
        return deepcopy(config)
    else:
        return config.to_dict()


class Strategy(ABC):

    # These are actually instance attributes. Place them here to improve user
    # experience with IDE
    cfg: Dict = dict()
    model_cfg: Optional[Dict] = None
    optim_cfg: Optional[Dict] = None
    scheduler_cfg: Optional[Dict] = None
    model: Optional[ModelType] = None
    optim: Optional[OptimType] = None
    scheduler: Optional[SchedulerType] = None

    # class attributes, each new model_wrapper or optim_wrapper in MMEngine
    # should be declared here
    builtin_model_wrappers: tuple = (
        'MMDistributedDataParallel', 'MMSeparateDistributedDataParallel',
        'MMFullyShardedDataParallel'
    )
    builtin_optim_wrappers: tuple = (
        'OptimWraper', 'AmpOptimWrapper', 'OptimWrapperDict',
        'ZeroRedundancyOptimizer'
    )

    def __init__(self):
        self._base_is_initialized = True
        self._config_is_stored = False

    @abstractmethod
    def setup(self,
              model: Union[ModelType, ConfigType],
              optim: Union[OptimType, ConfigType, None] = None,
              scheduler: Union[SchedulerType, ConfigType, None] = None,
              *,
              cfg: Optional[ConfigType] = None) -> None:
        pass

    @abstractmethod
    def setup_distributed(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs) -> None:
        pass

    @property
    def load_before_setup(self) -> bool:
        return True

    def _store_config_or_instance(
                self,
                model: Union[ModelType, ConfigType] = None,
                optim: Union[OptimType, ConfigType, None] = None,
                scheduler: Union[SchedulerType, ConfigType, None] = None,
                *,
                cfg: Optional[ConfigType] = None) -> None:
        # do not setup twice
        if self._config_is_stored:
            raise RuntimeError(
                'Strategy should not be setup twice. This is very likely an '
                'internal error of MMEngine, please contact maintainers for '
                'help or fix')

        # make a copy of full cfg for further checks
        if cfg is None:
            self.cfg = dict()
        else:
            self.cfg = copy_or_to_dict(cfg)

        # save model as a config or instance
        if model is None or isinstance(model, _ModelType):
            self.model = model
            self.model_cfg = copy_or_to_dict(self.cfg.get('model', {}))
        elif isinstance(model, _ConfigType):
            self.model_cfg = copy_or_to_dict(model)
        else:
            raise TypeError(
                f'valid model types are {ModelType} or {ConfigType}, '
                f'but got {type(model)}')

        # save optim as a config or instance
        if optim is None or isinstance(optim, _OptimType):
            self.optim = optim
            self.optim_cfg = copy_or_to_dict(self.cfg.get('optim_wrapper', {}))
        elif isinstance(optim, _ConfigType):
            self.optim_cfg = copy_or_to_dict(optim)
        else:
            raise TypeError(
                f'valid optim types are {OptimType} or {ConfigType}, '
                f'but got {type(optim)}')

        # save scheduler as a config or instance
        if scheduler is None or isinstance(scheduler, _SchedulerType):
            self.scheduler = scheduler
            scheduler_cfg = self.cfg.get('param_scheduler', {})
            self.scheduler_cfg = copy_or_to_dict(scheduler_cfg)
        elif isinstance(scheduler, _ConfigType):
            self.scheduler_cfg = copy_or_to_dict(scheduler)
        else:
            raise TypeError(
                f'valid scheduler types are {SchedulerType} or {ConfigType}, '
                f'but got {type(optim)}')

        self._config_is_stored = True
