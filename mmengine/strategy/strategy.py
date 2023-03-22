# Copyright (c) OpenMMLab. All rights reserved.
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

from mmengine.logging import MMLogger
from mmengine.optim import OptimWrapper, OptimWrapperDict, _ParamScheduler
from .utils import (ConfigType, ModelType, OptimType, SchedulerType,
                    _ConfigType, _ModelType, _OptimType, copy_to_dict)


class Mode(Enum):
    UNSET = 0  # Nothing will be setup
    VAL = 10  # Only setup model. Same for TEST
    TEST = 11
    TRAIN = 20  # Setup model, optimizer, param_scheduler

    @property
    def priority(self):
        return self.value // 10

    def __eq__(self, other):
        if self.__class__ is not other.__class__:
            raise NotImplementedError()
        return self.priority == other.priority

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            raise NotImplementedError()
        return self.priority < other.priority

    def __le__(self, other):
        if self.__class__ is not other.__class__:
            raise NotImplementedError()
        return self.priority <= other.priority

    def __ge__(self, other):
        if self.__class__ is not other.__class__:
            raise NotImplementedError()
        return self.priority >= other.priority


class Strategy(ABC):

    # class attributes, each new model_wrapper or optim_wrapper in MMEngine
    # should be declared here
    builtin_model_wrappers: tuple = ('MMDistributedDataParallel',
                                     'MMSeparateDistributedDataParallel',
                                     'MMFullyShardedDataParallel')
    builtin_optim_wrappers: tuple = ('OptimWraper', 'OptimWrapperDict',
                                     'AmpOptimWrapper', 'ApexOptimWrapper')
    builtin_optim_constructors: tuple = ('DefaultOptimWrapperConstructor', )

    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or MMLogger.get_current_instance()
        self.cfg: Dict = dict()
        self.model_cfg: Optional[Dict] = None
        self.optim_cfg: Optional[Dict] = None
        self.scheduler_cfg: Union[List[Dict], Dict, None] = None
        self.model: Optional[ModelType] = None
        self.optim: Optional[OptimType] = None
        self.schedulers: Optional[SchedulerType] = None
        self.mode: Mode = Mode.UNSET

    @abstractmethod
    def setup(
            self,
            model: Union[ModelType, ConfigType],
            optim: Union[OptimType, ConfigType, None] = None,
            scheduler: Union[SchedulerType, ConfigType, None] = None,
            *,
            mode: Mode = Mode.TRAIN,
            cfg: Optional[ConfigType] = None,
            # Below are for compatibility
            max_epochs: Optional[int] = None,
            max_iters: Optional[int] = None,
            auto_scale_lr: Optional[Dict] = None) -> Tuple:
        pass

    @abstractmethod
    def setup_distributed(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def save_checkpoint(self,
                        out_dir: str,
                        name: str,
                        meta: Dict = None,
                        save_optimizer: bool = True,
                        save_param_scheduler: bool = True,
                        *,
                        file_client_args: Optional[Dict] = None,
                        backend_args: Optional[Dict] = None,
                        callback: Optional[Callable] = None) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self,
                        load_dir: str,
                        name: Optional[str] = None,
                        load_optimizer: bool = False,
                        load_param_scheduler: bool = False,
                        *,
                        strict: bool = False,
                        map_location: Union[str, Callable] = 'cpu',
                        callback: Optional[Callable] = None) -> Optional[Dict]:
        pass

    def _store_config_or_instance(self,
                                  model: Union[ModelType, ConfigType] = None,
                                  optim: Union[OptimType, ConfigType,
                                               None] = None,
                                  scheduler: Union[SchedulerType, ConfigType,
                                                   None] = None,
                                  *,
                                  cfg: Optional[ConfigType] = None) -> None:
        # make a copy of full cfg for further checks
        if cfg is None:
            self.cfg = dict()
        else:
            self.cfg = copy_to_dict(cfg)

        # save model as a config or instance
        if model is None:
            pass
        elif isinstance(model, _ModelType):
            self.model = model
        elif isinstance(model, _ConfigType):
            self.model_cfg = copy_to_dict(model)
        else:
            raise TypeError(
                f'valid model types are {ModelType} or {ConfigType}, '
                f'but got {type(model)}')

        # save optim as a config or instance
        # Since there are 3 levels of instances (Optimizer, OptimWrapper,
        # OptimWrapperDict), the logic here is slightly complex
        if optim is None:
            pass
        elif isinstance(optim, _OptimType):
            self.optim = optim
        elif isinstance(optim, _ConfigType):
            if 'constructor' in optim:
                # When constructor is given, the whole OptimWrapper should be
                # built by the given constructor. No check is available
                self.optim_cfg = copy_to_dict(optim)
            elif optim.get('optimizer') is None:
                # Neither constructor nor optimizer given, it must be a
                # partially built OptimWrapperDict, with all values being
                # instances of OptimWrapper
                for name, optim_wrapper in optim.items():
                    if not isinstance(optim_wrapper, OptimWrapper):
                        raise ValueError(
                            'each item mush be an optimizer object when '
                            '"optimizer" and "constructor" are not in '
                            f'optim_wrapper, but got {name}={optim_wrapper}')
                self.optim = OptimWrapperDict(**optim)
            else:
                # No constructor, but optimizer given. Basic use case.
                # Note that it might contain Optimizer instance
                self.optim_cfg = copy_to_dict(optim)
        else:
            raise TypeError(
                f'valid optim types are {OptimType} or {ConfigType}, '
                f'but got {type(optim)}')

        # save scheduler as a config or instance
        # Schedulers have list & dict form. Should check carefully
        if scheduler is None:
            pass
        elif isinstance(scheduler, _ParamScheduler):
            self.schedulers = [scheduler]
        elif isinstance(scheduler, (list, tuple)):
            if all(isinstance(s, _ParamScheduler) for s in scheduler):
                # All instances, no need to build
                self.schedulers = scheduler
            else:
                # At least one is not instance, should build
                self.scheduler_cfg = [copy_to_dict(s) for s in scheduler]
        elif isinstance(scheduler, _ConfigType):
            # There are 2 cases:
            #   1) "type" given: just a single param_scheduler
            #   2) "type" not given: keys correspond to OptimWrapperDict, each
            #      item could be a _ParamScheduler, List[_ParamScheduler],
            #      Dict, List[Dict]. We cannot further analyze this case
            #      unless OptimWrapper has been built.
            self.scheduler_cfg = scheduler
        else:
            raise TypeError(
                f'valid scheduler types are {SchedulerType} or {ConfigType}, '
                f'but got {type(scheduler)}')
