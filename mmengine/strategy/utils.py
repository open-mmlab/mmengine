# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Set, Union

import torch.nn as nn

from mmengine.config import Config, ConfigDict
from mmengine.optim import OptimWrapper, OptimWrapperDict, _ParamScheduler

# Type Aliases, used for type hints
ConfigType = Union[Config, ConfigDict, Dict]
ModelType = Union[nn.Module]
OptimType = Union[OptimWrapper, OptimWrapperDict]
SchedulerType = Union[_ParamScheduler, List[_ParamScheduler],
                      Dict[str, List[_ParamScheduler]]]

# Type Aliases, used for runtime type checks
_ConfigType = (Config, ConfigDict, dict)
# include BaseModel and all model wrapper types
_ModelType = nn.Module
_OptimType = (OptimWrapper, OptimWrapperDict)

NOTSET = object()


def copy_to_dict(cfg: ConfigType) -> Dict:
    if isinstance(cfg, (Config, ConfigDict)):
        return cfg.to_dict()
    else:
        return deepcopy(cfg)


def dfs_dict(cfg: Any,
             condition: Callable,
             memo: Optional[Set] = None) -> bool:
    if cfg is None:
        return False
    if memo is None:
        memo = set()
    # be aware of self-reference dicts, which leads to infinite loop
    if id(cfg) in memo:
        return False
    memo.add(id(cfg))
    if isinstance(cfg, dict):
        return any(dfs_dict(c, condition, memo) for _, c in cfg.items())
    else:
        return condition(cfg)


def inconsistent_keys(x: Dict, y: Dict):
    common_keys = set(x.keys()) & set(y.keys())
    return {k for k in common_keys if x[k] != y[k]}
