# Copyright (c) OpenMMLab. All rights reserved.
from .config import (Config, ConfigDict, ConfigList, ConfigSet, ConfigTuple,
                     DictAction)
from .new_config import read_base

__all__ = [
    'Config', 'ConfigDict', 'DictAction', 'read_base', 'ConfigList',
    'ConfigSet', 'ConfigTuple'
]
