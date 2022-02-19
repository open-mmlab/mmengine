# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import Config  # isort:skip

cfg = Config.fromfile('tests/data/config/py_config/simple_config.py')
item5 = cfg.item1[0] + cfg.item2.a
