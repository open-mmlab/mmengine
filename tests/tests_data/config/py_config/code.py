from mmcv import Config  # isort:skip

cfg = Config.fromfile('tests/tests_data/config/py_config/simple_config.py')
item5 = cfg.item1[0] + cfg.item2.a
