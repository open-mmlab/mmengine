import os.path as osp
from unittest import TestCase

import torch

# from mmengine.config.auto_call_config import Config, LazyAttr
from mmengine.config.lazy import LazyCall
from mmengine.config.lazy_config import Config


class TestConfig(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = osp.join(
            osp.dirname(__file__), '..', 'data', 'config', 'auto_lazy_config')
        super().setUpClass()

    def test_lazy_attr(self):
        cfg_path = osp.join(self.data_dir, 'lazy_attr.py')
        cfg = Config.fromfile(cfg_path)
        result = cfg.result
        assert isinstance(result, LazyCall)
        assert isinstance(result.build(), torch.Tensor)

    def test_lazy_import_module(self):
        cfg_path = osp.join(self.data_dir, 'lazy_import_module.py')
