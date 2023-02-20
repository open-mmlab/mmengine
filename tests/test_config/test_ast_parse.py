import ast
import os.path as osp
from unittest import TestCase

import torch
import torch.amp as amp
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mmengine.config.lazy import LazyCall, LazyModule
from mmengine.config.lazy_ast import Transform, import_to_lazymodule


class TestConfig(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = osp.join(
            osp.dirname(__file__), '..', 'data', 'config', 'auto_lazy_config')
        super().setUpClass()

    def test_lazy(self):
        cfg_path = osp.join(self.data_dir, 'lazy_import_module.py')
        with open(cfg_path) as f:
            codestr = f.read()
        codeobj = ast.parse(codestr)
        global_dict = {'LazyModule': LazyModule}
        codeobj = Transform(global_dict).visit(codeobj)
        codeobj = import_to_lazymodule(codeobj)
        codeobj = ast.fix_missing_locations(codeobj)

        exec(compile(codeobj, cfg_path, mode='exec'), global_dict, global_dict)
        self.assertIsInstance(global_dict['torch'], LazyModule)
        self.assertEqual(global_dict['torch'].build(), torch)
        self.assertEqual(getattr(global_dict['torch'].build(), 'amp'), amp)
        self.assertEqual(global_dict['nn'].build(), torch.nn)
        self.assertEqual(global_dict['Variable'].build(),
                         torch.autograd.Variable)
        self.assertEqual(global_dict['DataLoader'].build(), DataLoader)

        self.assertIsInstance(global_dict['tensor'], LazyCall)
        self.assertIsInstance(global_dict['tensor_mean'], LazyCall)
        self.assertIsInstance(global_dict['tensor_sum'], LazyCall)

        self.assertIsInstance(global_dict['tensor'].build(), torch.Tensor)
        self.assertIsInstance(global_dict['tensor_mean'].build(), torch.Tensor)
        self.assertIsInstance(global_dict['tensor_sum'].build(), torch.Tensor)

    # def test_lazy_call(self):
