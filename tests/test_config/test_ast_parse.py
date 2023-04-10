# Copyright (c) OpenMMLab. All rights reserved.
import ast
import os
import os.path as osp
from unittest import TestCase

import torch
import torch.amp as amp
import torch.functional as functional

from mmengine.config.lazy import LazyAttr, LazyModule
from mmengine.config.lazy_ast import Transform, _gather_abs_import_lazymodule
from mmengine.dataset import BaseDataset
from mmengine.model import BaseModel


class TestTransform(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = osp.join(
            osp.dirname(__file__), '..', 'data', 'config',
            'lazy_module_config')
        super().setUpClass()

    def test_lazy_module(self):
        cfg_path = osp.join(self.data_dir, 'test_ast_transform.py')
        with open(cfg_path) as f:
            codestr = f.read()
        codeobj = ast.parse(codestr)
        global_dict = {'LazyModule': LazyModule}
        codeobj = Transform(global_dict).visit(codeobj)
        codeobj = _gather_abs_import_lazymodule(codeobj)
        codeobj = ast.fix_missing_locations(codeobj)

        exec(compile(codeobj, cfg_path, mode='exec'), global_dict, global_dict)
        # 1. absolute import
        # 1.1 import module as LazyModule
        lazy_torch = global_dict['torch']
        self.assertIsInstance(lazy_torch, LazyModule)

        # 1.2 getattr as LazyAttr
        self.assertIsInstance(lazy_torch.amp, LazyAttr)
        self.assertIsInstance(lazy_torch.functional, LazyAttr)

        # 1.3 Build module from LazyModule. amp and functional can be accessed
        imported_torch = lazy_torch.build()
        self.assertIs(imported_torch.amp, amp)
        self.assertIs(imported_torch.functional, functional)

        # 1.4 Build module from LazyAttr
        imported_amp = lazy_torch.amp.build()
        imported_functional = lazy_torch.functional.build()
        self.assertIs(imported_amp, amp)
        self.assertIs(imported_functional, functional)

        # 1.5 import ... as, and build module from LazyModule
        lazy_nn = global_dict['nn']
        self.assertIsInstance(lazy_nn, LazyModule)
        self.assertIs(lazy_nn.build(), torch.nn)
        self.assertIsInstance(lazy_nn.Conv2d, LazyAttr)
        self.assertIs(lazy_nn.Conv2d.build(), torch.nn.Conv2d)

        # 1.6 import built in module
        imported_os = global_dict['os']
        self.assertIs(imported_os, os)

        # 2. Relative import
        # 2.1 from ... import ...
        lazy_BaseModel = global_dict['BaseModel']
        self.assertIsInstance(lazy_BaseModel, LazyModule)
        self.assertIs(lazy_BaseModel.build(), BaseModel)

        # 2.2 from ... import ... as ...
        lazy_Dataset = global_dict['Dataset']
        self.assertIsInstance(lazy_Dataset, LazyModule)
        self.assertIs(lazy_Dataset.build(), BaseDataset)
