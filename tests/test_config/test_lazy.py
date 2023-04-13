# Copyright (c) OpenMMLab. All rights reserved.
import ast
import os
import os.path as osp
from unittest import TestCase

import torch
import torch.amp as amp
import torch.functional as functional

import mmengine
import mmengine.model
from mmengine.config.lazy import LazyAttr, LazyObject
from mmengine.config.utils import Transform, _gather_abs_import_lazyobj
from mmengine.dataset import BaseDataset
from mmengine.model import BaseModel


class TestTransform(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = osp.join(  # type: ignore
            osp.dirname(__file__), '..', 'data', 'config',
            'lazy_module_config')
        super().setUpClass()

    def test_lazy_module(self):
        cfg_path = osp.join(self.data_dir, 'test_ast_transform.py')
        with open(cfg_path) as f:
            codestr = f.read()
        codeobj = ast.parse(codestr)
        global_dict = {'LazyObject': LazyObject}
        codeobj = Transform(global_dict).visit(codeobj)
        codeobj = _gather_abs_import_lazyobj(codeobj)
        codeobj = ast.fix_missing_locations(codeobj)

        exec(compile(codeobj, cfg_path, mode='exec'), global_dict, global_dict)
        # 1. absolute import
        # 1.1 import module as LazyObject
        lazy_torch = global_dict['torch']
        self.assertIsInstance(lazy_torch, LazyObject)

        # 1.2 getattr as LazyAttr
        self.assertIsInstance(lazy_torch.amp, LazyAttr)
        self.assertIsInstance(lazy_torch.functional, LazyAttr)

        # 1.3 Build module from LazyObject. amp and functional can be accessed
        imported_torch = lazy_torch.build()
        self.assertIs(imported_torch.amp, amp)
        self.assertIs(imported_torch.functional, functional)

        # 1.4 Build module from LazyAttr
        imported_amp = lazy_torch.amp.build()
        imported_functional = lazy_torch.functional.build()
        self.assertIs(imported_amp, amp)
        self.assertIs(imported_functional, functional)

        # 1.5 import ... as, and build module from LazyObject
        lazy_nn = global_dict['nn']
        self.assertIsInstance(lazy_nn, LazyObject)
        self.assertIs(lazy_nn.build(), torch.nn)
        self.assertIsInstance(lazy_nn.Conv2d, LazyAttr)
        self.assertIs(lazy_nn.Conv2d.build(), torch.nn.Conv2d)

        # 1.6 import built in module
        imported_os = global_dict['os']
        self.assertIs(imported_os, os)

        # 2. Relative import
        # 2.1 from ... import ...
        lazy_BaseModel = global_dict['BaseModel']
        self.assertIsInstance(lazy_BaseModel, LazyObject)
        self.assertIs(lazy_BaseModel.build(), BaseModel)

        # 2.2 from ... import ... as ...
        lazy_Dataset = global_dict['Dataset']
        self.assertIsInstance(lazy_Dataset, LazyObject)
        self.assertIs(lazy_Dataset.build(), BaseDataset)


class TestLazyObject(TestCase):

    def test_init(self):
        LazyObject('mmengine')
        LazyObject('mmengine.model')
        LazyObject('mmengine.model', 'BaseModule')

        # module must be str
        with self.assertRaises(TypeError):
            LazyObject(1)

        # imported must be a sequence of string or None
        with self.assertRaises(TypeError):
            LazyObject('mmengine', ['error_type'])

    def test_build(self):
        lazy_mmengine = LazyObject('mmengine')
        self.assertIs(lazy_mmengine.build(), mmengine)
        lazy_mmengine_models = LazyObject('mmengine')
        self.assertIs(lazy_mmengine_models.build(),
                      __import__('mmengine.model'))
        lazy_base_model = LazyObject('mmengine.model', 'BaseModel')
        self.assertIs(lazy_base_model.build(), BaseModel)

        mmengine_module = LazyObject(['mmengine.model', 'mmengine'])
        self.assertIs(mmengine_module.build(), mmengine)


class TestLazyAttr(TestCase):
    # Since LazyAttr should only be built from LazyObect, we only test
    # the build method here.
    def test_build(self):
        lazy_mmengine = LazyObject('mmengine')
        mmengine_model = lazy_mmengine.model
        self.assertIs(mmengine_model.build(), mmengine.model)
        lazy_base_model = lazy_mmengine.model.base_model
        self.assertIs(lazy_base_model.build(), mmengine.model.base_model)
