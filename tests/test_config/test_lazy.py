# Copyright (c) OpenMMLab. All rights reserved.
import copy
from importlib import import_module
from unittest import TestCase

import mmengine
from mmengine.config.lazy import LazyObject
from mmengine.fileio import LocalBackend


class TestLazyObject(TestCase):

    def test_init(self):
        LazyObject('mmengine')
        LazyObject('mmengine.fileio')
        LazyObject('mmengine.fileio', 'LocalBackend')

    def test_build(self):
        lazy_mmengine = LazyObject('mmengine')
        self.assertIs(lazy_mmengine.build(), mmengine)

        lazy_mmengine_fileio = LazyObject('mmengine.fileio')
        self.assertIs(lazy_mmengine_fileio.build(),
                      import_module('mmengine.fileio'))

        lazy_local_backend = LazyObject('LocalBackend',
                                        LazyObject('mmengine.fileio'))
        self.assertIs(lazy_local_backend.build(), LocalBackend)

        copied = copy.deepcopy(lazy_local_backend)
        self.assertDictEqual(copied.__dict__, lazy_local_backend.__dict__)

        with self.assertRaises(TypeError):
            lazy_mmengine()

        with self.assertRaises(ImportError):
            LazyObject('unknown').build()

        lazy_mmengine = LazyObject('mmengine')
        local_backend = lazy_mmengine.fileio.LocalBackend
        self.assertIs(local_backend.build(), LocalBackend)

        copied = copy.deepcopy(local_backend)
        self.assertDictEqual(copied.__dict__, local_backend.__dict__)

        with self.assertRaises(TypeError):
            local_backend()

        with self.assertRaisesRegex(ImportError, 'Failed to import'):
            local_backend.unknown.build()
