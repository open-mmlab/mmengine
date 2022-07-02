# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from unittest import TestCase

from parameterized import parameterized

from mmengine.fileio import LmdbBackend


class TestLmdbBackend(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
        cls.lmdb_path = cls.test_data_dir / 'demo.lmdb'

    @parameterized.expand([[Path], [str]])
    def test_get_bytes(self, path_type):
        backend = LmdbBackend(path_type(self.lmdb_path))
        img_bytes = backend.get_bytes('baboon')
        try:
            import mmcv
        except ImportError:
            pass
        else:
            img = mmcv.imfrombytes(img_bytes)
            self.assertEqual(img.shape, (120, 125, 3))

    def test_get_text(self):
        backend = LmdbBackend(self.lmdb_path)
        with self.assertRaises(NotImplementedError):
            backend.get_text('filepath')
