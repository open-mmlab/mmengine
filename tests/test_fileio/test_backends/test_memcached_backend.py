# Copyright (c) OpenMMLab. All rights reserved.
import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

from parameterized import parameterized

from mmengine.fileio import MemcachedBackend

sys.modules['mc'] = MagicMock()


class MockMemcachedClient:

    def __init__(self, server_list_cfg, client_cfg):
        pass

    def Get(self, filepath, buffer):
        with open(filepath, 'rb') as f:
            buffer.content = f.read()


class TestMemcachedBackend(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mc_cfg = dict(server_list_cfg='', client_cfg='', sys_path=None)
        cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
        cls.img_path = cls.test_data_dir / 'color.jpg'
        cls.img_shape = (300, 400, 3)

    @parameterized.expand([[Path], [str]])
    @patch('mc.MemcachedClient.GetInstance', MockMemcachedClient)
    @patch('mc.pyvector', MagicMock)
    @patch('mc.ConvertBuffer', lambda x: x.content)
    def test_get_bytes(self, path_type):
        backend = MemcachedBackend(**self.mc_cfg)
        img_bytes = backend.get_bytes(path_type(self.img_path))
        self.assertEqual(self.img_path.open('rb').read(), img_bytes)
        try:
            import mmcv
        except ImportError:
            pass
        else:
            img = mmcv.imfrombytes(img_bytes)
            self.assertEqual(img.shape, self.img_shape)

    @patch('mc.MemcachedClient.GetInstance', MockMemcachedClient)
    @patch('mc.pyvector', MagicMock)
    @patch('mc.ConvertBuffer', lambda x: x.content)
    def test_get_text(self):
        backend = MemcachedBackend(**self.mc_cfg)
        with self.assertRaises(NotImplementedError):
            backend.get_text('filepath')
