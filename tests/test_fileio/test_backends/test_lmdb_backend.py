# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from unittest import TestCase

import cv2
import numpy as np
from parameterized import parameterized

from mmengine.fileio.backends import LmdbBackend


def imfrombytes(content):
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return img


class TestLmdbBackend(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
        cls.lmdb_path = cls.test_data_dir / 'demo.lmdb'

    @parameterized.expand([[Path], [str]])
    def test_get(self, path_type):
        backend = LmdbBackend(path_type(self.lmdb_path))
        img_bytes = backend.get('baboon')
        img = imfrombytes(img_bytes)
        self.assertEqual(img.shape, (120, 125, 3))

    def test_get_text(self):
        backend = LmdbBackend(self.lmdb_path)
        with self.assertRaises(NotImplementedError):
            backend.get_text('filepath')
