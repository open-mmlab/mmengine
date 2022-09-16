# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from unittest import TestCase

import cv2
import numpy as np

from mmengine.fileio.backends import HTTPBackend


def imfrombytes(content):
    img_np = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    return img


def imread(path):
    with open(path, 'rb') as f:
        content = f.read()
        img = imfrombytes(content)
    return img


class TestHTTPBackend(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.img_url = (
            'https://download.openmmlab.com/mmengine/test-data/color.jpg')
        cls.img_shape = (300, 400, 3)
        cls.text_url = (
            'https://download.openmmlab.com/mmengine/test-data/filelist.txt')
        cls.test_data_dir = Path(__file__).parent.parent.parent / 'data'
        cls.text_path = cls.test_data_dir / 'filelist.txt'

    def test_get(self):
        backend = HTTPBackend()
        img_bytes = backend.get(self.img_url)
        img = imfrombytes(img_bytes)
        self.assertEqual(img.shape, self.img_shape)

    def test_get_text(self):
        backend = HTTPBackend()
        text = backend.get_text(self.text_url)
        self.assertEqual(self.text_path.open('r').read(), text)

    def test_get_local_path(self):
        backend = HTTPBackend()
        with backend.get_local_path(self.img_url) as filepath:
            img = imread(filepath)
            self.assertEqual(img.shape, self.img_shape)
