# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

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
        cls.img_path = cls.test_data_dir / 'color.jpg'
        cls.text_path = cls.test_data_dir / 'filelist.txt'

    def test_get(self):
        backend = HTTPBackend()
        with patch('mmengine.fileio.backends.http_backend.urlopen') as urlopen:
            urlopen.return_value.read.return_value = self.img_path.read_bytes()
            img_bytes = backend.get(self.img_url)
            urlopen.assert_called_once_with(self.img_url)
        img = imfrombytes(img_bytes)
        self.assertEqual(img.shape, self.img_shape)

    def test_get_text(self):
        backend = HTTPBackend()
        expected_text = self.text_path.read_text()
        with patch('mmengine.fileio.backends.http_backend.urlopen') as urlopen:
            urlopen.return_value.read.return_value = expected_text.encode()
            text = backend.get_text(self.text_url)
            urlopen.assert_called_once_with(self.text_url)
        self.assertEqual(expected_text, text)

    def test_get_local_path(self):
        backend = HTTPBackend()
        with patch('mmengine.fileio.backends.http_backend.urlopen') as urlopen:
            urlopen.return_value.read.return_value = self.img_path.read_bytes()
            with backend.get_local_path(self.img_url) as filepath:
                img = imread(filepath)
            urlopen.assert_called_once_with(self.img_url)
        self.assertEqual(img.shape, self.img_shape)
