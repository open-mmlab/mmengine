# Copyright (c) OpenMMLab. All rights reserved.
import random
from unittest import TestCase

import numpy as np
import pytest
import torch

from mmengine.data import PixelData


class TestPixelData(TestCase):

    def setup_data(self):
        metainfo = dict(
            img_id=random.randint(0, 100),
            img_shape=(random.randint(400, 600), random.randint(400, 600)))
        image = np.random.randint(0, 255, (4, 20, 40))
        featmap = torch.randint(0, 255, (10, 20, 40))
        pixel_data = PixelData(metainfo=metainfo, image=image, featmap=featmap)
        return pixel_data

    def test_set_data(self):
        pixel_data = self.setup_data()

        # test set '_metainfo_fields' or '_data_fields'
        with self.assertRaises(AttributeError):
            pixel_data._metainfo_fields = 1
        with self.assertRaises(AttributeError):
            pixel_data._data_fields = 1

        # value only supports (torch.Tensor, np.ndarray)
        with self.assertRaises(AssertionError):
            pixel_data.v = 'value'

        # The shape in PixelData must be the same
        with self.assertRaises(AssertionError):
            pixel_data.map2 = torch.randint(0, 255, (3, 21, 41))

        pixel_data.map2 = torch.randint(0, 255, (3, 20, 40))
        assert 'map2' in pixel_data

    def test_getitem(self):
        pixel_data = PixelData()

        pixel_data = self.setup_data()
        slice_pixel_data = pixel_data[10:15, 20:30]
        assert slice_pixel_data.shape == (5, 10)

        pixel_data = self.setup_data()
        slice_pixel_data = pixel_data[10, 20:30]
        assert slice_pixel_data.shape == (1, 10)

        # must be tuple
        item = torch.Tensor([1, 2, 3, 4])
        with pytest.raises(TypeError):
            pixel_data[item]
        item = 1
        with pytest.raises(TypeError):
            pixel_data[item]
        item = (5.5, 5)
        with pytest.raises(TypeError):
            pixel_data[item]

    def test_shape(self):
        pixel_data = self.setup_data()
        assert pixel_data.shape == (20, 40)
        pixel_data = PixelData()
        assert pixel_data.shape is None


if __name__ == '__main__':
    t = TestPixelData()
    t.test_getitem()
