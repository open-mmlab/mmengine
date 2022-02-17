# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import numpy as np
import torch
import random
import sys
from mmengine.data import BaseDataElement, BaseDataSample
from unittest.mock import MagicMock


def get_demo_datasample():
    metainfo = dict(
        img_id=random.randint(0, 100),
        img_shape=(random.randint(400, 600), random.randint(400, 600)))
    gt_instances = BaseDataElement(
        data=dict(bboxes=torch.rand((5, 4)), labels=torch.rand((5,))))
    pred_instances = BaseDataElement(
        data=dict(bboxes=torch.rand((5, 4)), scores=torch.rand((5,))))
    data = dict(gt_instances=gt_instances, pred_instances=pred_instances)
    instances = BaseDataSample(data=data, metainfo=metainfo)
    return instances


class TestLocalWriter:
    def test_add_image(self):
        local_visualizer = LocalVisualizer()
        local_writer = LocalWriter()
        local_writer.bind_visualizer(local_visualizer)

        image = np.random.randint(0, 256, size=(10, 10, 3))
        local_writer.add_image('img', image)

        instances = get_demo_datasample()
        local_writer.add_image('img', image, instances)

    def test_add_scaler(self):
        local_writer = LocalWriter()
        local_writer.add_scaler('map', 0.9)

    def test_add_hyperparams(self):
        local_writer = LocalWriter()
        local_writer.add_hyperparams('hyper', dict(lr=0.01))


class TestWandWriter:

    sys.modules['petrel_client'] = MagicMock()

    def test_add_image(self):
        local_visualizer = LocalVisualizer()
        local_writer = WandWriter(use_visualizer=True)
        image = np.random.randint(0, 256, size=(10, 10, 3))

        # test no visualizer
        with pytest.raises(AssertionError):
            local_writer.add_image('img', image)

        local_writer.bind_visualizer(local_visualizer)
        local_writer.add_image('img', image)
        instances = get_demo_datasample()
        local_writer.add_image('img', image, instances)

        local_writer = WandWriter(use_visualizer=False)
        local_writer.add_image('img', image)
        instances = get_demo_datasample()
        local_writer.add_image('img', image, instances)

    def test_add_scaler(self):
        local_writer = WandWriter()
        local_writer.add_scaler('map', 0.9)

    def test_add_hyperparams(self):
        local_writer = WandWriter()
        local_writer.add_hyperparams('hyper', dict(lr=0.01))


class TestTensorboardWriter:

    sys.modules['torch.utils.tensorboard.SummaryWriter'] = MagicMock()

    def test_add_image(self):
        local_visualizer = LocalVisualizer()

        # test no visualizer
        with pytest.raises(AssertionError):
            TensorboardWriter(use_visualizer=False)

        image = np.random.randint(0, 256, size=(10, 10, 3))
        local_writer = TensorboardWriter(use_visualizer=True)
        local_writer.bind_visualizer(local_visualizer)
        local_writer.add_image('img', image)
        instances = get_demo_datasample()
        local_writer.add_image('img', image, instances)

    def test_add_scaler(self):
        local_writer = TensorboardWriter()
        local_writer.add_scaler('map', 0.9)

    def test_add_hyperparams(self):
        local_writer = TensorboardWriter()
        local_writer.add_hyperparams('hyper', dict(lr=0.01))
