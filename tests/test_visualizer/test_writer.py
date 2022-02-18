# Copyright (c) OpenMMLab. All rights reserved.
import random
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mmengine.data import BaseDataElement, BaseDataSample
from mmengine.visualizer import (VISUALIZERS, LocalWriter, TensorboardWriter,
                                 WandbWriter)


def get_demo_datasample():
    metainfo = dict(
        img_id=random.randint(0, 100),
        img_shape=(random.randint(400, 600), random.randint(400, 600)))
    gt_instances = BaseDataElement(
        data=dict(bboxes=torch.rand((5, 4)), labels=torch.rand((5, ))))
    pred_instances = BaseDataElement(
        data=dict(bboxes=torch.rand((5, 4)), scores=torch.rand((5, ))))
    data = dict(gt_instances=gt_instances, pred_instances=pred_instances)
    instances = BaseDataSample(data=data, metainfo=metainfo)
    return instances


class TestLocalWriter:

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3))
        instances = get_demo_datasample()

        local_writer = LocalWriter(visuailzer=dict(type='Visualizer'))
        local_writer.add_image('img', image)
        local_writer.add_image('img', image, instances)

        bboxes = np.array([[1, 1, 2, 2], [1, 1.5, 1, 2.5]])
        local_writer.visualizer.draw_bboxes(bboxes)
        local_writer.add_image('img', local_writer.visualizer.get_image())

        visuailzer = VISUALIZERS.build(dict(type='Visualizer'))
        local_writer = LocalWriter(visuailzer=visuailzer)
        local_writer.add_image('img', image)
        local_writer.add_image('img', image, instances)

        # test `visuailzer` parameter
        with pytest.raises(AssertionError):
            LocalWriter(visuailzer=dict(a='Visualizer'))
        with pytest.raises(AssertionError):

            class A:
                pass

            LocalWriter(visuailzer=A())

        # test not visuailzer
        with pytest.raises(AssertionError):
            local_writer = LocalWriter()
            local_writer.add_image('img', image)

    def test_add_scaler(self):
        local_writer = LocalWriter()
        local_writer.add_scaler('map', 0.9)

    def test_add_hyperparams(self):
        local_writer = LocalWriter()
        local_writer.add_hyperparams('hyper', dict(lr=0.01))


class TestWandbWriter:
    sys.modules['wandb'] = MagicMock()

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3))
        instances = get_demo_datasample()

        wandb_writer = WandbWriter()
        assert not wandb_writer.visualizer
        wandb_writer.add_image('img', image, instances)

        wandb_writer = WandbWriter(visuailzer=dict(type='Visualizer'))
        assert wandb_writer.visualizer
        wandb_writer.add_image('img', image)
        wandb_writer.add_image('img', image, instances)

        wandb_writer.visualizer.set_image(image)
        wandb_writer.add_image('img', wandb_writer.visualizer.get_image())

        # TODO test file exist

    def test_add_scaler(self):
        wandb_writer = WandbWriter()
        wandb_writer.add_scaler('map', 0.9)

    def test_add_hyperparams(self):
        wandb_writer = WandbWriter()
        wandb_writer.add_hyperparams('hyper', dict(lr=0.01))


class TestTensorboardWriter:
    sys.modules['torch.utils.tensorboard.SummaryWriter'] = MagicMock()

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3))
        instances = get_demo_datasample()

        tensorboard_writer = TensorboardWriter()
        assert not tensorboard_writer.visualizer
        tensorboard_writer.add_image('img', image, instances)

        tensorboard_writer = TensorboardWriter(
            visuailzer=dict(type='Visualizer'))
        assert tensorboard_writer.visualizer
        tensorboard_writer.add_image('img', image)
        tensorboard_writer.add_image('img', image, instances)

        tensorboard_writer.visualizer.set_image(image)
        tensorboard_writer.add_image('img',
                                     tensorboard_writer.visualizer.get_image())

        # test no visualizer
        with pytest.raises(AssertionError):
            tensorboard_writer = TensorboardWriter()
            tensorboard_writer.add_image('img', image, instances)

    def test_add_scaler(self):
        tensorboard_writer = TensorboardWriter()
        tensorboard_writer.add_scaler('map', 0.9)

    def test_add_hyperparams(self):
        tensorboard_writer = TensorboardWriter()
        tensorboard_writer.add_hyperparams('hyper', dict(lr=0.01))
