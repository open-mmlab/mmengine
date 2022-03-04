# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from mmengine.fileio import load
from mmengine.registry import VISUALIZERS, WRITERS
from mmengine.visualization import (ComposedWriter, LocalWriter,
                                    TensorboardWriter, WandbWriter)


def draw(self, data_sample, image=None, show_gt=True, show_pred=True):
    self.set_image(image)


class TestLocalWriter:

    def test_init(self):
        # visuailzer must be a dictionary or an instance
        # of Visualizer and its subclasses
        with pytest.raises(AssertionError):
            LocalWriter('temp_dir', [dict(type='Visualizer')])

        # 'save_hyperparams_name' format must be yaml
        with pytest.raises(AssertionError):
            LocalWriter('temp_dir', save_hyperparams_name='a.txt')

        # 'save_scalar_name' format must be json
        with pytest.raises(AssertionError):
            LocalWriter('temp_dir', save_scalar_name='a.yaml')

        local_writer = LocalWriter('temp_dir')
        assert os.path.exists(local_writer._save_dir)
        shutil.rmtree('temp_dir')

        local_writer = WRITERS.build(
            dict(
                type='LocalWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir'))
        assert os.path.exists(local_writer._save_dir)
        shutil.rmtree('temp_dir')

    def test_experiment(self):
        local_writer = LocalWriter('temp_dir')
        assert local_writer.experiment == local_writer
        shutil.rmtree('temp_dir')

    def test_add_hyperparams(self):
        local_writer = LocalWriter('temp_dir')

        # 'params_dict' must be dict
        with pytest.raises(AssertionError):
            local_writer.add_hyperparams(['lr', 0])

        params_dict = dict(lr=0.1, wd=[1.0, 0.1, 0.001], mode='linear')
        local_writer.add_hyperparams(params_dict)
        out_dict = load(local_writer._save_hyperparams_name, 'yaml')
        assert out_dict == params_dict
        shutil.rmtree('temp_dir')

    @patch('mmengine.visualization.visualizer.Visualizer.draw', draw)
    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)

        # The visuailzer parameter must be set when
        # the local_writer object is instantiated and
        # the `add_image` method is called.
        with pytest.raises(AssertionError):
            local_writer = LocalWriter('temp_dir')
            local_writer.add_image('img', image)

        local_writer = LocalWriter('temp_dir', dict(type='Visualizer'))
        local_writer.add_image('img', image)
        assert os.path.exists(
            os.path.join(local_writer._save_img_folder, 'img_0.png'))

        bboxes = np.array([[1, 1, 2, 2], [1, 1.5, 1, 2.5]])
        local_writer.visualizer.draw_bboxes(bboxes)
        local_writer.add_image(
            'img', local_writer.visualizer.get_image(), step=2)
        assert os.path.exists(
            os.path.join(local_writer._save_img_folder, 'img_2.png'))

        visuailzer = VISUALIZERS.build(dict(type='Visualizer'))
        local_writer = LocalWriter('temp_dir', visuailzer)
        local_writer.add_image('img', image)
        assert os.path.exists(
            os.path.join(local_writer._save_img_folder, 'img_0.png'))

        shutil.rmtree('temp_dir')

    def test_add_scalar(self):
        local_writer = LocalWriter('temp_dir')
        local_writer.add_scalar('map', 0.9)
        out_dict = load(local_writer._save_scalar_name, 'json')
        assert out_dict == {'map': 0.9, 'step': 0}
        shutil.rmtree('temp_dir')

        # test append mode
        local_writer = LocalWriter('temp_dir')
        local_writer.add_scalar('map', 0.9, step=0)
        local_writer.add_scalar('map', 0.95, step=1)
        with open(local_writer._save_scalar_name) as f:
            out_dict = f.read()
        assert out_dict == '{"map": 0.9, "step": 0}\n{"map": ' \
                           '0.95, "step": 1}\n'
        shutil.rmtree('temp_dir')

    def test_add_scalars(self):
        local_writer = LocalWriter('temp_dir')
        input_dict = {'map': 0.7, 'acc': 0.9}
        local_writer.add_scalars(input_dict)
        out_dict = load(local_writer._save_scalar_name, 'json')
        assert out_dict == {'map': 0.7, 'acc': 0.9, 'step': 0}

        # test append mode
        local_writer.add_scalars({'map': 0.8, 'acc': 0.8}, step=1)
        with open(local_writer._save_scalar_name) as f:
            out_dict = f.read()
        assert out_dict == '{"map": 0.7, "acc": 0.9, ' \
                           '"step": 0}\n{"map": 0.8, "acc": 0.8, "step": 1}\n'

        # test file_name
        local_writer = LocalWriter('temp_dir')
        local_writer.add_scalars(input_dict, file_name='temp.json')
        assert os.path.exists(local_writer._save_scalar_name)
        assert os.path.exists(
            os.path.join(local_writer._save_dir, 'temp.json'))

        shutil.rmtree('temp_dir')


class TestTensorboardWriter:
    sys.modules['torch.utils.tensorboard'] = MagicMock()
    sys.modules['tensorboardX'] = MagicMock()

    def test_init(self):
        # visuailzer must be a dictionary or an instance
        # of Visualizer and its subclasses
        with pytest.raises(AssertionError):
            LocalWriter('temp_dir', [dict(type='Visualizer')])

        TensorboardWriter('temp_dir')
        WRITERS.build(
            dict(
                type='TensorboardWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir'))

    def test_experiment(self):
        tensorboard_writer = TensorboardWriter('temp_dir')
        assert tensorboard_writer.experiment == tensorboard_writer._tensorboard

    def test_add_graph(self):

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 1)

            def forward(self, x, y=None):
                return self.conv(x)

        tensorboard_writer = TensorboardWriter('temp_dir')

        # input must be tensor
        with pytest.raises(AssertionError):
            tensorboard_writer.add_graph(Model(), np.zeros([1, 1, 3, 3]))

        # input must be 4d tensor
        with pytest.raises(AssertionError):
            tensorboard_writer.add_graph(Model(), torch.zeros([1, 3, 3]))

        # If the input is a list, the inner element must be a 4d tensor
        with pytest.raises(AssertionError):
            tensorboard_writer.add_graph(
                Model(), [torch.zeros([1, 1, 3, 3]),
                          torch.zeros([1, 3, 3])])

        tensorboard_writer.add_graph(Model(), torch.zeros([1, 1, 3, 3]))
        tensorboard_writer.add_graph(
            Model(), [torch.zeros([1, 1, 3, 3]),
                      torch.zeros([1, 1, 3, 3])])

    def test_add_hyperparams(self):
        tensorboard_writer = TensorboardWriter('temp_dir')

        # 'params_dict' must be dict
        with pytest.raises(AssertionError):
            tensorboard_writer.add_hyperparams(['lr', 0])

        params_dict = dict(lr=0.1, wd=0.2, mode='linear')
        tensorboard_writer.add_hyperparams(params_dict)

    @patch('mmengine.visualization.visualizer.Visualizer.draw', draw)
    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)

        # The visuailzer parameter must be set when
        # the local_writer object is instantiated and
        # the `add_image` method is called.
        with pytest.raises(AssertionError):
            tensorboard_writer = TensorboardWriter('temp_dir')
            tensorboard_writer.add_image('img', image)

        tensorboard_writer = TensorboardWriter('temp_dir',
                                               dict(type='Visualizer'))
        tensorboard_writer.add_image('img', image)

        bboxes = np.array([[1, 1, 2, 2], [1, 1.5, 1, 2.5]])
        tensorboard_writer.visualizer.draw_bboxes(bboxes)
        tensorboard_writer.add_image(
            'img', tensorboard_writer.visualizer.get_image(), step=2)

        visuailzer = VISUALIZERS.build(dict(type='Visualizer'))
        tensorboard_writer = TensorboardWriter('temp_dir', visuailzer)
        tensorboard_writer.add_image('img', image)

    def test_add_scalar(self):
        tensorboard_writer = TensorboardWriter('temp_dir')
        tensorboard_writer.add_scalar('map', 0.9)
        # test append mode
        tensorboard_writer.add_scalar('map', 0.9, step=0)
        tensorboard_writer.add_scalar('map', 0.95, step=1)

    def test_add_scalars(self):
        tensorboard_writer = TensorboardWriter('temp_dir')
        input_dict = {'map': 0.7, 'acc': 0.9}
        tensorboard_writer.add_scalars(input_dict)
        # test append mode
        tensorboard_writer.add_scalars({'map': 0.8, 'acc': 0.8}, step=1)


class TestWandbWriter:
    sys.modules['wandb'] = MagicMock()

    def test_init(self):
        WandbWriter()
        WRITERS.build(
            dict(
                type='WandbWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir'))

    def test_experiment(self):
        wandb_writer = WandbWriter()
        assert wandb_writer.experiment == wandb_writer._wandb

    def test_add_hyperparams(self):
        wandb_writer = WandbWriter()

        # 'params_dict' must be dict
        with pytest.raises(AssertionError):
            wandb_writer.add_hyperparams(['lr', 0])

        params_dict = dict(lr=0.1, wd=0.2, mode='linear')
        wandb_writer.add_hyperparams(params_dict)

    @patch('mmengine.visualization.visualizer.Visualizer.draw', draw)
    @patch('mmengine.visualization.writer.WandbWriter.add_image_to_wandb',
           Mock)
    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)

        wandb_writer = WandbWriter()
        wandb_writer.add_image('img', image)

        wandb_writer = WandbWriter(visualizer=dict(type='Visualizer'))
        bboxes = np.array([[1, 1, 2, 2], [1, 1.5, 1, 2.5]])
        wandb_writer.visualizer.set_image(image)
        wandb_writer.visualizer.draw_bboxes(bboxes)
        wandb_writer.add_image(
            'img', wandb_writer.visualizer.get_image(), step=2)

        visuailzer = VISUALIZERS.build(dict(type='Visualizer'))
        wandb_writer = WandbWriter(visualizer=visuailzer)
        wandb_writer.add_image('img', image)

    def test_add_scalar(self):
        wandb_writer = WandbWriter()
        wandb_writer.add_scalar('map', 0.9)
        # test append mode
        wandb_writer.add_scalar('map', 0.9, step=0)
        wandb_writer.add_scalar('map', 0.95, step=1)

    def test_add_scalars(self):
        wandb_writer = WandbWriter()
        input_dict = {'map': 0.7, 'acc': 0.9}
        wandb_writer.add_scalars(input_dict)
        # test append mode
        wandb_writer.add_scalars({'map': 0.8, 'acc': 0.8}, step=1)


class TestComposedWriter:
    sys.modules['torch.utils.tensorboard'] = MagicMock()
    sys.modules['tensorboardX'] = MagicMock()
    sys.modules['wandb'] = MagicMock()

    def test_init(self):

        class A:
            pass

        # The writers inner element must be a dictionary or a
        # subclass of Writer.
        with pytest.raises(AssertionError):
            ComposedWriter(writers=[A()])

        composed_writer = ComposedWriter(writers=[
            WandbWriter(),
            dict(
                type='TensorboardWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir')
        ])
        assert len(composed_writer._writer) == 2

    def test_get_writer(self):
        composed_writer = ComposedWriter(writers=[
            WandbWriter(),
            dict(
                type='TensorboardWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir')
        ])
        assert isinstance(composed_writer.get_writer(0), WandbWriter)
        assert isinstance(composed_writer.get_writer(1), TensorboardWriter)

    def test_get_experiment(self):
        composed_writer = ComposedWriter(writers=[
            WandbWriter(),
            dict(
                type='TensorboardWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir')
        ])
        assert composed_writer.get_experiment(
            0) == composed_writer._writer[0].experiment
        assert composed_writer.get_experiment(
            1) == composed_writer._writer[1].experiment

    def test_add_hyperparams(self):
        composed_writer = ComposedWriter(writers=[
            WandbWriter(),
            dict(
                type='TensorboardWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir')
        ])

        # 'params_dict' must be dict
        with pytest.raises(AssertionError):
            composed_writer.add_hyperparams(['lr', 0])

        params_dict = dict(lr=0.1, wd=0.2, mode='linear')
        composed_writer.add_hyperparams(params_dict)

    def test_add_graph(self):

        composed_writer = ComposedWriter(writers=[
            WandbWriter(),
            dict(
                type='TensorboardWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir')
        ])

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 1)

            def forward(self, x, y=None):
                return self.conv(x)

        # input must be tensor
        with pytest.raises(AssertionError):
            composed_writer.add_graph(Model(), np.zeros([1, 1, 3, 3]))

        # input must be 4d tensor
        with pytest.raises(AssertionError):
            composed_writer.add_graph(Model(), torch.zeros([1, 3, 3]))

        # If the input is a list, the inner element must be a 4d tensor
        with pytest.raises(AssertionError):
            composed_writer.add_graph(
                Model(), [torch.zeros([1, 1, 3, 3]),
                          torch.zeros([1, 3, 3])])

        composed_writer.add_graph(Model(), torch.zeros([1, 1, 3, 3]))
        composed_writer.add_graph(
            Model(), [torch.zeros([1, 1, 3, 3]),
                      torch.zeros([1, 1, 3, 3])])

    @patch('mmengine.visualization.visualizer.Visualizer.draw', draw)
    @patch('mmengine.visualization.writer.WandbWriter.add_image_to_wandb',
           Mock)
    def test_add_image(self):
        composed_writer = ComposedWriter(writers=[
            WandbWriter(),
            dict(
                type='TensorboardWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir')
        ])

        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        composed_writer.add_image('img', image)

        bboxes = np.array([[1, 1, 2, 2], [1, 1.5, 1, 2.5]])
        composed_writer.get_writer(1).visualizer.draw_bboxes(bboxes)
        composed_writer.get_writer(1).add_image(
            'img',
            composed_writer.get_writer(1).visualizer.get_image(),
            step=2)

    def test_add_scalar(self):
        composed_writer = ComposedWriter(writers=[
            WandbWriter(),
            dict(
                type='TensorboardWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir')
        ])
        composed_writer.add_scalar('map', 0.9)
        # test append mode
        composed_writer.add_scalar('map', 0.9, step=0)
        composed_writer.add_scalar('map', 0.95, step=1)

    def test_add_scalars(self):
        composed_writer = ComposedWriter(writers=[
            WandbWriter(),
            dict(
                type='TensorboardWriter',
                visualizer=dict(type='Visualizer'),
                save_dir='temp_dir')
        ])
        input_dict = {'map': 0.7, 'acc': 0.9}
        composed_writer.add_scalars(input_dict)
        # test append mode
        composed_writer.add_scalars({'map': 0.8, 'acc': 0.8}, step=1)
