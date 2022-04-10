# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from mmengine.fileio import load
from mmengine.registry import VISBACKEND
from mmengine.visualization import (LocalVisBackend, TensorboardVisBackend,
                                    WandbVisBackend)


class TestLocalVisBackend:

    def test_init(self):

        # 'params_save_file' format must be yaml
        with pytest.raises(AssertionError):
            LocalVisBackend('temp_dir', params_save_file='a.txt')

        # 'scalar_save_file' format must be json
        with pytest.raises(AssertionError):
            LocalVisBackend('temp_dir', scalar_save_file='a.yaml')

        local_vis_backend = LocalVisBackend('temp_dir')
        assert os.path.exists(local_vis_backend._save_dir)
        shutil.rmtree('temp_dir')

        local_vis_backend = VISBACKEND.build(
            dict(type='LocalVisBackend', save_dir='temp_dir'))
        assert os.path.exists(local_vis_backend._save_dir)
        shutil.rmtree('temp_dir')

    def test_experiment(self):
        local_vis_backend = LocalVisBackend('temp_dir')
        assert local_vis_backend.experiment == local_vis_backend
        shutil.rmtree('temp_dir')

    def test_add_config(self):
        local_vis_backend = LocalVisBackend('temp_dir')

        # 'params_dict' must be dict
        with pytest.raises(AssertionError):
            local_vis_backend.add_config(['lr', 0])

        params_dict = dict(lr=0.1, wd=[1.0, 0.1, 0.001], mode='linear')
        local_vis_backend.add_config(params_dict)
        out_dict = load(local_vis_backend._params_save_file, 'yaml')
        assert out_dict == params_dict
        shutil.rmtree('temp_dir')

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        local_vis_backend = LocalVisBackend('temp_dir')
        local_vis_backend.add_image('img', image)
        assert os.path.exists(
            os.path.join(local_vis_backend._img_save_dir, 'img_0.png'))

        local_vis_backend.add_image('img', image, step=2)
        assert os.path.exists(
            os.path.join(local_vis_backend._img_save_dir, 'img_2.png'))

        shutil.rmtree('temp_dir')

    def test_add_scalar(self):
        local_vis_backend = LocalVisBackend('temp_dir')
        local_vis_backend.add_scalar('map', 0.9)
        out_dict = load(local_vis_backend._scalar_save_file, 'json')
        assert out_dict == {'map': 0.9, 'step': 0}
        shutil.rmtree('temp_dir')

        # test append mode
        local_vis_backend = LocalVisBackend('temp_dir')
        local_vis_backend.add_scalar('map', 0.9, step=0)
        local_vis_backend.add_scalar('map', 0.95, step=1)
        with open(local_vis_backend._scalar_save_file) as f:
            out_dict = f.read()
        assert out_dict == '{"map": 0.9, "step": 0}\n{"map": ' \
                           '0.95, "step": 1}\n'
        shutil.rmtree('temp_dir')

    def test_add_scalars(self):
        local_vis_backend = LocalVisBackend('temp_dir')
        input_dict = {'map': 0.7, 'acc': 0.9}
        local_vis_backend.add_scalars(input_dict)
        out_dict = load(local_vis_backend._scalar_save_file, 'json')
        assert out_dict == {'map': 0.7, 'acc': 0.9, 'step': 0}

        # test append mode
        local_vis_backend.add_scalars({'map': 0.8, 'acc': 0.8}, step=1)
        with open(local_vis_backend._scalar_save_file) as f:
            out_dict = f.read()
        assert out_dict == '{"map": 0.7, "acc": 0.9, ' \
                           '"step": 0}\n{"map": 0.8, "acc": 0.8, "step": 1}\n'

        # test file_path
        local_vis_backend = LocalVisBackend('temp_dir')
        local_vis_backend.add_scalars(input_dict, file_path='temp.json')
        assert os.path.exists(local_vis_backend._scalar_save_file)
        assert os.path.exists(
            os.path.join(local_vis_backend._save_dir, 'temp.json'))

        # file_path and scalar_save_file cannot be the same
        with pytest.raises(AssertionError):
            local_vis_backend.add_scalars(input_dict, file_path='scalars.json')

        shutil.rmtree('temp_dir')


class TestTensorboardVisBackend:
    sys.modules['torch.utils.tensorboard'] = MagicMock()
    sys.modules['tensorboardX'] = MagicMock()

    def test_init(self):

        TensorboardVisBackend('temp_dir')
        VISBACKEND.build(
            dict(type='TensorboardVisBackend', save_dir='temp_dir'))

    def test_experiment(self):
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        assert (tensorboard_vis_backend.experiment ==
                tensorboard_vis_backend._tensorboard)

    def test_add_graph(self):

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 2, 1)

            def forward(self, x, y=None):
                return self.conv(x)

        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')

        # input must be tensor
        with pytest.raises(AssertionError):
            tensorboard_vis_backend.add_graph(Model(), np.zeros([1, 1, 3, 3]))

        # input must be 4d tensor
        with pytest.raises(AssertionError):
            tensorboard_vis_backend.add_graph(Model(), torch.zeros([1, 3, 3]))

        # If the input is a list, the inner element must be a 4d tensor
        with pytest.raises(AssertionError):
            tensorboard_vis_backend.add_graph(
                Model(), [torch.zeros([1, 1, 3, 3]),
                          torch.zeros([1, 3, 3])])

        tensorboard_vis_backend.add_graph(Model(), torch.zeros([1, 1, 3, 3]))
        tensorboard_vis_backend.add_graph(
            Model(), [torch.zeros([1, 1, 3, 3]),
                      torch.zeros([1, 1, 3, 3])])

    def test_add_config(self):
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')

        # 'params_dict' must be dict
        with pytest.raises(AssertionError):
            tensorboard_vis_backend.add_config(['lr', 0])

        params_dict = dict(lr=0.1, wd=0.2, mode='linear')
        tensorboard_vis_backend.add_config(params_dict)

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)

        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        tensorboard_vis_backend.add_image('img', image)

        tensorboard_vis_backend.add_image('img', image, step=2)

    def test_add_scalar(self):
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        tensorboard_vis_backend.add_scalar('map', 0.9)
        # test append mode
        tensorboard_vis_backend.add_scalar('map', 0.9, step=0)
        tensorboard_vis_backend.add_scalar('map', 0.95, step=1)

    def test_add_scalars(self):
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        # The step value must be passed through the parameter
        with pytest.raises(AssertionError):
            tensorboard_vis_backend.add_scalars({
                'map': 0.7,
                'acc': 0.9,
                'step': 1
            })

        input_dict = {'map': 0.7, 'acc': 0.9}
        tensorboard_vis_backend.add_scalars(input_dict)
        # test append mode
        tensorboard_vis_backend.add_scalars({'map': 0.8, 'acc': 0.8}, step=1)


class TestWandbVisBackend:
    sys.modules['wandb'] = MagicMock()

    def test_init(self):
        WandbVisBackend()
        VISBACKEND.build(dict(type='WandbVisBackend', save_dir='temp_dir'))

    def test_experiment(self):
        wandb_vis_backend = WandbVisBackend()
        assert wandb_vis_backend.experiment == wandb_vis_backend._wandb

    def test_add_config(self):
        wandb_vis_backend = WandbVisBackend()

        # 'params_dict' must be dict
        with pytest.raises(AssertionError):
            wandb_vis_backend.add_config(['lr', 0])

        params_dict = dict(lr=0.1, wd=0.2, mode='linear')
        wandb_vis_backend.add_config(params_dict)

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)

        wandb_vis_backend = WandbVisBackend()
        wandb_vis_backend.add_image('img', image)

        wandb_vis_backend.add_image('img', image, step=2)

    def test_add_scalar(self):
        wandb_vis_backend = WandbVisBackend()
        wandb_vis_backend.add_scalar('map', 0.9)
        # test append mode
        wandb_vis_backend.add_scalar('map', 0.9, step=0)
        wandb_vis_backend.add_scalar('map', 0.95, step=1)

    def test_add_scalars(self):
        wandb_vis_backend = WandbVisBackend()
        input_dict = {'map': 0.7, 'acc': 0.9}
        wandb_vis_backend.add_scalars(input_dict)
        # test append mode
        wandb_vis_backend.add_scalars({'map': 0.8, 'acc': 0.8}, step=1)
