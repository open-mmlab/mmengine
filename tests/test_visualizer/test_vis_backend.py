# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import sys
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mmengine import Config
from mmengine.fileio import load
from mmengine.registry import VISBACKENDS
from mmengine.visualization import (LocalVisBackend, TensorboardVisBackend,
                                    WandbVisBackend)


class TestLocalVisBackend:

    def test_init(self):
        # 'config_save_file' format must be py
        with pytest.raises(AssertionError):
            LocalVisBackend('temp_dir', config_save_file='a.txt')

        # 'scalar_save_file' format must be json
        with pytest.raises(AssertionError):
            LocalVisBackend('temp_dir', scalar_save_file='a.yaml')

        local_vis_backend = VISBACKENDS.build(
            dict(type='LocalVisBackend', save_dir='temp_dir'))
        assert isinstance(local_vis_backend, LocalVisBackend)

    def test_experiment(self):
        local_vis_backend = LocalVisBackend('temp_dir')
        assert local_vis_backend.experiment == local_vis_backend

    def test_add_config(self):
        cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        local_vis_backend = LocalVisBackend('temp_dir')
        local_vis_backend.add_config(cfg)
        assert os.path.exists(local_vis_backend._config_save_file)
        shutil.rmtree('temp_dir')

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3))
        local_vis_backend = LocalVisBackend('temp_dir')

        # image must be in np.uint8 format
        with pytest.raises(AssertionError):
            local_vis_backend.add_image('img', image)

        local_vis_backend.add_image('img', image.astype(np.uint8))
        assert os.path.exists(
            os.path.join(local_vis_backend._img_save_dir, 'img_0.png'))
        local_vis_backend.add_image('img', image.astype(np.uint8), step=2)
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
        local_vis_backend.add_scalar('map', 1, step=0)
        local_vis_backend.add_scalar('map', 0.95, step=1)
        # local_vis_backend.add_scalar('map', torch.IntTensor(1), step=2)
        local_vis_backend.add_scalar('map', np.array(0.9), step=2)
        with open(local_vis_backend._scalar_save_file) as f:
            out_dict = f.read()
        assert out_dict == '{"map": 1, "step": 0}\n' \
                           '{"map": 0.95, "step": 1}\n' \
                           '{"map": 0.9, "step": 2}\n'
        shutil.rmtree('temp_dir')

        local_vis_backend = LocalVisBackend('temp_dir')
        local_vis_backend.add_scalar('map', torch.tensor(1.))
        assert os.path.exists(local_vis_backend._scalar_save_file)
        shutil.rmtree('temp_dir')

    def test_add_scalars(self):
        input_dict = {'map': 0.7, 'acc': 0.9}
        local_vis_backend = LocalVisBackend('temp_dir')
        local_vis_backend.add_scalars(input_dict)
        assert input_dict == {'map': 0.7, 'acc': 0.9}
        out_dict = load(local_vis_backend._scalar_save_file, 'json')
        assert out_dict == {'map': 0.7, 'acc': 0.9, 'step': 0}

        # test append mode
        local_vis_backend.add_scalars({'map': 0.8, 'acc': 0.8}, step=1)
        with open(local_vis_backend._scalar_save_file) as f:
            out_dict = f.read()
        assert out_dict == '{"map": 0.7, "acc": 0.9, ' \
                           '"step": 0}\n{"map": 0.8, "acc": 0.8, "step": 1}\n'

        # test file_path
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
        VISBACKENDS.build(
            dict(type='TensorboardVisBackend', save_dir='temp_dir'))

    def test_experiment(self):
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        assert (tensorboard_vis_backend.experiment ==
                tensorboard_vis_backend._tensorboard)
        shutil.rmtree('temp_dir')

    def test_add_config(self):
        cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        tensorboard_vis_backend.add_config(cfg)
        shutil.rmtree('temp_dir')

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        tensorboard_vis_backend.add_image('img', image)
        tensorboard_vis_backend.add_image('img', image, step=2)
        shutil.rmtree('temp_dir')

    def test_add_scalar(self):
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        tensorboard_vis_backend.add_scalar('map', 0.9)
        # test append mode
        tensorboard_vis_backend.add_scalar('map', 0.9, step=0)
        tensorboard_vis_backend.add_scalar('map', 0.95, step=1)
        # test with numpy
        with warnings.catch_warnings(record=True) as record:
            tensorboard_vis_backend.add_scalar('map', np.array(0.9), step=0)
            tensorboard_vis_backend.add_scalar('map', np.array(0.95), step=1)
            tensorboard_vis_backend.add_scalar('map', np.array(9), step=0)
            tensorboard_vis_backend.add_scalar('map', np.array(95), step=1)
            tensorboard_vis_backend.add_scalar('map', np.array([9])[0], step=0)
            tensorboard_vis_backend.add_scalar(
                'map', np.array([95])[0], step=1)
        assert len(record) == 0
        # test with tensor
        tensorboard_vis_backend.add_scalar('map', torch.tensor(0.9), step=0)
        tensorboard_vis_backend.add_scalar('map', torch.tensor(0.95), step=1)
        # Unprocessable data will output a warning message
        with pytest.warns(Warning):
            tensorboard_vis_backend.add_scalar('map', [0.95])
        shutil.rmtree('temp_dir')

    def test_add_scalars(self):
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        # The step value must be passed through the parameter
        with pytest.raises(AssertionError):
            tensorboard_vis_backend.add_scalars({
                'map': 0.7,
                'acc': 0.9,
                'step': 1
            })

        # Unprocessable data will output a warning message
        with pytest.warns(Warning):
            tensorboard_vis_backend.add_scalars({
                'map': [1, 2],
            })

        input_dict = {'map': 0.7, 'acc': 0.9}
        tensorboard_vis_backend.add_scalars(input_dict)
        # test append mode
        tensorboard_vis_backend.add_scalars({'map': 0.8, 'acc': 0.8}, step=1)
        shutil.rmtree('temp_dir')

    def test_close(self):
        tensorboard_vis_backend = TensorboardVisBackend('temp_dir')
        tensorboard_vis_backend._init_env()
        tensorboard_vis_backend.close()
        shutil.rmtree('temp_dir')


class TestWandbVisBackend:
    sys.modules['wandb'] = MagicMock()
    sys.modules['wandb.run'] = MagicMock()

    def test_init(self):
        WandbVisBackend('temp_dir')
        VISBACKENDS.build(dict(type='WandbVisBackend', save_dir='temp_dir'))

    def test_experiment(self):
        wandb_vis_backend = WandbVisBackend('temp_dir')
        assert wandb_vis_backend.experiment == wandb_vis_backend._wandb
        shutil.rmtree('temp_dir')

    def test_add_config(self):
        cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        wandb_vis_backend = WandbVisBackend('temp_dir', log_code_name='code')
        _wandb = wandb_vis_backend.experiment
        _wandb.run.dir = 'temp_dir'
        wandb_vis_backend.add_config(cfg)
        shutil.rmtree('temp_dir')

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        wandb_vis_backend = WandbVisBackend('temp_dir')
        wandb_vis_backend.add_image('img', image)
        wandb_vis_backend.add_image('img', image)
        shutil.rmtree('temp_dir')

    def test_add_scalar(self):
        wandb_vis_backend = WandbVisBackend('temp_dir')
        wandb_vis_backend.add_scalar('map', 0.9)
        # test append mode
        wandb_vis_backend.add_scalar('map', 0.9)
        wandb_vis_backend.add_scalar('map', 0.95)
        shutil.rmtree('temp_dir')

    def test_add_scalars(self):
        wandb_vis_backend = WandbVisBackend('temp_dir')
        input_dict = {'map': 0.7, 'acc': 0.9}
        wandb_vis_backend.add_scalars(input_dict)
        # test append mode
        wandb_vis_backend.add_scalars({'map': 0.8, 'acc': 0.8})
        wandb_vis_backend.add_scalars({'map': [0.8], 'acc': 0.8})
        shutil.rmtree('temp_dir')

    def test_close(self):
        wandb_vis_backend = WandbVisBackend('temp_dir')
        wandb_vis_backend._init_env()
        wandb_vis_backend.close()
        shutil.rmtree('temp_dir')
