# Copyright (c) OpenMMLab. All rights reserved.
import os
import platform
import shutil
import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from mmengine import Config
from mmengine.fileio import load
from mmengine.registry import VISBACKENDS
from mmengine.utils import digit_version, is_installed
from mmengine.visualization import (AimVisBackend, ClearMLVisBackend,
                                    DVCLiveVisBackend, LocalVisBackend,
                                    MLflowVisBackend, NeptuneVisBackend,
                                    TensorboardVisBackend, WandbVisBackend)


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

    def test_define_metric_cfg(self):
        # list of dict
        define_metric_cfg = [
            dict(name='test1', step_metric='iter'),
            dict(name='test2', step_metric='epoch'),
        ]
        wandb_vis_backend = WandbVisBackend(
            'temp_dir', define_metric_cfg=define_metric_cfg)
        wandb_vis_backend._init_env()
        wandb_vis_backend._wandb.define_metric.assert_any_call(
            name='test1', step_metric='iter')
        wandb_vis_backend._wandb.define_metric.assert_any_call(
            name='test2', step_metric='epoch')

        # dict
        define_metric_cfg = dict(test3='max')
        wandb_vis_backend = WandbVisBackend(
            'temp_dir', define_metric_cfg=define_metric_cfg)
        wandb_vis_backend._init_env()
        wandb_vis_backend._wandb.define_metric.assert_any_call(
            'test3', summary='max')

        shutil.rmtree('temp_dir')


class TestMLflowVisBackend:

    def test_init(self):
        MLflowVisBackend('temp_dir')
        VISBACKENDS.build(dict(type='MLflowVisBackend', save_dir='temp_dir'))

    def test_experiment(self):
        mlflow_vis_backend = MLflowVisBackend('temp_dir')
        assert mlflow_vis_backend.experiment == mlflow_vis_backend._mlflow

    def test_create_experiment(self):
        with patch('mlflow.create_experiment') as mock_create_experiment:
            MLflowVisBackend(
                'temp_dir', exp_name='test',
                artifact_location='foo')._init_env()
            mock_create_experiment.assert_any_call(
                'test', artifact_location='foo')

    def test_add_config(self):
        cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        mlflow_vis_backend = MLflowVisBackend('temp_dir')
        mlflow_vis_backend.add_config(cfg)

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        mlflow_vis_backend = MLflowVisBackend('temp_dir')
        mlflow_vis_backend.add_image('img.png', image)

    def test_add_scalar(self):
        mlflow_vis_backend = MLflowVisBackend('temp_dir')
        mlflow_vis_backend.add_scalar('map', 0.9)
        # test append mode
        mlflow_vis_backend.add_scalar('map', 0.9)
        mlflow_vis_backend.add_scalar('map', 0.95)

    def test_add_scalars(self):
        mlflow_vis_backend = MLflowVisBackend('temp_dir')
        input_dict = {'map': 0.7, 'acc': 0.9}
        mlflow_vis_backend.add_scalars(input_dict)
        # test append mode
        mlflow_vis_backend.add_scalars({'map': 0.8, 'acc': 0.8})

    def test_close(self):
        cfg = Config(dict(work_dir='temp_dir'))
        mlflow_vis_backend = MLflowVisBackend('temp_dir')
        mlflow_vis_backend._init_env()
        mlflow_vis_backend.add_config(cfg)
        mlflow_vis_backend.close()
        shutil.rmtree('temp_dir')


@patch.dict(sys.modules, {'clearml': MagicMock()})
class TestClearMLVisBackend:

    def test_init(self):
        ClearMLVisBackend('temp_dir')
        VISBACKENDS.build(dict(type='ClearMLVisBackend', save_dir='temp_dir'))

    def test_experiment(self):
        clearml_vis_backend = ClearMLVisBackend('temp_dir')
        assert clearml_vis_backend.experiment == clearml_vis_backend._clearml

    def test_add_config(self):
        cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        clearml_vis_backend = ClearMLVisBackend('temp_dir')
        clearml_vis_backend.add_config(cfg)

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        clearml_vis_backend = ClearMLVisBackend('temp_dir')
        clearml_vis_backend.add_image('img.png', image)

    def test_add_scalar(self):
        clearml_vis_backend = ClearMLVisBackend('temp_dir')
        clearml_vis_backend.add_scalar('map', 0.9)
        # test append mode
        clearml_vis_backend.add_scalar('map', 0.9)
        clearml_vis_backend.add_scalar('map', 0.95)

    def test_add_scalars(self):
        clearml_vis_backend = ClearMLVisBackend('temp_dir')
        input_dict = {'map': 0.7, 'acc': 0.9}
        clearml_vis_backend.add_scalars(input_dict)
        # test append mode
        clearml_vis_backend.add_scalars({'map': 0.8, 'acc': 0.8})

    def test_close(self):
        cfg = Config(dict(work_dir='temp_dir'))
        clearml_vis_backend = ClearMLVisBackend('temp_dir')
        clearml_vis_backend._init_env()
        clearml_vis_backend.add_config(cfg)
        clearml_vis_backend.close()


@pytest.mark.skipif(
    not is_installed('neptune'), reason='Neptune is not installed.')
class TestNeptuneVisBackend:

    def test_init(self):
        NeptuneVisBackend()
        VISBACKENDS.build(dict(type='NeptuneVisBackend'))

    def test_experiment(self):
        neptune_vis_backend = NeptuneVisBackend()
        assert neptune_vis_backend.experiment == neptune_vis_backend._neptune

    def test_add_config(self):
        cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        neptune_vis_backend = NeptuneVisBackend()
        neptune_vis_backend.add_config(cfg)

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        neptune_vis_backend = NeptuneVisBackend()
        neptune_vis_backend.add_image('img', image)
        neptune_vis_backend.add_image('img', image, step=1)

    def test_add_scalar(self):
        neptune_vis_backend = NeptuneVisBackend()
        neptune_vis_backend.add_scalar('map', 0.9)
        neptune_vis_backend.add_scalar('map', 0.9, step=1)
        neptune_vis_backend.add_scalar('map', 0.95, step=2)

    def test_add_scalars(self):
        neptune_vis_backend = NeptuneVisBackend()
        input_dict = {'map': 0.7, 'acc': 0.9}
        neptune_vis_backend.add_scalars(input_dict)

    def test_close(self):
        neptune_vis_backend = NeptuneVisBackend()
        neptune_vis_backend._init_env()
        neptune_vis_backend.close()


@pytest.mark.skipif(
    digit_version(platform.python_version()) < digit_version('3.8'),
    reason='DVCLiveVisBackend does not support python version < 3.8')
class TestDVCLiveVisBackend:

    def test_init(self):
        DVCLiveVisBackend('temp_dir')
        VISBACKENDS.build(dict(type='DVCLiveVisBackend', save_dir='temp_dir'))

    def test_experiment(self):
        dvclive_vis_backend = DVCLiveVisBackend('temp_dir')
        assert dvclive_vis_backend.experiment == dvclive_vis_backend._dvclive
        shutil.rmtree('temp_dir')

    def test_add_config(self):
        cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        dvclive_vis_backend = DVCLiveVisBackend('temp_dir')
        dvclive_vis_backend.add_config(cfg)
        shutil.rmtree('temp_dir')

    def test_add_image(self):
        img = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        dvclive_vis_backend = DVCLiveVisBackend('temp_dir')
        dvclive_vis_backend.add_image('img', img)
        shutil.rmtree('temp_dir')

    def test_add_scalar(self):
        dvclive_vis_backend = DVCLiveVisBackend('temp_dir')
        dvclive_vis_backend.add_scalar('mAP', 0.9)
        # test append mode
        dvclive_vis_backend.add_scalar('mAP', 0.9)
        dvclive_vis_backend.add_scalar('mAP', 0.95)
        shutil.rmtree('temp_dir')

    def test_add_scalars(self):
        dvclive_vis_backend = DVCLiveVisBackend('temp_dir')
        input_dict = {'map': 0.7, 'acc': 0.9}
        dvclive_vis_backend.add_scalars(input_dict)
        # test append mode
        dvclive_vis_backend.add_scalars({'map': 0.8, 'acc': 0.8})
        shutil.rmtree('temp_dir')

    def test_close(self):
        cfg = Config(dict(work_dir='temp_dir'))
        dvclive_vis_backend = DVCLiveVisBackend('temp_dir')
        dvclive_vis_backend._init_env()
        dvclive_vis_backend.add_config(cfg)
        dvclive_vis_backend.close()
        shutil.rmtree('temp_dir')


@pytest.mark.skipif(
    platform.system() == 'Windows',
    reason='Aim does not support Windows for now.')
class TestAimVisBackend:

    def test_init(self):
        AimVisBackend()
        VISBACKENDS.build(dict(type='AimVisBackend'))

    def test_experiment(self):
        aim_vis_backend = AimVisBackend()
        assert aim_vis_backend.experiment == aim_vis_backend._aim_run

    def test_add_config(self):
        cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        aim_vis_backend = AimVisBackend()
        aim_vis_backend.add_config(cfg)

    def test_add_image(self):
        image = np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        aim_vis_backend = AimVisBackend()
        aim_vis_backend.add_image('img', image)
        aim_vis_backend.add_image('img', image, step=1)

    def test_add_scalar(self):
        aim_vis_backend = AimVisBackend()
        aim_vis_backend.add_scalar('map', 0.9)
        aim_vis_backend.add_scalar('map', 0.9, step=1)
        aim_vis_backend.add_scalar('map', 0.95, step=2)

    def test_add_scalars(self):
        aim_vis_backend = AimVisBackend()
        input_dict = {'map': 0.7, 'acc': 0.9}
        aim_vis_backend.add_scalars(input_dict)

    def test_close(self):
        aim_vis_backend = AimVisBackend()
        aim_vis_backend._init_env()
        aim_vis_backend.close()
