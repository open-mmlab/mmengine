# Copyright (c) OpenMMLab. All rights reserved.
import os
from abc import ABCMeta, abstractmethod

import cv2

from mmengine.fileio import dump
from mmengine.logging import BaseGlobalAccessible
from mmengine.registry import VISUALIZERS, WRITERS
from mmengine.utils import TORCH_VERSION
from .visualizer import Visualizer


class BaseWriter(metaclass=ABCMeta):
    """Base class for experiment writer."""

    def __init__(self, visualizer=None, save_dir=None):
        self._save_dir = save_dir
        self._visualizer = visualizer
        if visualizer:
            if isinstance(visualizer, dict):
                self._visualizer = VISUALIZERS.build(visualizer)
            else:
                assert isinstance(visualizer, Visualizer), \
                    f'visualizer should be an instance of Visualizer, ' \
                    f'but got {type(visualizer)}'

    @property
    def visualizer(self):
        return self._visualizer

    @property
    @abstractmethod
    def experiment(self):
        """Return the experiment object associated with this writer."""
        pass

    def add_hyperparams(self, name, hparam_dict, **kwargs):
        """Record hyperparameters."""
        pass

    def add_graph(self, model, input_array=None, **kwargs):
        """Record model graph."""
        pass

    def add_image(self,
                  name,
                  image,
                  data_samples=None,
                  step=0,
                  show_gt=True,
                  show_pred=True,
                  **kwargs):
        pass

    def add_scalar(self, name, value, step=0, **kwargs) -> None:
        pass

    def add_scalars(self, name, value_dict, step=0, **kwargs) -> None:
        pass

    def close(self):
        pass


@WRITERS.register_module()
class LocalWriter(BaseWriter):

    def __init__(self,
                 visualizer,
                 save_dir=None,
                 save_img_folder='writer_image',
                 save_hyperparams_name='hyperparams.txt',
                 save_scaler_name='scaler.json',
                 img_show=False):
        assert save_dir is not None
        super(LocalWriter, self).__init__(visualizer, save_dir)
        self._save_img_folder = os.path.join(self._save_dir, save_img_folder)
        self._save_scaler_name = os.path.join(self._save_dir, save_scaler_name)
        self._save_hyperparams_name = os.path.join(self._save_dir,
                                                   save_hyperparams_name)
        self._img_show = img_show

    @property
    def experiment(self):
        return self

    def add_hyperparams(self, name, params_dict, **kwargs):
        """Record hyperparameters."""
        self._dump(self._save_hyperparams_name, 'txt', {name: params_dict})

    def add_image(self,
                  name,
                  image,
                  data_samples=None,
                  step=0,
                  show_gt=True,
                  show_pred=True,
                  **kwargs):
        assert self.visualizer
        self.visualizer.draw(data_samples, image, show_gt, show_pred)
        if self._img_show:
            self.visualizer.show()
        else:
            os.makedirs(self._save_img_folder, exist_ok=True)
            save_file_name = f'{name}_{step}.png'
            cv2.imwrite(
                os.path.join(self._save_img_folder, save_file_name),
                self.visualizer.get_image())

    def _dump(self, file_name, file_format, value_dict):
        with open(file_name, 'a+') as f:
            if file_format == 'json':
                dump(value_dict, f, file_format=file_format)
            else:
                f.write(value_dict)
            f.write('\n')

    def add_scaler(self, name, value, step=0):
        self._dump(self._save_scaler_name, 'json', {name: value, 'step': step})

    def add_scalars(self, name, value_dict, step=0, **kwargs) -> None:
        self._dump(self._save_scaler_name, 'json', {
            name: value_dict,
            'step': step
        })


@WRITERS.register_module()
class WandbWriter(BaseWriter):

    def __init__(self,
                 init_kwargs=None,
                 commit=True,
                 with_step=False,
                 sync=True,
                 visualizer=None,
                 save_dir=None):
        super(WandbWriter, self).__init__(visualizer, save_dir)
        self._commit = commit
        self._with_step = with_step
        self._sync = sync
        self._wandb = self._setup_env(init_kwargs)

    @property
    def experiment(self):
        return self._wandb

    def _setup_env(self, init_kwargs):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        if init_kwargs:
            wandb.init(**init_kwargs)
        else:
            wandb.init()

        return wandb

    def add_hyperparams(self, name, hparam_dict, **kwargs):
        self._wandb.log({name: hparam_dict},
                        commit=self._commit,
                        sync=self._sync)

    def add_image(self,
                  name,
                  image,
                  data_samples=None,
                  step=0,
                  show_gt=True,
                  show_pred=True,
                  **kwargs):
        if self.visualizer:
            self.visualizer.draw(data_samples, image, show_gt, show_pred)
            self._wandb.log({name: self.visualizer.get_image()},
                            commit=self._commit,
                            step=step,
                            sync=self._sync)
        else:
            self.add_image_to_wandb(name, image, data_samples, step, show_gt,
                                    show_pred, **kwargs)

    def add_scalar(self, name, value, step=0, **kwargs) -> None:
        self._wandb.log({name: value},
                        commit=self._commit,
                        step=step,
                        sync=self._sync)

    def add_scalars(self, name, value_dict, step=0, **kwargs) -> None:
        self._wandb.log({name: value_dict},
                        commit=self._commit,
                        step=step,
                        sync=self._sync)

    def add_image_to_wandb(self,
                           name,
                           image,
                           data_samples,
                           step,
                           show_gt=True,
                           show_pred=True,
                           **kwargs):
        raise NotImplementedError()

    def close(self):
        if hasattr(self, '_wandb'):
            self._wandb.join()


@WRITERS.register_module()
class TensorboardWriter(BaseWriter):
    """Write all scalars to a tensorboard file."""

    def __init__(self, visualizer=None, save_dir=None, log_dir='tf_writer'):
        assert save_dir is not None
        super(TensorboardWriter, self).__init__(visualizer, save_dir)
        self._tensorboard = self._setup_env(log_dir)

    def _setup_env(self, log_dir):
        if TORCH_VERSION == 'parrots':
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        self.log_dir = os.path.join(self._save_dir, log_dir)
        return SummaryWriter(self.log_dir)

    @property
    def experiment(self):
        return self._tensorboard

    def add_graph(self, model, input_array=None, **kwargs):
        assert input_array.ndim == 4
        self._tensorboard.add_graph(model, input_array)

    def add_hyperparams(self, name, hparam_dict, **kwargs):
        self._tensorboard.add_hparams(hparam_dict, {})

    def add_image(self,
                  name,
                  image,
                  data_samples=None,
                  step=0,
                  show_gt=True,
                  show_pred=True,
                  **kwargs):
        assert self.visualizer
        self.visualizer.draw(data_samples, image, show_gt, show_pred)
        self._tensorboard.add_image(name, self.visualizer.get_image())

    def add_scalar(self, name, value, step=0, **kwargs) -> None:
        self._tensorboard.add_scalar(name, value, step)

    def add_scalars(self, name, value_dict, step=0, **kwargs) -> None:
        self._tensorboard.add_scalars(name, value_dict, step)

    def close(self):
        if hasattr(self, '_tensorboard'):
            self._tensorboard.close()


class ComposedWriter(BaseGlobalAccessible):

    def __init__(self, writers, name='composed_writer'):
        super(ComposedWriter, self).__init__(name)
        assert isinstance(writers, list)
        self._writer = []
        for writer in writers:
            if isinstance(writer, dict):
                self._writer.append(WRITERS.build(writer))
            else:
                assert isinstance(writer, BaseWriter), \
                    f'writer should be an instance of a subclass of ' \
                    f'BaseWriter, but got {type(writer)}'
                self._writer.append(writer)

    def get_writer(self, index):
        return self._writer[index]

    def get_experiment(self, index):
        return self._writer[index].experiment

    def add_hyperparams(self, name, hparam_dict, **kwargs):
        for writer in self._writer:
            writer.add_hyperparams(name, hparam_dict, **kwargs)

    def add_graph(self, model, input_array=None, **kwargs):
        """Record model graph."""
        for writer in self._writer:
            writer.add_graph(model, input_array, **kwargs)

    def add_image(self,
                  name,
                  image,
                  data_samples=None,
                  step=0,
                  show_gt=True,
                  show_pred=True,
                  **kwargs):
        for writer in self._writer:
            writer.add_image(name, image, data_samples, step, show_gt,
                             show_pred, **kwargs)

    def add_scalar(self, name, value, step=0, **kwargs) -> None:
        for writer in self._writer:
            writer.add_scalar(name, value, step, **kwargs)

    def add_scalars(self, name, value_dict, step=0, **kwargs) -> None:
        for writer in self._writer:
            writer.add_scalars(name, value_dict, step, **kwargs)

    def close(self):
        for writer in self._writer:
            writer.close()
