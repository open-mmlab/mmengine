# Copyright (c) OpenMMLab. All rights reserved.
import os
import time
from abc import ABCMeta, abstractmethod
from os import PathLike
from typing import Any, List, Optional, Union

import cv2
import numpy as np
import torch

from mmengine.data import BaseDataSample
from mmengine.fileio import dump
from mmengine.logging import BaseGlobalAccessible
from mmengine.registry import VISUALIZERS, WRITERS
from mmengine.utils import TORCH_VERSION
from .visualizer import Visualizer


class BaseWriter(metaclass=ABCMeta):
    """Base class for writer.

    Args:
        visualizer (dict, :obj:`Visualizer`, optional):
            Visualizer instance or dictionary. Default None.
        save_dir (str, optional): The root path of the save file
            for write data. Default None.
    """

    def __init__(self,
                 visualizer: Optional[Union[dict, 'Visualizer']] = None,
                 save_dir: Optional[Union[str, PathLike[str]]] = None):
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
    def visualizer(self) -> 'Visualizer':
        """Return the visualizer object."""
        return self._visualizer  # type: ignore

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this writer."""
        pass

    def add_hyperparams(self, params_dict: dict, **kwargs) -> None:
        """Add a set of hyperparameters.

        Args:
            params_dict (dict): Each key-value pair in the dictionary is the
                  name of the hyper parameter and it's corresponding value.
                  The type of the value can be one of `bool`, `string`,
                   `float`, `int`, or `None`.
        """
        pass

    def add_graph(self,
                  model: torch.nn.Module,
                  input_array: Optional[Union[torch.Tensor,
                                              List[torch.Tensor]]] = None,
                  **kwargs) -> None:
        """Record graph.

        Args:
            model (torch.nn.Module): Model to draw.
            input_array (torch.Tensor or list of torch.Tensor): A variable
                or a tuple of variables to be fed.
        """
        pass

    def add_image(self,
                  name: str,
                  image: Union[torch.Tensor, np.ndarray],
                  data_samples: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image.

        Args:
            name (str): The unique identifier for the image to save.
            image (torch.Tensor, np.ndarray): The image to be saved in
                CHW format.
            data_samples (:obj:`BaseDataSample`,optional): The data structure
                of OpenMMlab.
            draw_gt (bool): Whether to draw the ground truth. Default True.
            draw_pred (bool): Whether to draw the predicted result.
                Default True.
            step (int): Global step value to record. Default 0.
        """
        pass

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record scalar data.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float or int): Value to save.
            step (int): Global step value to record. Default 0.
        """
        pass

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_name: Optional[str] = None,
                    **kwargs) -> None:
        """Record scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default 0.
            file_name (str, optional): The filename where you want to
                save the data additionally. Default None.
        """
        pass

    def close(self) -> None:
        """close an opened object."""
        pass


@WRITERS.register_module()
class LocalWriter(BaseWriter):
    """Local write class. It can write images, hyperparameters, scalars, etc.
    to the local hard disk.

    Examples:
        >>> from mmengine.visualization import LocalWriter
        >>> import numpy as np
        >>> local_writer = LocalWriter(dict(type='Visualizer'),
        save_dir='temp_dir')
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> local_writer.add_image('img', img)
        >>> local_writer.add_scaler('mAP', 0.6)
        >>> local_writer.add_scalars({'loss': [1, 2, 3],'acc':0.8})

    Args:
        visualizer (dict, :obj:`Visualizer`): Visualizer instance or
            dictionary.
        save_dir (str, optional): The root path of the save file
            for write data.
        save_img_folder (str): The save image folder name.
            Default 'writer_image'.
        save_hyperparams_name (str): The save hyperparam file name.
            Default 'hyperparams.yaml'.
        save_scalar_name (str):  The save scalar values file name.
            Default 'scalars.json'.
        img_show (bool): Whether to show the image when calling add_image.
            Default False.
    """

    def __init__(self,
                 visualizer: Optional[Union[dict, 'Visualizer']],
                 save_dir: Union[str, PathLike[str]],
                 save_img_folder: str = 'writer_image',
                 save_hyperparams_name: str = 'hyperparams.yaml',
                 save_scalar_name: str = 'scalars.json',
                 img_show: bool = False):
        assert save_dir is not None
        assert save_hyperparams_name.split('.')[-1] == 'yaml'
        assert save_scalar_name.split('.')[-1] == 'json'
        super(LocalWriter, self).__init__(visualizer, save_dir)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self._save_dir = os.path.join(
            self._save_dir, f'write_data_{timestamp}')  # type: ignore
        os.makedirs(self._save_dir, exist_ok=True)
        self._save_img_folder = os.path.join(self._save_dir, save_img_folder)
        self._save_scalar_name = os.path.join(self._save_dir, save_scalar_name)
        self._save_hyperparams_name = os.path.join(self._save_dir,
                                                   save_hyperparams_name)
        self._img_show = img_show

    @property
    def experiment(self) -> 'LocalWriter':
        """Return the experiment object associated with this writer."""
        return self

    def add_hyperparams(self, params_dict: dict, **kwargs) -> None:
        """Record hyperparameters.

        Args:
            params_dict (dict): The dict of hyperparameters to save.
        """
        self._dump(self._save_hyperparams_name, 'yaml', params_dict)

    def add_image(self,
                  name: str,
                  image: Union[torch.Tensor, np.ndarray],
                  data_samples: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image.

        Args:
            name (str): The unique identifier for the image to save.
            image (torch.Tensor, np.ndarray): The image to be saved
                in CHW format.
            data_samples (:obj:`BaseDataSample`,optional): The data structure
                of OpenMMlab.
            draw_gt (bool): Whether to draw the ground truth. Default True.
            draw_pred (bool): Whether to draw the predicted result.
                Default True.
            step (int): Global step value to record. Default 0.
        """

        assert self.visualizer
        self.visualizer.draw(data_samples, image, draw_gt, draw_pred)
        if self._img_show:
            self.visualizer.show()
        else:
            os.makedirs(self._save_img_folder, exist_ok=True)
            save_file_name = f'{name}_{step}.png'
            cv2.imwrite(
                os.path.join(self._save_img_folder, save_file_name),
                self.visualizer.get_image())

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Add scalar data to summary.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float or int): Value to save.
            step (int): Global step value to record. Default 0.
        """
        self._dump(self._save_scalar_name, 'json', {name: value, 'step': step})

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_name: Optional[str] = None,
                    **kwargs) -> None:
        """Record scalars. The scalar dict will be written to the default and
        specified files if 'file_name' is specified.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default 0.
            file_name (str, optional): The filename where you want
                to save the data additionally. Default None.
        """
        assert isinstance(scalar_dict, dict)
        scalar_dict.setdefault('step', step)
        if file_name is not None:
            assert file_name.split('.')[-1] == 'json'
            save_file_name = os.path.join(
                self._save_dir,  # type: ignore
                file_name)
            self._dump(save_file_name, 'json', scalar_dict)
        self._dump(self._save_scalar_name, 'json', scalar_dict)

    def _dump(self, file_name: str, file_format: str,
              value_dict: dict) -> None:
        """dump dict to file.

        Args:
           file_name (str): The file name to save data.
           file_format (str): The file format to save data.
           value_dict (dict) : Save dict data.
        """
        with open(file_name, 'a+') as f:
            dump(value_dict, f, file_format=file_format)
            f.write('\n')


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
        """Return Wandb object."""
        return self._wandb

    def _setup_env(self, init_kwargs: dict):
        """Setup env.

        Args:
            init_kwargs (dict): The init args.

        Return:
            :obj:`wandb`
        """
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

    def add_hyperparams(self, params_dict: dict, **kwargs) -> None:
        """Add a set of hyperparameters to be compared in Wandb.

        Args:
            params_dict (dict): Each key-value pair in the dictionary
                is the name of the hyper parameter and it's
                corresponding value. The type of the value can be
                one of `bool`, `string`, `float`, `int`, or `None`.
        """
        self._wandb.log(params_dict, commit=self._commit, sync=self._sync)

    def add_image(self,
                  name: str,
                  image: Union[torch.Tensor, np.ndarray],
                  data_samples: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image to Wandb.

        Args:
            name (str): The unique identifier for the image to save.
            image (torch.Tensor, np.ndarray): The image to be saved in
                CHW format.
            data_samples (:obj:`BaseDataSample`,optional): The data structure
                of OpenMMlab.
            draw_gt (bool): Whether to draw the ground truth. Default True.
            draw_pred (bool): Whether to draw the predicted result.
                Default True.
            step (int): Global step value to record. Default 0.
        """
        if self.visualizer:
            self.visualizer.draw(data_samples, image, draw_gt, draw_pred)
            self._wandb.log({name: self.visualizer.get_image()},
                            commit=self._commit,
                            step=step,
                            sync=self._sync)
        else:
            self.add_image_to_wandb(name, image, data_samples, draw_gt,
                                    draw_pred, step, **kwargs)

    def add_scalar(self, name, value, step=0, **kwargs) -> None:
        self._wandb.log({name: value},
                        commit=self._commit,
                        step=step,
                        sync=self._sync)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_name: Optional[str] = None,
                    **kwargs) -> None:
        """Add scalars data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default 0.
            file_name (str, optional): Useless parameter. Just for
                interface unification. Default None.
        """
        self._wandb.log(
            scalar_dict, commit=self._commit, step=step, sync=self._sync)

    def add_image_to_wandb(self,
                           name,
                           image,
                           data_samples,
                           show_gt=True,
                           show_pred=True,
                           step=0,
                           **kwargs):
        raise NotImplementedError()

    def close(self) -> None:
        """close an opened wandb object."""
        if hasattr(self, '_wandb'):
            self._wandb.join()


@WRITERS.register_module()
class TensorboardWriter(BaseWriter):
    """Tensorboard write class. It can write images, hyperparameters, scalars,
    etc. to a tensorboard file.

    Args:
        visualizer (dict, :obj:`Visualizer`): Visualizer instance
            or dictionary.
        save_dir (str, optional): The root path of the save file
            for write data.
        log_dir (str): Save directory location. Default 'tf_writer'.
    """

    def __init__(self,
                 visualizer: Optional[Union[dict, 'Visualizer']],
                 save_dir: Union[str, PathLike[str]],
                 log_dir: str = 'tf_writer'):
        assert save_dir is not None
        super(TensorboardWriter, self).__init__(visualizer, save_dir)
        self._tensorboard = self._setup_env(log_dir)

    def _setup_env(self, log_dir: str):
        """Setup env.

        Args:
            log_dir (str): Save directory location. Default 'tf_writer'.

        Return:
            :obj:`SummaryWriter`
        """
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

        self.log_dir = os.path.join(self._save_dir, log_dir)  # type: ignore
        return SummaryWriter(self.log_dir)

    @property
    def experiment(self):
        """Return Tensorboard object."""
        return self._tensorboard

    def add_graph(self,
                  model: torch.nn.Module,
                  input_array: Optional[Union[torch.Tensor,
                                              List[torch.Tensor]]] = None,
                  **kwargs) -> None:
        """Add graph data to tensorboard.

        Args:
            model (torch.nn.Module): Model to draw.
            input_array (torch.Tensor or list of torch.Tensor): A variable
                or a tuple of variables to be fed.
        """
        if isinstance(input_array, list):
            for array in input_array:
                assert array.ndim == 4
        else:
            assert input_array and input_array.ndim == 4
        self._tensorboard.add_graph(model, input_array)

    def add_hyperparams(self, params_dict: dict, **kwargs) -> None:
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            params_dict (dict): Each key-value pair in the dictionary is the
                  name of the hyper parameter and it's corresponding value.
                  The type of the value can be one of `bool`, `string`,
                   `float`, `int`, or `None`.
        """
        self._tensorboard.add_hparams(params_dict, {})

    def add_image(self,
                  name: str,
                  image: Union[torch.Tensor, np.ndarray],
                  data_samples: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image to tensorboard.

        Args:
            name (str): The unique identifier for the image to save.
            image (torch.Tensor, np.ndarray): The image to be saved in
                CHW format.
            data_samples (:obj:`BaseDataSample`,optional): The data structure
                of OpenMMlab.
            draw_gt (bool): Whether to draw the ground truth. Default True.
            draw_pred (bool): Whether to draw the predicted result.
                Default True.
            step (int): Global step value to record. Default 0.
        """
        assert self.visualizer
        self.visualizer.draw(data_samples, image, draw_gt, draw_pred)
        self._tensorboard.add_image(
            name, self.visualizer.get_image(), step, dataformats='HWC')

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Add scalar data to summary.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float or int): Value to save.
            step (int): Global step value to record. Default 0.
        """
        self._tensorboard.add_scalar(name, value, step)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_name: Optional[str] = None,
                    **kwargs) -> None:
        """Add scalars data to summary.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default 0.
            file_name (str, optional): Useless parameter. Just for
                interface unification. Default None.
        """
        assert isinstance(scalar_dict, dict)
        for key, value in scalar_dict.items():
            if 'step' != key:
                self.add_scalar(key, value, step)

    def close(self):
        """close an opened tensorboard object."""
        if hasattr(self, '_tensorboard'):
            self._tensorboard.close()


class ComposedWriter(BaseGlobalAccessible):
    """Wrapper class to compose multiple a subclass of :class:`BaseWriter`
    instances. By inheriting BaseGlobalAccessible, it can be accessed anywhere
    once instantiated.

    Args:
        name (str): The name of the instance. Defaults to 'composed_writer'.
        writers (list, optional): The writers to compose.
    """

    def __init__(self,
                 name: str = 'composed_writer',
                 writers: Optional[List[Union[dict, 'BaseWriter']]] = None):
        super(ComposedWriter, self).__init__(name)
        self._writer = []
        if writers is not None:
            assert isinstance(writers, list)
            for writer in writers:
                if isinstance(writer, dict):
                    self._writer.append(WRITERS.build(writer))
                else:
                    assert isinstance(writer, BaseWriter), \
                        f'writer should be an instance of a subclass of ' \
                        f'BaseWriter, but got {type(writer)}'
                    self._writer.append(writer)

    def get_writer(self, index: int) -> 'BaseWriter':
        """Returns the writer object corresponding to the specified index."""
        return self._writer[index]

    def get_experiment(self, index: int) -> Any:
        """Returns the writer experiment object corresponding to the specified
        index."""
        return self._writer[index].experiment

    def add_hyperparams(self, params_dict: dict, **kwargs):
        """Record hyperparameters.

        Args:
            params_dict (dict): The dictionary of hyperparameters to save.
        """
        for writer in self._writer:
            writer.add_hyperparams(params_dict, **kwargs)

    def add_graph(self,
                  model: torch.nn.Module,
                  input_array: Optional[Union[torch.Tensor,
                                              List[torch.Tensor]]] = None,
                  **kwargs) -> None:
        """Record graph data.

        Args:
            model (torch.nn.Module): Model to draw.
            input_array (torch.Tensor or list of torch.Tensor): A variable
                or a tuple of variables to be fed.
        """
        for writer in self._writer:
            writer.add_graph(model, input_array, **kwargs)

    def add_image(self,
                  name: str,
                  image: Union[torch.Tensor, np.ndarray],
                  data_samples: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image to tensorboard.

        Args:
            name (str): The unique identifier for the image to save.
            image (torch.Tensor, np.ndarray): The image to be saved
                in CHW format.
            data_samples (:obj:`BaseDataSample`,optional): The data structure
                of OpenMMlab.
            draw_gt (bool): Whether to draw the ground truth. Default True.
            draw_pred (bool): Whether to draw the predicted result.
                Default True.
            step (int): Global step value to record. Default 0.
        """
        for writer in self._writer:
            writer.add_image(name, image, data_samples, draw_gt, draw_pred,
                             step, **kwargs)

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record scalar data.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float or int): Value to save.
            step (int): Global step value to record. Default 0.
        """
        for writer in self._writer:
            writer.add_scalar(name, value, step, **kwargs)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_name: Optional[str] = None,
                    **kwargs) -> None:
        """Record scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default 0.
            file_name (str, optional): The filename where you want to
                save the data additionally. Default None.
        """
        for writer in self._writer:
            writer.add_scalars(scalar_dict, step, file_name, **kwargs)

    def close(self) -> None:
        """close an opened object."""
        for writer in self._writer:
            writer.close()
