# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time
from abc import ABCMeta, abstractmethod
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

    Each writer can inherit ``BaseWriter`` and implement
    the required functions.

    Args:
        visualizer (dict, :obj:`Visualizer`, optional):
            Visualizer instance or dictionary. Default to None.
        save_dir (str, optional): The root directory to save
            the files produced by the writer. Default to None.
    """

    def __init__(self,
                 visualizer: Optional[Union[dict, 'Visualizer']] = None,
                 save_dir: Optional[str] = None):
        self._save_dir = save_dir
        if self._save_dir:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            self._save_dir = osp.join(
                self._save_dir, f'write_data_{timestamp}')  # type: ignore
        self._visualizer = visualizer
        if visualizer:
            if isinstance(visualizer, dict):
                self._visualizer = VISUALIZERS.build(visualizer)
            else:
                assert isinstance(visualizer, Visualizer), \
                    'visualizer should be an instance of Visualizer, ' \
                    f'but got {type(visualizer)}'

    @property
    def visualizer(self) -> 'Visualizer':
        """Return the visualizer object.

        You can get the drawing backend through the visualizer property for
        custom drawing.
        """
        return self._visualizer  # type: ignore

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this writer.

        The experiment attribute can get the write backend, such as wandb,
        tensorboard. If you want to write other data, such as writing a table,
        you can directly get the write backend through experiment.
        """
        pass

    def add_params(self, params_dict: dict, **kwargs) -> None:
        """Record a set of parameters.

        Args:
            params_dict (dict): Each key-value pair in the dictionary is the
                  name of the parameters and it's corresponding value.
        """
        pass

    def add_graph(self, model: torch.nn.Module,
                  input_tensor: Union[torch.Tensor,
                                      List[torch.Tensor]], **kwargs) -> None:
        """Record graph.

        Args:
            model (torch.nn.Module): Model to draw.
            input_tensor (torch.Tensor, list[torch.Tensor]): A variable
                or a tuple of variables to be fed.
        """
        pass

    def add_image(self,
                  name: str,
                  image: Optional[np.ndarray] = None,
                  gt_sample: Optional['BaseDataSample'] = None,
                  pred_sample: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            gt_sample (:obj:`BaseDataSample`, optional): The ground truth data
                structure of OpenMMlab. Default to None.
            pred_sample (:obj:`BaseDataSample`, optional): The predicted result
                data structure of OpenMMlab. Default to None.
            draw_gt (bool): Whether to draw the ground truth. Default: True.
            draw_pred (bool): Whether to draw the predicted result.
                Default to True.
            step (int): Global step value to record. Default to 0.
        """
        pass

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record scalar.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float, int): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        pass

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Default to None.
        """
        pass

    def close(self) -> None:
        """close an opened object."""
        pass


@WRITERS.register_module()
class LocalWriter(BaseWriter):
    """Local write class.

    It can write image, hyperparameters, scalars, etc.
    to the local hard disk. You can get the drawing backend
    through the visualizer property for custom drawing.

    Examples:
        >>> from mmengine.visualization import LocalWriter
        >>> import numpy as np
        >>> local_writer = LocalWriter(dict(type='DetVisualizer'),\
            save_dir='temp_dir')
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> local_writer.add_image('img', img)
        >>> local_writer.add_scaler('mAP', 0.6)
        >>> local_writer.add_scalars({'loss': [1, 2, 3], 'acc': 0.8})
        >>> local_writer.add_params(dict(lr=0.1, mode='linear'))

        >>> local_writer.visualizer.draw_bboxes(np.array([0, 0, 1, 1]), \
            edgecolors='g')
        >>> local_writer.add_image('img', \
            local_writer.visualizer.get_image())

    Args:
        save_dir (str): The root directory to save the files
            produced by the writer.
        visualizer (dict, :obj:`Visualizer`, optional): Visualizer
            instance or dictionary. Default to None
        img_save_dir (str): The directory to save images.
            Default to 'writer_image'.
        params_save_file (str): The file to save parameters.
            Default to 'parameters.yaml'.
        scalar_save_file (str):  The file to save scalar values.
            Default to 'scalars.json'.
        img_show (bool): Whether to show the image when calling add_image.
            Default to False.
    """

    def __init__(self,
                 save_dir: str,
                 visualizer: Optional[Union[dict, 'Visualizer']] = None,
                 img_save_dir: str = 'writer_image',
                 params_save_file: str = 'parameters.yaml',
                 scalar_save_file: str = 'scalars.json',
                 img_show: bool = False):
        assert params_save_file.split('.')[-1] == 'yaml'
        assert scalar_save_file.split('.')[-1] == 'json'
        super(LocalWriter, self).__init__(visualizer, save_dir)
        os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        self._img_save_dir = osp.join(
            self._save_dir,  # type: ignore
            img_save_dir)
        self._scalar_save_file = osp.join(
            self._save_dir,  # type: ignore
            scalar_save_file)
        self._params_save_file = osp.join(
            self._save_dir,  # type: ignore
            params_save_file)
        self._img_show = img_show

    @property
    def experiment(self) -> 'LocalWriter':
        """Return the experiment object associated with this writer."""
        return self

    def add_params(self, params_dict: dict, **kwargs) -> None:
        """Record parameters to disk.

        Args:
            params_dict (dict): The dict of parameters to save.
        """
        assert isinstance(params_dict, dict)
        self._dump(params_dict, self._params_save_file, 'yaml')

    def add_image(self,
                  name: str,
                  image: Optional[np.ndarray] = None,
                  gt_sample: Optional['BaseDataSample'] = None,
                  pred_sample: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image to disk.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            gt_sample (:obj:`BaseDataSample`, optional): The ground truth data
                structure of OpenMMlab. Default to None.
            pred_sample (:obj:`BaseDataSample`, optional): The predicted result
                data structure of OpenMMlab. Default to None.
            draw_gt (bool): Whether to draw the ground truth. Default to True.
            draw_pred (bool): Whether to draw the predicted result.
                Default to True.
            step (int): Global step value to record. Default to 0.
        """
        assert self.visualizer, 'Please instantiate the visualizer ' \
                                'object with initialization parameters.'
        self.visualizer.draw(image, gt_sample, pred_sample, draw_gt, draw_pred)
        if self._img_show:
            self.visualizer.show()
        else:
            drawn_image = cv2.cvtColor(self.visualizer.get_image(),
                                       cv2.COLOR_RGB2BGR)
            os.makedirs(self._img_save_dir, exist_ok=True)
            save_file_name = f'{name}_{step}.png'
            cv2.imwrite(
                osp.join(self._img_save_dir, save_file_name), drawn_image)

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Add scalar data to disk.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float, int): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        self._dump({name: value, 'step': step}, self._scalar_save_file, 'json')

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record scalars. The scalar dict will be written to the default and
        specified files if ``file_name`` is specified.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the ``file_path`` file at the same time
                if the ``file_path`` parameter is specified.
                Default to None.
        """
        assert isinstance(scalar_dict, dict)
        scalar_dict.setdefault('step', step)
        if file_path is not None:
            assert file_path.split('.')[-1] == 'json'
            new_save_file_path = osp.join(
                self._save_dir,  # type: ignore
                file_path)
            assert new_save_file_path != self._scalar_save_file, \
                '"file_path" and "scalar_save_file" have the same name, ' \
                'please set "file_path" to another value'
            self._dump(scalar_dict, new_save_file_path, 'json')
        self._dump(scalar_dict, self._scalar_save_file, 'json')

    def _dump(self, value_dict: dict, file_path: str,
              file_format: str) -> None:
        """dump dict to file.

        Args:
           value_dict (dict) : Save dict data.
           file_path (str): The file path to save data.
           file_format (str): The file format to save data.
        """
        with open(file_path, 'a+') as f:
            dump(value_dict, f, file_format=file_format)
            f.write('\n')


@WRITERS.register_module()
class WandbWriter(BaseWriter):
    """Write various types of data to wandb.

    Examples:
        >>> from mmengine.visualization import WandbWriter
        >>> import numpy as np
        >>> wandb_writer = WandbWriter(dict(type='DetVisualizer'))
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> wandb_writer.add_image('img', img)
        >>> wandb_writer.add_scaler('mAP', 0.6)
        >>> wandb_writer.add_scalars({'loss': [1, 2, 3],'acc': 0.8})
        >>> wandb_writer.add_params(dict(lr=0.1, mode='linear'))

        >>> wandb_writer.visualizer.draw_bboxes(np.array([0, 0, 1, 1]), \
            edgecolors='g')
        >>> wandb_writer.add_image('img', \
            wandb_writer.visualizer.get_image())

        >>> wandb_writer = WandbWriter()
        >>> assert wandb_writer.visualizer is None
        >>> wandb_writer.add_image('img', img)

    Args:
        init_kwargs (dict, optional): wandb initialization
            input parameters. Default to None.
        commit: (bool, optional) Save the metrics dict to the wandb server
                and increment the step.  If false `wandb.log` just
                updates the current metrics dict with the row argument
                and metrics won't be saved until `wandb.log` is called
                with `commit=True`. Default to True.
        visualizer (dict, :obj:`Visualizer`, optional):
            Visualizer instance or dictionary. Default to None.
        save_dir (str, optional): The root directory to save the files
            produced by the writer. Default to None.
    """

    def __init__(self,
                 init_kwargs: Optional[dict] = None,
                 commit: Optional[bool] = True,
                 visualizer: Optional[Union[dict, 'Visualizer']] = None,
                 save_dir: Optional[str] = None):
        super(WandbWriter, self).__init__(visualizer, save_dir)
        self._commit = commit
        self._wandb = self._setup_env(init_kwargs)

    @property
    def experiment(self):
        """Return wandb object.

        The experiment attribute can get the wandb backend, If you want to
        write other data, such as writing a table, you can directly get the
        wandb backend through experiment.
        """
        return self._wandb

    def _setup_env(self, init_kwargs: Optional[dict] = None) -> Any:
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

    def add_params(self, params_dict: dict, **kwargs) -> None:
        """Record a set of parameters to be compared in wandb.

        Args:
            params_dict (dict): Each key-value pair in the dictionary
                is the name of the parameters and it's
                corresponding value.
        """
        assert isinstance(params_dict, dict)
        self._wandb.log(params_dict, commit=self._commit)

    def add_image(self,
                  name: str,
                  image: Optional[np.ndarray] = None,
                  gt_sample: Optional['BaseDataSample'] = None,
                  pred_sample: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image to wandb.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            gt_sample (:obj:`BaseDataSample`, optional): The ground truth data
                structure of OpenMMlab. Default to None.
            pred_sample (:obj:`BaseDataSample`, optional): The predicted result
                data structure of OpenMMlab. Default to None.
            draw_gt (bool): Whether to draw the ground truth. Default: True.
            draw_pred (bool): Whether to draw the predicted result.
                Default to True.
            step (int): Global step value to record. Default to 0.
        """
        if self.visualizer:
            self.visualizer.draw(image, gt_sample, pred_sample, draw_gt,
                                 draw_pred)
            self._wandb.log({name: self.visualizer.get_image()},
                            commit=self._commit,
                            step=step)
        else:
            self.add_image_to_wandb(name, image, gt_sample, pred_sample,
                                    draw_gt, draw_pred, step, **kwargs)

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record scalar data to wandb.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float, int): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        self._wandb.log({name: value}, commit=self._commit, step=step)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record scalar's data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Default to None.
        """
        self._wandb.log(scalar_dict, commit=self._commit, step=step)

    def add_image_to_wandb(self,
                           name: str,
                           image: np.ndarray,
                           gt_sample: Optional['BaseDataSample'] = None,
                           pred_sample: Optional['BaseDataSample'] = None,
                           draw_gt: bool = True,
                           draw_pred: bool = True,
                           step: int = 0,
                           **kwargs) -> None:
        """Record image to wandb.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray): The image to be saved. The format
                should be BGR.
            gt_sample (:obj:`BaseDataSample`, optional): The ground truth data
                structure of OpenMMlab. Default to None.
            pred_sample (:obj:`BaseDataSample`, optional): The predicted result
                data structure of OpenMMlab. Default to None.
            draw_gt (bool): Whether to draw the ground truth. Default to True.
            draw_pred (bool): Whether to draw the predicted result.
                Default to True.
            step (int): Global step value to record. Default to 0.
        """
        raise NotImplementedError()

    def close(self) -> None:
        """close an opened wandb object."""
        if hasattr(self, '_wandb'):
            self._wandb.join()


@WRITERS.register_module()
class TensorboardWriter(BaseWriter):
    """Tensorboard write class. It can write images, hyperparameters, scalars,
    etc. to a tensorboard file.

    Its drawing function is provided by Visualizer.

    Examples:
        >>> from mmengine.visualization import TensorboardWriter
        >>> import numpy as np
        >>> tensorboard_writer = TensorboardWriter(dict(type='DetVisualizer'),\
            save_dir='temp_dir')
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> tensorboard_writer.add_image('img', img)
        >>> tensorboard_writer.add_scaler('mAP', 0.6)
        >>> tensorboard_writer.add_scalars({'loss': 0.1,'acc':0.8})
        >>> tensorboard_writer.add_params(dict(lr=0.1, mode='linear'))

        >>> tensorboard_writer.visualizer.draw_bboxes(np.array([0, 0, 1, 1]), \
            edgecolors='g')
        >>> tensorboard_writer.add_image('img', \
            tensorboard_writer.visualizer.get_image())

    Args:
        save_dir (str): The root directory to save the files
            produced by the writer.
        visualizer (dict, :obj:`Visualizer`, optional): Visualizer instance
            or dictionary. Default to None.
        log_dir (str): Save directory location. Default to 'tf_writer'.
    """

    def __init__(self,
                 save_dir: str,
                 visualizer: Optional[Union[dict, 'Visualizer']] = None,
                 log_dir: str = 'tf_logs'):
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

        self.log_dir = osp.join(self._save_dir, log_dir)  # type: ignore
        return SummaryWriter(self.log_dir)

    @property
    def experiment(self):
        """Return Tensorboard object."""
        return self._tensorboard

    def add_graph(self, model: torch.nn.Module,
                  input_tensor: Union[torch.Tensor,
                                      List[torch.Tensor]], **kwargs) -> None:
        """Record graph data to tensorboard.

        Args:
            model (torch.nn.Module): Model to draw.
            input_tensor (torch.Tensor, list[torch.Tensor]): A variable
                or a tuple of variables to be fed.
        """
        if isinstance(input_tensor, list):
            for array in input_tensor:
                assert array.ndim == 4
                assert isinstance(array, torch.Tensor)
        else:
            assert isinstance(input_tensor,
                              torch.Tensor) and input_tensor.ndim == 4
        self._tensorboard.add_graph(model, input_tensor)

    def add_params(self, params_dict: dict, **kwargs) -> None:
        """Record a set of hyperparameters to be compared in TensorBoard.

        Args:
            params_dict (dict): Each key-value pair in the dictionary is the
                  name of the hyper parameter and it's corresponding value.
                  The type of the value can be one of `bool`, `string`,
                   `float`, `int`, or `None`.
        """
        assert isinstance(params_dict, dict)
        self._tensorboard.add_hparams(params_dict, {})

    def add_image(self,
                  name: str,
                  image: Optional[np.ndarray] = None,
                  gt_sample: Optional['BaseDataSample'] = None,
                  pred_sample: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image to tensorboard.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            gt_sample (:obj:`BaseDataSample`, optional): The ground truth data
                structure of OpenMMlab. Default to None.
            pred_sample (:obj:`BaseDataSample`, optional): The predicted result
                data structure of OpenMMlab. Default to None.
            draw_gt (bool): Whether to draw the ground truth. Default to True.
            draw_pred (bool): Whether to draw the predicted result.
                Default to True.
            step (int): Global step value to record. Default to 0.
        """
        assert self.visualizer, 'Please instantiate the visualizer ' \
                                'object with initialization parameters.'
        self.visualizer.draw(image, gt_sample, pred_sample, draw_gt, draw_pred)
        self._tensorboard.add_image(
            name, self.visualizer.get_image(), step, dataformats='HWC')

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record scalar data to summary.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float, int): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        self._tensorboard.add_scalar(name, value, step)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record scalar's data to summary.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Default to None.
        """
        assert isinstance(scalar_dict, dict)
        assert 'step' not in scalar_dict, 'Please set it directly ' \
                                          'through the step parameter'
        for key, value in scalar_dict.items():
            self.add_scalar(key, value, step)

    def close(self):
        """close an opened tensorboard object."""
        if hasattr(self, '_tensorboard'):
            self._tensorboard.close()


class ComposedWriter(BaseGlobalAccessible):
    """Wrapper class to compose multiple a subclass of :class:`BaseWriter`
    instances. By inheriting BaseGlobalAccessible, it can be accessed anywhere
    once instantiated.

    Examples:
        >>> from mmengine.visualization import ComposedWriter
        >>> import numpy as np
        >>> composed_writer= ComposedWriter.create_instance( \
            'composed_writer', writers=[dict(type='LocalWriter', \
            visualizer=dict(type='DetVisualizer'), \
            save_dir='temp_dir'), dict(type='WandbWriter')])
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> composed_writer.add_image('img', img)
        >>> composed_writer.add_scalar('mAP', 0.6)
        >>> composed_writer.add_scalars({'loss': 0.1,'acc':0.8})
        >>> composed_writer.add_params(dict(lr=0.1, mode='linear'))

    Args:
        name (str): The name of the instance. Defaults: 'composed_writer'.
        writers (list, optional): The writers to compose. Default to None
    """

    def __init__(self,
                 name: str = 'composed_writer',
                 writers: Optional[List[Union[dict, 'BaseWriter']]] = None):
        super().__init__(name)
        self._writers = []
        if writers is not None:
            assert isinstance(writers, list)
            for writer in writers:
                if isinstance(writer, dict):
                    self._writers.append(WRITERS.build(writer))
                else:
                    assert isinstance(writer, BaseWriter), \
                        f'writer should be an instance of a subclass of ' \
                        f'BaseWriter, but got {type(writer)}'
                    self._writers.append(writer)

    def __len__(self):
        return len(self._writers)

    def get_writer(self, index: int) -> 'BaseWriter':
        """Returns the writer object corresponding to the specified index."""
        return self._writers[index]

    def get_experiment(self, index: int) -> Any:
        """Returns the writer's experiment object corresponding to the
        specified index."""
        return self._writers[index].experiment

    def get_visualizer(self, index: int) -> 'Visualizer':
        """Returns the writer's visualizer object corresponding to the
        specified index."""
        return self._writers[index].visualizer

    def add_params(self, params_dict: dict, **kwargs):
        """Record parameters.

        Args:
            params_dict (dict): The dictionary of parameters to save.
        """
        for writer in self._writers:
            writer.add_params(params_dict, **kwargs)

    def add_graph(self, model: torch.nn.Module,
                  input_array: Union[torch.Tensor,
                                     List[torch.Tensor]], **kwargs) -> None:
        """Record graph data.

        Args:
            model (torch.nn.Module): Model to draw.
            input_array (torch.Tensor, list[torch.Tensor]): A variable
                or a tuple of variables to be fed.
        """
        for writer in self._writers:
            writer.add_graph(model, input_array, **kwargs)

    def add_image(self,
                  name: str,
                  image: Optional[np.ndarray] = None,
                  gt_sample: Optional['BaseDataSample'] = None,
                  pred_sample: Optional['BaseDataSample'] = None,
                  draw_gt: bool = True,
                  draw_pred: bool = True,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            gt_sample (:obj:`BaseDataSample`, optional): The ground truth data
                structure of OpenMMlab. Default to None.
            pred_sample (:obj:`BaseDataSample`, optional): The predicted result
                data structure of OpenMMlab. Default to None.
            draw_gt (bool): Whether to draw the ground truth. Default to True.
            draw_pred (bool): Whether to draw the predicted result.
                Default to True.
            step (int): Global step value to record. Default to 0.
        """
        for writer in self._writers:
            writer.add_image(name, image, gt_sample, pred_sample, draw_gt,
                             draw_pred, step, **kwargs)

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record scalar data.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float, int): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        for writer in self._writers:
            writer.add_scalar(name, value, step, **kwargs)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Default to None.
        """
        for writer in self._writers:
            writer.add_scalars(scalar_dict, step, file_path, **kwargs)

    def close(self) -> None:
        """close an opened object."""
        for writer in self._writers:
            writer.close()
