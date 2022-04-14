# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import time
from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np
import torch

from mmengine.config import Config
from mmengine.fileio import dump
from mmengine.registry import VISBACKENDS
from mmengine.utils import TORCH_VERSION


class BaseVisBackend(metaclass=ABCMeta):
    """Base class for vis backend.

    All backends must inherit ``BaseVisBackend`` and implement
    the required functions.

    Args:
        save_dir (str, optional): The root directory to save
            the files produced by the backend. Default to None.
    """

    def __init__(self, save_dir: Optional[str] = None):
        self._save_dir = save_dir
        if self._save_dir:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            self._save_dir = osp.join(self._save_dir,
                                      f'vis_data_{timestamp}')  # type: ignore

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this writer.

        The experiment attribute can get the visualizer backend, such as wandb,
        tensorboard. If you want to write other data, such as writing a table,
        you can directly get the visualizer backend through experiment.
        """
        pass

    def add_config(self, config: Config, **kwargs) -> None:
        """Record a set of parameters.

        Args:
            config (Config): The Config object
        """
        pass

    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record graph.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        pass

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
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


@VISBACKENDS.register_module()
class LocalVisBackend(BaseVisBackend):
    """Local vis backend class.

    It can write image, config, scalars, etc.
    to the local hard disk. You can get the drawing backend
    through the visualizer property for custom drawing.

    Examples:
        >>> from mmengine.visualization import LocalVisBackend
        >>> import numpy as np
        >>> local_vis_backend = LocalVisBackend(save_dir='temp_dir')
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> local_vis_backend.add_image('img', img)
        >>> local_vis_backend.add_scaler('mAP', 0.6)
        >>> local_vis_backend.add_scalars({'loss': [1, 2, 3], 'acc': 0.8})
        >>> local_vis_backend.add_image('img', image)

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the writer. If it is none, it means no data
            is stored. Default None.
        img_save_dir (str): The directory to save images.
            Default to 'writer_image'.
        config_save_file (str): The file to save parameters.
            Default to 'parameters.yaml'.
        scalar_save_file (str):  The file to save scalar values.
            Default to 'scalars.json'.
    """

    def __init__(self,
                 save_dir: Optional[str] = None,
                 img_save_dir: str = 'vis_image',
                 config_save_file: str = 'config.py',
                 scalar_save_file: str = 'scalars.json'):
        assert config_save_file.split('.')[-1] == 'py'
        assert scalar_save_file.split('.')[-1] == 'json'
        super(LocalVisBackend, self).__init__(save_dir)
        if self._save_dir is not None:
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
            self._img_save_dir = osp.join(
                self._save_dir,  # type: ignore
                img_save_dir)
            self._scalar_save_file = osp.join(
                self._save_dir,  # type: ignore
                scalar_save_file)
            self._config_save_file = osp.join(
                self._save_dir,  # type: ignore
                config_save_file)

    @property
    def experiment(self) -> 'LocalVisBackend':
        """Return the experiment object associated with this visualizer
        backend."""
        return self

    def add_config(self, config: Config, **kwargs) -> None:
        # TODO
        assert isinstance(config, Config)

    def add_image(self,
                  name: str,
                  image: np.ndarray = None,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image to disk.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """

        drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        os.makedirs(self._img_save_dir, exist_ok=True)
        save_file_name = f'{name}_{step}.png'
        cv2.imwrite(osp.join(self._img_save_dir, save_file_name), drawn_image)

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


@VISBACKENDS.register_module()
class WandbVisBackend(BaseVisBackend):
    """Write various types of data to wandb.

    Examples:
        >>> from mmengine.visualization import WandbVisBackend
        >>> import numpy as np
        >>> wandb_vis_backend = WandbVisBackend()
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> wandb_vis_backend.add_image('img', img)
        >>> wandb_vis_backend.add_scaler('mAP', 0.6)
        >>> wandb_vis_backend.add_scalars({'loss': [1, 2, 3],'acc': 0.8})
        >>> wandb_vis_backend.add_image('img', img)

    Args:
        init_kwargs (dict, optional): wandb initialization
            input parameters. Default to None.
        commit: (bool, optional) Save the metrics dict to the wandb server
                and increment the step.  If false `wandb.log` just
                updates the current metrics dict with the row argument
                and metrics won't be saved until `wandb.log` is called
                with `commit=True`. Default to True.
        save_dir (str, optional): The root directory to save the files
            produced by the writer. Default to None.
    """

    def __init__(self,
                 init_kwargs: Optional[dict] = None,
                 commit: Optional[bool] = True,
                 save_dir: Optional[str] = None):
        super(WandbVisBackend, self).__init__(save_dir)
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

    def add_config(self, config: Config, **kwargs) -> None:
        # TODO
        pass

    def add_image(self,
                  name: str,
                  image: np.ndarray = None,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image to wandb.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        self._wandb.log({name: image}, commit=self._commit, step=step)

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

    def close(self) -> None:
        """close an opened wandb object."""
        if hasattr(self, '_wandb'):
            self._wandb.join()


@VISBACKENDS.register_module()
class TensorboardVisBackend(BaseVisBackend):
    """Tensorboard class. It can write images, config, scalars, etc. to a
    tensorboard file.

    Its drawing function is provided by Visualizer.

    Examples:
        >>> from mmengine.visualization import TensorboardVisBackend
        >>> import numpy as np
        >>> tensorboard_visualizer = TensorboardVisBackend(save_dir='temp_dir')
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> tensorboard_visualizer.add_image('img', img)
        >>> tensorboard_visualizer.add_scaler('mAP', 0.6)
        >>> tensorboard_visualizer.add_scalars({'loss': 0.1,'acc':0.8})
        >>> tensorboard_visualizer.add_image('img', image)

    Args:
        save_dir (str): The root directory to save the files
            produced by the backend.
        log_dir (str): Save directory location. Default to 'tf_logs'.
    """

    def __init__(self,
                 save_dir: Optional[str] = None,
                 log_dir: str = 'tf_logs'):
        super(TensorboardVisBackend, self).__init__(save_dir)
        if save_dir is not None:
            self._tensorboard = self._setup_env(log_dir)

    def _setup_env(self, log_dir: str):
        """Setup env.

        Args:
            log_dir (str): Save directory location.

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
        if self._save_dir is None:
            return SummaryWriter(f'./{log_dir}')
        else:
            self.log_dir = osp.join(self._save_dir, log_dir)  # type: ignore
            return SummaryWriter(self.log_dir)

    @property
    def experiment(self):
        """Return Tensorboard object."""
        return self._tensorboard

    def add_config(self, config: Config, **kwargs) -> None:
        # TODO
        pass

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record image to tensorboard.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        self._tensorboard.add_image(name, image, step, dataformats='HWC')

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
