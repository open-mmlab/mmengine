# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import warnings
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
        # lazy creation for save dir
        self._is_created = False

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """Return the experiment object associated with this vis backend.

        The experiment attribute can get the visualizer backend, such as wandb,
        tensorboard. If you want to write other data, such as writing a table,
        you can directly get the visualizer backend through experiment.
        """
        pass

    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config.

        Args:
            config (Config): The Config object
        """
        pass

    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model graph.

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
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        pass

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar.

        Args:
            name (str): The scalar identifier.
            value (int, float): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        pass

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars' data.

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
    through the experiment property for custom drawing.

    Examples:
        >>> from mmengine.visualization import LocalVisBackend
        >>> import numpy as np
        >>> local_vis_backend = LocalVisBackend(save_dir='temp_dir')
        >>> img = np.random.randint(0, 256, size=(10, 10, 3))
        >>> local_vis_backend.add_image('img', img)
        >>> local_vis_backend.add_scalar('mAP', 0.6)
        >>> local_vis_backend.add_scalars({'loss': [1, 2, 3], 'acc': 0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> local_vis_backend.add_config(cfg)

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer. If it is none, it means no data
            is stored. Default to None.
        img_save_dir (str): The directory to save images.
            Default to 'vis_image'.
        config_save_file (str): The file name to save config.
            Default to 'config.py'.
        scalar_save_file (str):  The file name to save scalar values.
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
        self._img_save_dir = img_save_dir
        self._config_save_file = config_save_file
        self._scalar_save_file = scalar_save_file

    def __mkdir_or_exist(self):
        """If ``save_dir`` is None, the directory will not be created, and
        ``add_xxx`` will have no effect."""
        if self._save_dir is None:
            return False

        if self._is_created is False:
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
            self._img_save_dir = osp.join(
                self._save_dir,  # type: ignore
                self._img_save_dir)
            self._config_save_file = osp.join(
                self._save_dir,  # type: ignore
                self._config_save_file)
            self._scalar_save_file = osp.join(
                self._save_dir,  # type: ignore
                self._scalar_save_file)
            self._is_created = True
        return True

    @property
    def experiment(self) -> 'LocalVisBackend':
        """Return the experiment object associated with this visualizer
        backend."""
        return self

    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to disk.

        Args:
            config (Config): The Config object
        """
        assert isinstance(config, Config)
        if self.__mkdir_or_exist():
            config.dump(self._config_save_file)

    def add_image(self,
                  name: str,
                  image: np.array,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to disk.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        if self.__mkdir_or_exist():
            assert image.dtype == np.uint8
            drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            os.makedirs(self._img_save_dir, exist_ok=True)
            save_file_name = f'{name}_{step}.png'
            cv2.imwrite(
                osp.join(self._img_save_dir, save_file_name), drawn_image)

    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to disk.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        if self.__mkdir_or_exist():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self._dump({
                name: value,
                'step': step
            }, self._scalar_save_file, 'json')

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars to disk.

        The scalar dict will be written to the default and
        specified files if ``file_path`` is specified.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values. The value must be dumped format.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the ``file_path`` file at the same time
                if the ``file_path`` parameter is specified.
                Default to None.
        """
        assert isinstance(scalar_dict, dict)
        scalar_dict.setdefault('step', step)

        if self.__mkdir_or_exist():
            if file_path is not None:
                assert file_path.split('.')[-1] == 'json'
                new_save_file_path = osp.join(
                    self._save_dir,  # type: ignore
                    file_path)
                assert new_save_file_path != self._scalar_save_file, \
                    '``file_path`` and ``scalar_save_file`` have the ' \
                    'same name, please set ``file_path`` to another value'
                self._dump(scalar_dict, new_save_file_path, 'json')
            self._dump(scalar_dict, self._scalar_save_file, 'json')

    def _dump(self, value_dict: dict, file_path: str,
              file_format: str) -> None:
        """dump dict to file.

        Args:
           value_dict (dict) : The dict data to saved.
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
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> wandb_vis_backend.add_config(cfg)

    Args:
        save_dir (str, optional): The root directory to save the files
            produced by the visualizer. Default to None.
        init_kwargs (dict, optional): wandb initialization
            input parameters. Default to None.
        commit: (bool, optional) Save the metrics dict to the wandb server
                and increment the step.  If false `wandb.log` just
                updates the current metrics dict with the row argument
                and metrics won't be saved until `wandb.log` is called
                with `commit=True`. Default to True.
    """

    def __init__(self,
                 save_dir: Optional[str] = None,
                 init_kwargs: Optional[dict] = None,
                 commit: Optional[bool] = True):
        super(WandbVisBackend, self).__init__(save_dir)
        self._init_kwargs = init_kwargs
        self._commit = commit

    def __mkdir_or_exist(self):
        """If ``save_dir`` is None, the directory will not be created, and
        ``add_xxx`` will have no effect."""
        if self._save_dir is None:
            return False

        if self._is_created is False:
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
            if self._init_kwargs is None:
                self._init_kwargs = {'dir': self._save_dir}
            else:
                self._init_kwargs.setdefault('dir', self._save_dir)
            self._wandb = self.__setup_env(self._init_kwargs)
            self._is_created = True
        return True

    @property
    def experiment(self):
        """Return wandb object.

        The experiment attribute can get the wandb backend, If you want to
        write other data, such as writing a table, you can directly get the
        wandb backend through experiment.
        """
        if self.__mkdir_or_exist():
            return self._wandb
        else:
            return None

    def __setup_env(self, init_kwargs: dict) -> Any:
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

        wandb.init(**init_kwargs)
        return wandb

    def add_config(self, config: Config, **kwargs) -> None:
        if self.__mkdir_or_exist():
            cfg_path = os.path.join(self._wandb.run.dir, 'config.py')
            config.dump(cfg_path)
            # Files under run.dir are automatically uploaded,
            # so no need to manually call save.
            # self._wandb.save(cfg_path)

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to wandb.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Useless parameter. Wandb does not
                need this parameter. Default to 0.
        """
        if self.__mkdir_or_exist():
            self._wandb.log({name: image}, commit=self._commit)

    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to wandb.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Useless parameter. Wandb does not
                need this parameter. Default to 0.
        """
        if self.__mkdir_or_exist():
            self._wandb.log({name: value}, commit=self._commit)

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to wandb.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Useless parameter. Wandb does not
                need this parameter. Default to 0.
            file_path (str, optional): Useless parameter. Just for
                interface unification. Default to None.
        """
        if self.__mkdir_or_exist():
            self._wandb.log(scalar_dict, commit=self._commit)

    def close(self) -> None:
        """close an opened wandb object."""
        if hasattr(self, '_wandb'):
            self._wandb.join()


@VISBACKENDS.register_module()
class TensorboardVisBackend(BaseVisBackend):
    """Tensorboard class. It can write images, config, scalars, etc. to a
    tensorboard file.

    Examples:
        >>> from mmengine.visualization import TensorboardVisBackend
        >>> import numpy as np
        >>> tensorboard_vis_backend = \
        >>>     TensorboardVisBackend(save_dir='temp_dir')
        >>> img=np.random.randint(0, 256, size=(10, 10, 3))
        >>> tensorboard_vis_backend.add_image('img', img)
        >>> tensorboard_vis_backend.add_scaler('mAP', 0.6)
        >>> tensorboard_vis_backend.add_scalars({'loss': 0.1,'acc':0.8})
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> tensorboard_vis_backend.add_config(cfg)

    Args:
        save_dir (str): The root directory to save the files
            produced by the backend. Default to None.
    """

    def __init__(self, save_dir: Optional[str] = None):
        super(TensorboardVisBackend, self).__init__(save_dir)

    def __mkdir_or_exist(self):
        """If ``save_dir`` is None, the directory will not be created, and
        ``add_xxx`` will have no effect."""
        if self._save_dir is None:
            return False

        if self._is_created is False:
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
            self._tensorboard = self.__setup_env()
            self._is_created = True
        return True

    def __setup_env(self) -> Any:
        """Setup env.

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
        return SummaryWriter(self._save_dir)

    @property
    def experiment(self):
        """Return Tensorboard object."""
        if self.__mkdir_or_exist():
            return self._tensorboard
        else:
            return None

    def add_config(self, config: Config, **kwargs) -> None:
        """Record the config to tensorboard.

        Args:
            config (Config): The Config object
        """
        if self.__mkdir_or_exist():
            self._tensorboard.add_text('config', config.pretty_text)

    def add_image(self,
                  name: str,
                  image: np.ndarray,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to tensorboard.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Global step value to record. Default to 0.
        """
        if self.__mkdir_or_exist():
            self._tensorboard.add_image(name, image, step, dataformats='HWC')

    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to tensorboard.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        if self.__mkdir_or_exist():
            if isinstance(value, (int, float, torch.Tensor, np.ndarray)):
                self._tensorboard.add_scalar(name, value, step)
            else:
                warnings.warn(
                    f'Got {type(value)}, but numpy array, torch tensor, '
                    f'int or float are expected. skip it！')

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalar's data to tensorboard.

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
        if self.__mkdir_or_exist():
            for key, value in scalar_dict.items():
                self.add_scalar(key, value, step)

    def close(self):
        """close an opened tensorboard object."""
        if hasattr(self, '_tensorboard'):
            self._tensorboard.close()
