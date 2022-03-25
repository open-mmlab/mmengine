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


class BaseWriter(metaclass=ABCMeta):
    def __init__(self, save_dir: Optional[str] = None):
        self._save_dir = save_dir
        if self._save_dir:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            self._save_dir = osp.join(
                self._save_dir, f'write_data_{timestamp}')  # type: ignore

    @property
    @abstractmethod
    def experiment(self) -> Any:
        pass

    def add_params(self, params_dict: dict, **kwargs) -> None:
        pass

    def add_graph(self, model: torch.nn.Module,
                  input_tensor: Union[torch.Tensor,
                                      List[torch.Tensor]], **kwargs) -> None:
        pass

    def add_image(self,
                  name: str,
                  image: Union[np.ndarray, Any],
                  step: int = 0,
                  **kwargs) -> None:
        pass

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        pass

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


@WRITERS.register_module(force=True)
class LocalWriter(BaseWriter):
    def __init__(self,
                 save_dir: str,
                 img_save_dir: str = 'writer_image',
                 params_save_file: str = 'parameters.yaml',
                 scalar_save_file: str = 'scalars.json',
                 img_show: bool = False,
                 wait_time=0):
        assert params_save_file.split('.')[-1] == 'yaml'
        assert scalar_save_file.split('.')[-1] == 'json'
        super(LocalWriter, self).__init__(save_dir)
        os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        self._img_save_dir = osp.join(
            self._save_dir,  # type: ignore
            img_save_dir)
        self._img_show = img_show
        self._wait_time = wait_time

    @property
    def experiment(self) -> 'LocalWriter':
        return self

    def add_image(self,
                  name: str,
                  image,
                  step: int = 0,
                  **kwargs) -> None:
        assert isinstance(image, np.ndarray)
        if self._img_show:
            # show image
            cv2.namedWindow(name, 0)
            cv2.imshow(name, image)
            cv2.waitKey(self._wait_time)
        else:
            drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            os.makedirs(self._img_save_dir, exist_ok=True)
            save_file_name = f'{name}_{step}.png'
            cv2.imwrite(
                osp.join(self._img_save_dir, save_file_name), drawn_image)


# 要支持懒加载，否则每次即使不用他默认也会创建文件夹
@WRITERS.register_module(force=True)
class WandbWriter(BaseWriter):
    def __init__(self,
                 init_kwargs: Optional[dict] = None,
                 commit: Optional[bool] = False,
                 save_dir: Optional[str] = None):
        super(WandbWriter, self).__init__(save_dir)
        self._commit = commit
        self._wandb = self._setup_env(init_kwargs)

    @property
    def experiment(self):
        return self._wandb

    def _setup_env(self, init_kwargs: Optional[dict] = None) -> Any:
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

    def add_image(self,
                  name: str,
                  image,
                  step: int = 0,
                  **kwargs) -> None:
        self._wandb.log({name: self._wandb.Image(image)},
                        commit=self._commit,
                        step=step)

    def close(self) -> None:
        """close an opened wandb object."""
        if hasattr(self, '_wandb'):
            self._wandb.join()

