# Copyright (c) OpenMMLab. All rights reserved.
import collections
import copy
import functools
import os.path as osp
import pickle
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from mmengine.fileio import list_from_file, load
from mmengine.registry import TRANSFORMS, build_from_cfg
from mmengine.utils import check_file_exist


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms: Sequence[Union[Dict, Callable]]):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, TRANSFORMS)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data: dict) -> object:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        """print ``self.transforms`` in sequence.

        Returns:
            str: format string
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


def full_init_before_called(old_func: Callable) -> Any:
    """Auto full_init decorator for class method.

     The decorated function will call ``obj.full_init()``, if
     ``obj._fully_initialized=True``.

    Args:
        old_func (Callable): Decorated function, make sure first arg is an
            instance with full_init method.

    Returns:
        Any: Depend on old_func
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        if not hasattr(obj, 'full_init'):
            raise AttributeError(f'{type(obj)} dont have full_init method')
        if not hasattr(obj, '_fully_initialized'):
            need_init = False
        else:
            need_init = getattr(obj, '_fully_initialized')
        if need_init:
            args[0].full_init()
            args[0]._fully_initialized = True
        return old_func(*args, **kwargs)

    return wrapper


class BaseDataset(Dataset, metaclass=ABCMeta):
    r""" BaseDataset for mmengine.

    The class inherited from ``BaseDataset`` need to implement
    parse_annotations method. The annotation format is shown as follows.

    .. code-block:: none

        {
            "metadata":
            {
              "dataset_type": "test_dataset",
              "task_name": "test_task"
            },
            "data_infos":
            [
              {
                "img_path": "test_img.jpg",
                "height": 604,
                "width": 640,
                "instances":
                [
                  {
                    "bbox": [0, 0, 10, 20],
                    "bbox_label": 1,
                    "mask": [[0,0],[0,10],[10,20],[20,0]],
                    "extra_anns": [1,2,3]
                  },
                  {
                    "bbox": [10, 10, 110, 120],
                    "bbox_label": 2,
                    "mask": [[10,10],[10,110],[110,120],[120,10]],
                    "extra_anns": [4,5,6]
                  }
                ]
              },
            ]
        }
    All subclasses should implement the following APIs:


    - ``parse_annotations()``

    Args:
        ann_file (str): Annotation file path.
        meta (dict, optional): meta infos for dataset, such as class
        information. Defaults to None.
        data_root (str, optional): data root for `data_prefix` and `ann_file`.
            Defaults to None.
        data_prefix (dict, optional): prefix for training data. Defaults to
            dict(img=None, ann=None).
        filter_cfg (dict, optional): filter cfg for ``filter_data``. Defaults
        to None
        num_samples (int, optional): support only use first few data in
            annotation file. Defaults to -1 and use all ``data_infos``
        serialize_data (bool, optional): whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
        pipeline (list, optional): Processing pipeline. Defaults to None.
        test_mode (bool, optional): test_mode=True during test phase. Defaults
            to False.
        lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False
    """
    META = dict()

    def __init__(self,
                 ann_file: str,
                 meta: Optional[Dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=None, ann=None),
                 filter_cfg: Optional[Dict] = None,
                 num_samples: int = -1,
                 serialize_data: bool = True,
                 pipeline: Optional[List] = None,
                 test_mode: bool = False,
                 lazy_init: bool = False):

        self.data_root = data_root
        self.data_prefix = copy.deepcopy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._num_samples = num_samples
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.data_infos = []
        # set meta information
        self._meta = self._get_meta_data(copy.deepcopy(meta))

        # join paths
        if self.data_root is not None:
            self._join_prefix()

        # build pipeline
        self.pipeline = Compose(pipeline)

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()
            self._fully_initialized = True

    @abstractmethod
    def parse_annotations(self,
                          raw_data_info: dict) -> Union[Dict, List[Dict]]:
        """Parse raw annotation to target format. This method must be
        implemented by class inherited from BaseDataset. ``parse_annotations``
        should return ``dict`` for img task and ``list[dict]`` for video task.

        Args:
            raw_data_info (dict): data_info load from ``ann_file``

        Returns:
            dict, list[dict]: parsed data_info
        """
        pass

    @classmethod
    def _get_meta_data(cls, in_meta: dict) -> dict:
        """collect meta infos from meta dict.

        Args:
            in_meta (dict): in_meta contains meta infos. if in_meta contains
             existed filename, it will be parsed by ``list_from_file``.

        Returns:
            dict: parsed meta infos.
        """
        # cls.META will be overwritten by in_meta
        out_meta = copy.deepcopy(cls.META)
        defined_keys = set()
        for k, v in in_meta.items():
            defined_keys.add(k)
            if isinstance(v, str):
                # if filename in in_meta, this key will be further parsed.
                # nested filename will be ignored.:
                if osp.isfile(v):
                    out_meta[k] = list_from_file(v)
                else:
                    out_meta[k] = v

            elif isinstance(v, (tuple, list, dict)):
                out_meta[k] = v
            else:
                raise ValueError(f'Unsupported type {type(v)} of {k}.')

        return out_meta

    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            self.data_prefix = dict(img='a/b/c/d/e')
            self.ann_file = 'a/b/c/f'
            self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            self.data_prefix = dict(img='/d/e')
            self.ann_file = 'a/b/c/f'
        """
        if not osp.isabs(self.ann_file):
            self.ann_file = osp.join(self.data_root, self.ann_file)

        for data_key, prefix in self.data_prefix.items():
            if prefix is None:
                self.data_prefix[data_key] = self.data_root
            elif not osp.isabs(prefix):
                self.data_prefix[data_key] = osp.join(self.data_root, prefix)

    def load_data_infos(self, ann_file: str) -> List[Dict]:
        """load annotation from ann_file.

        Args:
            ann_file (str): absolute annotation file path if ``self.root=None``
                or relative path ``self.root=/path/to/data/``

        Note:
            img_meta in ann_file has lowest priority, which will be over
            written by ``dataset.META`` and meta passed to __init__

        Returns:
            list[dict]: list of ``data_info``
        """
        check_file_exist(ann_file)
        anns = load(ann_file)
        meta_data, raw_data_infos = anns['meta_data'], anns['data_infos']

        # update self._meta
        for k, v in meta_data.items():
            # We only merge keys that are not contained in self._meta.
            if k in self._meta:
                continue

            if isinstance(v, str):
                if osp.isfile(v):
                    file_meta = list_from_file(v)
                    self.meta[k] = file_meta
                else:
                    self.meta[k] = v
            elif isinstance(v, (tuple, list, dict)):
                self._meta[k] = v
            else:
                raise ValueError(f'Unsupported type {type(v)} of {k}.')

        # load and parse data_infos
        data_infos = []
        for raw_data_info in raw_data_infos:
            data_info = self.parse_annotations(copy.deepcopy(raw_data_info))
            if isinstance(data_info, dict):
                data_infos.append(data_info)
            elif isinstance(data_info, list):
                # For video dataset, data_info must be a list of dict.
                for _ in data_info:
                    if not isinstance(_, dict):
                        raise TypeError('data_info must be a list[dict] if '
                                        'data_info is a list')
                data_infos.extend(data_info)
            else:
                raise TypeError

        return data_infos

    def filter_data(self) -> List[Dict]:
        """Filter data_infos according to filter_cfg. Defaults filter no data.

        Returns:
            list[dict]: Filtered results
        """
        if self.filter_cfg is None:
            return self.data_infos

    def slice_data(self):
        """Slice ``self.data_infos``. BaseDataset supports only using the first
        few data.

        Returns:
            list: slice of ``self.data_infos``
        """
        if self._num_samples > 0:
            return self.data_infos[:self._num_samples]
        else:
            return self.data_infos

    def _serialize_data(self, data_infos: List[Dict])\
            -> Tuple[np.ndarray, np.ndarray]:
        """Serialize self.data_infos, which will called in ``full_init``.

        Hold memory using serialized objects, data loader workers can use
        shared RAM from master process instead of making a copy.

        Args:
            data_infos (list[dict]): ``data_infos`` parsed by
            ``parse_annotations``.

        Returns:
            Tuple[list, np.ndarray]: serialize result and corresponding
                address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        serialized_data_infos = [_serialize(x) for x in data_infos]
        address_list = np.asarray([len(x) for x in serialized_data_infos],
                                  dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        serialized_data_infos = np.concatenate(serialized_data_infos)

        return serialized_data_infos, data_address

    def full_init(self):
        """Load annotation file and set ``self._fully_initialized`` to True.

        ``If ``lazy_init=True``, ``full_init`` will be called during the
        instantiation phase and ``self._fully_initialized`` will be set to
        True. If ``obj._fully_initialized=False``, the class method decorated
        by full_init_before_called will call ``full_init`` automatically.
        """
        if self._fully_initialized:
            return

        # load data information
        self.data_infos = self.load_data_infos(self.ann_file)
        # filter illegal data, such as data that has no annotations.
        self.data_infos = self.filter_data()
        # slice data_infos
        self.data_infos = self.slice_data()
        # serialize data_infos
        if self.serialize_data:
            self.data_infos, self.data_address = self._serialize_data(
                self.data_infos)

        self._fully_initialized = True

    @property
    def meta(self) -> dict:
        """Get meta of dataset.

        Returns:
            dict: meta_info collected from BaseDataset.META, annotation file
                and meta parameter during Instantiation.
        """
        return self._meta

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random int from 0 to ``len(self)``
        """
        return np.random.randint(0, len(self))

    def _prepare_data(self, idx) -> Any:
        """Get data after pipeline.

        Args:
            idx (int): index of dataset.

        Returns:
            Any: Depends on ``self.pipeline``
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    @full_init_before_called
    def get_data_info(self, idx: int) -> dict:
        """Get data_info by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: The i_th item of self.data_infos.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_infos[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = self.data_infos[idx]
        data_info['sample_idx'] = idx
        return data_info

    @full_init_before_called
    def __len__(self) -> int:
        """Get the length of filtered dataset.

        Returns:
            int: The length of filtered dataset.
        """
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_infos)

    def __getitem__(self, idx: int) -> dict:
        """Get the i_th item of ``self.data_infos`` after ``self.pipelines``.
        If target is ``None``, ``self._rand_another`` will be called until get
        a valid data.

        Args:
            idx: The index of self.data_infos

        Returns:
            dict: The i_th item of ``self.data_infos`` after ``self.pipelines``
        """
        if not self._fully_initialized:
            warnings.warn(
                'Please call `self.full_init()` manually to accrelate the '
                'speed.')
            self.full_init()

        if self.test_mode:
            return self._prepare_data(idx)
        while True:
            data = self._prepare_data(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data
