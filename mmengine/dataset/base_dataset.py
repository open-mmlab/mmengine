# Copyright (c) OpenMMLab. All rights reserved.
import copy
import functools
import os.path as osp
import pickle
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch.utils.data import Dataset

from mmengine.fileio import list_from_file, load
from mmengine.registry import TRANSFORMS
from mmengine.utils import check_file_exist


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms: Sequence[Union[dict, Callable]]):
        self.transforms: List[Callable] = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'{type(transform)} is not callable.')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict.')

    def __call__(self, data: dict) -> Optional[dict]:
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
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


def full_init_decorator(old_func: Callable) -> Any:
    """Auto full_init decorator for class method.

     The decorated function will call ``obj.full_init()``, if
     ``obj._fully_initialized=False``.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``full_init`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        if not hasattr(obj, 'full_init'):
            raise AttributeError(f"{type(obj)} doesn't have full_init "
                                 f"method.'")
        if not hasattr(obj, '_fully_initialized') or \
           not getattr(obj, '_fully_initialized'):
            obj.full_init()  # type: ignore
            obj.__setattr__('_fully_initialized', True)

        return old_func(obj, *args, **kwargs)

    return wrapper


class BaseDataset(Dataset, metaclass=ABCMeta):
    r""" BaseDataset for open source projects in OpenMMLab.

    The class inherited from ``BaseDataset`` need to implement
    ``parse_annotations`` method. The annotation format is shown as follows.

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

    All subclasses should implement ``parse_annotations()`` method.

    Args:
        ann_file (str): Annotation file path.
        meta (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img=None, ann=None).
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        num_samples (int, optional): Support using first few data in
            annotation file. Defaults to -1 which means using all
            ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
        max_refetch (int, optional): The maximum number of cycles to get a
            valid image. Defaults to 1000.
    """
    META: dict = dict()

    def __init__(self,
                 ann_file: str,
                 meta: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=None, ann=None),
                 filter_cfg: Optional[dict] = None,
                 num_samples: int = -1,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):

        self.data_root = data_root
        self.data_prefix = copy.deepcopy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._num_samples = num_samples
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_infos: List[dict] = []
        self.data_infos_bytes: bytearray = bytearray()

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

    @full_init_decorator
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(self.data_infos_bytes[start_addr:end_addr])
            data_info = pickle.loads(bytes)
        else:
            data_info = self.data_infos[idx]
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx
        return data_info

    def full_init(self):
        """Load annotation file and set ``self._fully_initialized`` to True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation phase and ``self._fully_initialized`` will be set to
        True. If ``obj._fully_initialized=False``, the class method decorated
        by full_init_decorator will call ``full_init`` automatically.
        """
        if self._fully_initialized:
            return

        # load data information
        self.data_infos = self.load_data_infos(self.ann_file)
        # filter illegal data, such as data that has no annotations.
        self.data_infos = self.filter_data()
        # slice data_infos
        self.data_infos = self._slice_data()
        # serialize data_infos
        if self.serialize_data:
            self.data_infos_bytes, self.data_address = self._serialize_data()
            self.data_infos = []
        self._fully_initialized = True

    @property
    def meta(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.META``,
            annotation file and meta parameter during instantiation.
        """
        return copy.deepcopy(self._meta)

    @abstractmethod
    def parse_annotations(self,
                          raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format. This method must be
        implemented by class inherited from ``BaseDataset``.
        ``parse_annotations`` should return ``dict``or ``List[dict]``. Each
        dict denotes a sample.

        Args:
            raw_data_info (dict): Raw annotation load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        pass

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg. Defaults return all
        ``data_infos``.

        If some ``data_infos`` should be filtered according to specific logic,
        the subclass should override this method.

        Returns:
            List[dict]: Filtered results.
        """
        return self.data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``ClassBalancedDataset`` requires a subclass which implements this
        method.

        Args:
            idx (int): The index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        raise NotImplementedError(f'{type(self)} must implement `get_cat_ids` '
                                  f'method')

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image of dataset after ``self.pipelines`` and
        ``full_init`` will be called if the  dataset has not been fully
        initialized.

        During training phase, if ``self.pipelines`` get ``None``,
        ``self._rand_another`` will be called until a valid image is retrieved.

        Args:
            idx (int): The index of self.data_infos

        Returns:
            dict: The idx-th image of dataset after ``self.pipelines``.
        """
        if not self._fully_initialized:
            warnings.warn(
                'Please call `self.full_init()` manually to accrelate the '
                'speed.')
            self.full_init()

        if self.test_mode:
            return self._prepare_data(idx)
        loop_count = 0
        while True:
            if loop_count >= self.max_refetch:
                raise Exception(f'Cannot find valid image after {loop_count}! '
                                f'please check your image path and pipelines')
            loop_count += 1
            data = self._prepare_data(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

    def load_data_infos(self, ann_file: str) -> List[dict]:
        """Load annotation from ann_file.

        If the annotation file doesn't conform to `OpenMMLab 2.0 dataset format
        <https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md>`_ .
        The subclass must override this method to load annotation.

        Args:
            ann_file (str): Absolute annotation file path if ``self.root=None``
                or relative path if ``self.root=/path/to/data/``.

        Note:
            `metadata` in ann_file has the lowest priority, which will be
            overwritten by ``dataset.META`` during instantiation

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        check_file_exist(ann_file)
        anns = load(ann_file)
        if not isinstance(anns, dict):
            raise TypeError('Wrong format annotation file!')
        if 'data_infos' not in anns or 'metadata' not in anns:
            raise ValueError('Annotation must have data_infos and metadata '
                             'keys')
        # allow meta
        meta_data, raw_data_infos = anns['metadata'], anns['data_infos']

        # update self._meta
        for k, v in meta_data.items():
            # We only merge keys that are not contained in self._meta.
            if k in self._meta:
                continue
            self._meta[k] = v
        # load and parse data_infos
        data_infos = []
        for raw_data_info in raw_data_infos:
            data_info = self.parse_annotations(copy.deepcopy(raw_data_info))
            if isinstance(data_info, dict):
                data_infos.append(data_info)
            elif isinstance(data_info, list):
                # data_info can also be a list of dict, which means one
                # data_info contains multiple samples.
                for _ in data_info:
                    if not isinstance(_, dict):
                        raise TypeError('data_info must be a list[dict] if '
                                        'data_info is a list')
                data_infos.extend(data_info)
            else:
                raise TypeError

        return data_infos

    @classmethod
    def _get_meta_data(cls, in_meta: dict = None) -> dict:
        """Collect meta information from meta dict.

        Args:
            in_meta (dict): Meta information dict. if in_meta contains existed
                filename, it will be parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # cls.META will be overwritten by in_meta
        cls_meta = copy.deepcopy(cls.META)
        if not in_meta:
            return cls_meta
        if not isinstance(in_meta, dict):
            raise TypeError('in_meta must be a dict!')
        defined_keys = set()
        for k, v in in_meta.items():
            defined_keys.add(k)
            if isinstance(v, str):
                # if filename in in_meta, this key will be further parsed.
                # nested filename will be ignored.:
                if osp.isfile(v):
                    cls_meta[k] = list_from_file(v)
                else:
                    cls_meta[k] = v

            else:
                cls_meta[k] = v
        return cls_meta

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

    def _slice_data(self) -> List[dict]:
        """Slice ``self.data_infos``. BaseDataset supports only using the first
        few data.

        Returns:
            List[dict]: A slice of ``self.data_infos``
        """
        if self._num_samples > 0:
            return self.data_infos[:self._num_samples]
        else:
            return self.data_infos

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_infos``, which will be called in
        ``full_init``.

        Hold memory using serialized objects, data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: serialize result and corresponding
                address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        serialized_data_infos_list = [_serialize(x) for x in self.data_infos]
        address_list = np.asarray([len(x) for x in serialized_data_infos_list],
                                  dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        serialized_data_infos = np.concatenate(serialized_data_infos_list)

        return serialized_data_infos, data_address

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random int from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))

    def _prepare_data(self, idx) -> Any:
        """Get data after pipeline.

        Args:
            idx (int): The index of dataset.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    @full_init_decorator
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_infos)
