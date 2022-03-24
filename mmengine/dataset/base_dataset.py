# Copyright (c) OpenMMLab. All rights reserved.
import copy
import functools
import gc
import os.path as osp
import pickle
import warnings
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
                # Build transform from dict.
                transform = TRANSFORMS.build(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a callable object or dict, '
                    f'but got {type(transform)}')

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # If transform get `None` data, terminate the loop immediately
            # and return `None`.
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


def force_full_init(old_func: Callable) -> Any:
    """Those methods decorated by ``force_full_init`` will be forced to call
    ``full_init`` if the instance has not been fully initiated.

    Args:
        old_func (Callable): Decorated function, make sure the first arg is an
            instance with ``full_init`` method.

    Returns:
        Any: Depends on old_func.
    """

    @functools.wraps(old_func)
    def wrapper(obj: object, *args, **kwargs):
        # The instance must have `full_init` method.
        if not hasattr(obj, 'full_init'):
            raise AttributeError(f'{type(obj)} does not have full_init '
                                 'method.')
        # If instance does not have `_fully_initialized` attribute or
        # `_fully_initialized` is False, call `full_init` and set
        # `_fully_initialized` to True
        if not getattr(obj, '_fully_initialized', False):
            warnings.warn('Attribute `_fully_initialized` is not defined in '
                          f'{type(obj)} or `type(obj)._fully_initialized is '
                          'False, `full_init` will be called and '
                          f'{type(obj)}._fully_initialized will be set to '
                          'True')
            obj.full_init()  # type: ignore
            obj._fully_initialized = True  # type: ignore

        return old_func(obj, *args, **kwargs)

    return wrapper


class BaseDataset(Dataset):
    r"""BaseDataset for open source projects in OpenMMLab.

    The annotation format is shown as follows.

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

    Args:
        ann_file (str): Annotation file path.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_root (str, optional): The root directory for ``data_prefix`` and
            ``ann_file``. Defaults to None.
        data_prefix (dict, optional): Prefix for training data. Defaults to
            dict(img=None, ann=None).
        filter_cfg (dict, optional): Config for filter data. Defaults to None.
        indices (int, optional): Support using first few data in
            annotation file to facilitate training/testing on a smaller
            dataset. Defaults to -1 which means using all ``data_infos``.
        serialize_data (bool, optional): Whether to hold memory using
            serialized objects, when enabled, data loader workers can use
            shared RAM from master process instead of making a copy. Defaults
            to True.
        pipeline (list, optional): Processing pipeline. Defaults to [].
        test_mode (bool, optional): ``test_mode=True`` means in test phase.
            Defaults to False.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. In some cases, such as visualization, only the meta
            information of the dataset is needed, which is not necessary to
            load annotation file. ``Basedataset`` can skip load annotations to
            save time by set ``lazy_init=False``. Defaults to False.
        max_refetch (int, optional): If ``Basedataset.prepare_data`` get a
            None img. The maximum extra number of cycles to get a valid
            image. Defaults to 1000.

    Note:
        BaseDataset collects meta information from `annotation file` (the
        lowest priority), ``BaseDataset.METAINFO``(medium) and `meta parameter`
        (highest) passed to constructors. The lower priority meta information
        will be overwritten by higher one.

    Examples:
        Assume the annotation file is given above.
        >>> class CustomDataset(BaseDataset):
        >>>     METAINFO: dict = dict(task_name='custom_task',
        >>>                           dataset_type='custom_type')
        >>> metainfo=dict(task_name='custom_task_name')
        >>> custom_dataset = CustomDataset(
        >>>                      'path/to/ann_file',
        >>>                      metainfo=metainfo
        >>> # meta information of annotation file will be overwritten by
        >>> # `CustomDataset.METAINFO`. The merged meta information will
        >>> # further be overwritten by argument `metainfo`.
        >>> custom_dataset.meta
        {'task_name': custom_task_name, dataset_type: custom_type}
    """

    METAINFO: dict = dict()
    _fully_initialized: bool = False

    def __init__(self,
                 ann_file: str,
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img=None, ann=None),
                 filter_cfg: Optional[dict] = None,
                 indices: Union[int, List[int]] = -1,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):

        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.ann_file = ann_file
        self.filter_cfg = copy.deepcopy(filter_cfg)
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_list_bytes: np.ndarray

        # Set meta information.
        self._metainfo = self._get_meta_info(copy.deepcopy(metainfo))

        # Join paths.
        if self.data_root is not None:
            self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        # Full initialize data.
        if not lazy_init:
            self.full_init()

    @force_full_init
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
            bytes = memoryview(
                self.data_list_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = self.data_list[idx]
        # Some codebase needs to record the positive index of data information
        # in dataset.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info

    def full_init(self):
        """Load annotation file and set ``BaseDataset._fully_initialized`` to
        True.

        If ``lazy_init=False``, ``full_init`` will be called during the
        instantiation and ``self._fully_initialized`` will be set to True. If
        ``obj._fully_initialized=False``, the class method decorated by
        ``force_full_init`` will call ``full_init`` automatically.

        Several steps to initialize annotation:

            - load_data_list: Load annotations from annotation file.
            - filter data information: Filter annotations according to
              filter_cfg.
            - slice_data: Slice dataset according to ``self._indices``
            - serialize_data: Serialize ``self.data_list`` if
            ``self.serialize_data`` is True.
        """
        if self._fully_initialized:
            return
        self._fully_initialized = True
        # load data information
        self.data_list = self.load_data_list(self.ann_file)
        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices > 0:
            self.data_list = self._get_unserialized_subdata(self._indices)

        # serialize data_infos
        if self.serialize_data:
            self.data_list_bytes, self.data_address = self._serialize_data()
            # Empty cache for preventing making multiple copies of
            # `self.data_info` when loading data multi-processes.
            self.data_list.clear()
            gc.collect()

    @property
    def meta(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and meta parameter during instantiation.
        """
        return copy.deepcopy(self._metainfo)

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        This method should return dict or list of dict. Each dict or list
        contains the data information of a training sample. If the protocol of
        the sample annotations is changed, this function can be overridden to
        update the parsing logic while keeping compatibility.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            list or list of dict: Parsed annotation.
        """
        return raw_data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg. Defaults return all
        ``data_infos``.

        If some ``data_infos`` could be filtered according to specific logic,
        the subclass should override this method.

        Returns:
            list of dict: Filtered results.
        """
        return self.data_list

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index. Dataset wrapped by ClassBalancedDataset
        must implement this method.

        The ``ClassBalancedDataset`` requires a subclass which implements this
        method.

        Args:
            idx (int): The index of data.

        Returns:
            list of int: All categories in the image of specified index.
        """
        raise NotImplementedError(f'{type(self)} must implement `get_cat_ids` '
                                  'method')

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. Although this may
        # work, but it will consume a lot of time and memory. Therefore, it is
        # recommended to manually call `full_init` before dataset fed into
        # dataloader to ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            warnings.warn(
                'Please call `full_init()` method manually to accelerate '
                'the speed.')
            self.full_init()

        if self.test_mode:
            # Get a testing phase data.
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data
        # Get a training phase data.
        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random enhancements may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')

    def load_data_list(self, ann_file: str) -> List[dict]:
        """Load annotations from an annotation file.

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/basedataset.md>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Args:
            ann_file (str): Absolute annotation file path if ``self.root=None``
                or relative path if ``self.root=/path/to/data/``.

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        check_file_exist(ann_file)
        annotations = load(ann_file)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_infos' not in annotations or 'metadata' not in annotations:
            raise ValueError('Annotation must have data_infos and metadata '
                             'keys')
        meta_data = annotations['metadata']
        raw_data_infos = annotations['data_infos']

        # update self._metainfo
        for k, v in meta_data.items():
            # We only merge keys that are not contained in self._metainfo.
            self._metainfo.setdefault(k, v)

        # load and parse data_infos
        data_infos = []
        for raw_data_info in raw_data_infos:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_infos.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_infos.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_infos

    @classmethod
    def _get_meta_info(cls, in_metainfo: dict = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            in_metainfo (dict): Meta information dict. If ``in_meta`` contains
                existed filename, it will be parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # cls.METAINFO will be overwritten by in_meta
        cls_meta = copy.deepcopy(cls.METAINFO)
        if in_metainfo is None:
            return cls_meta
        if not isinstance(in_metainfo, dict):
            raise TypeError(
                f'in_meta should be a dict, but got {type(in_metainfo)}')

        for k, v in in_metainfo.items():
            if isinstance(v, str) and osp.isfile(v):
                # if filename in in_meta, this key will be further parsed.
                # nested filename will be ignored.:
                cls_meta[k] = list_from_file(v)
            else:
                cls_meta[k] = v

        return cls_meta

    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> self.ann_file
            'a/b/c/f'
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
            >>> self.ann_file
            'a/b/c/f'
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        if not osp.isabs(self.ann_file):
            self.ann_file = osp.join(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if prefix is None:
                self.data_prefix[data_key] = self.data_root
            elif isinstance(prefix, str):
                if not osp.isabs(prefix):
                    self.data_prefix[data_key] = osp.join(
                        self.data_root, prefix)
            else:
                raise TypeError('prefix should be a string or None, but got '
                                f'{type(prefix)}')

    @force_full_init
    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Convert dataset to a subdataset.

        This method will overwrite the original dataset to a subset dataset. If
        ``type(indices)`` is int, ``get_subset_`` will return a subdataset
        which contains the first ``indices`` data information. If
        ``type(indices)`` is a list of int, the subdataset will contain the
        data information according to the index given in ``indices``.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> dataset.get_subset_(90)
              >>> len(dataset)
              90
              >>> dataset.get_subset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
              >>> len(dataset)
              10

        Args:
            indices (int or list of int): If ``type(indices)`` is int, indices
                represents the first few data of dataset. If  `type(indices)``
                is list, indices represents the target data information index
                which consist of subdataset.

        Returns:
            BaseDataset: A subset of dataset.
        """
        # Get subset of data from  serialized data or data information list
        # according to `self.serialize_data`.
        if self.serialize_data:
            self.data_list_bytes, self.data_address = \
                self._get_serialized_subdata(indices)
        else:
            self.data_list = self._get_unserialized_subdata(indices)

    @force_full_init
    def get_subset(self, indices: Union[List[int], int]) -> 'BaseDataset':
        """Get a subset of dataset.

        This method will return a subdataset of original dataset. If
        ``type(indices)`` is int, ``get_subset_`` will return a subdataset
        which contains the first ``indices`` data information. If
        ``type(indices)`` is a list of int, the subdataset will contain the
        data information according to the index given in ``indices``.

        Examples:
              >>> dataset = BaseDataset('path/to/ann_file')
              >>> len(dataset)
              100
              >>> subdataset = dataset.get_subset(90)
              >>> len(sub_dataset)
              90
              >>> subdataset = dataset.get_subset([0, 1, 2, 3, 4, 5, 6, 7,
              >>> 8, 9])
              >>> len(sub_dataset)
              10

        Args:
            indices (int or list of int): If ``type(indices)`` is int, indices
                represents the first few data of dataset. If  `type(indices)``
                is list, indices represents the target data information index
                which consist of subdataset.

        Returns:
            BaseDataset: A subset of dataset.
        """
        # Get subset of data from  serialized data or data information list
        # according to `self.serialize_data`. Since `_get_serialized_subdata`
        # will recalculate the subset data information,
        # `_copy_without_annotation` will copy all attributes except data
        # information.
        sub_dataset = self._copy_without_annotation()
        # Avoid calling `full_init` to overwrite `data_list`
        if self.serialize_data:
            sub_dataset.data_list_bytes, sub_dataset.data_address = \
                self._get_serialized_subdata(indices)  # type: ignore
        else:
            sub_dataset.data_list = self._get_unserialized_subdata(indices)
        return sub_dataset

    def _get_serialized_subdata(self, indices: Union[List[int], int]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Get subset of serialized data information list.

        Args:
            indices (int or list of int): If ``type(indices)`` is int, indices
                represents the first few data of serialized data information
                list. If  `type(indices)`` is list, indices represents the
                target data information index which consist of subset data
                information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of serialized data
            information list.
        """
        sub_data_list_bytes: Union[List, np.ndarray]
        sub_data_address: Union[List, np.ndarray]
        if isinstance(indices, int) and indices > 0:
            # Get the first few data information.
            end_addr = self.data_address[indices - 1].item()
            # Slicing operation of `np.ndarray` does not trigger a memory copy.
            sub_data_list_bytes = self.data_list_bytes[:end_addr].copy()
            # Since the buffer size of first few data information is not
            # changed,
            sub_data_address = self.data_address[:indices]
        elif isinstance(indices, list):
            sub_data_list_bytes = []
            sub_data_address = []
            for idx in indices:
                assert idx < len(self)
                start_addr = 0 if idx == 0 else self.data_address[idx -
                                                                  1].item()
                end_addr = self.data_address[idx].item()
                # Get data information by address.
                sub_data_list_bytes.append(
                    self.data_list_bytes[start_addr:end_addr])
                # Get data information size.
                sub_data_address.append(end_addr - start_addr)

            sub_data_list_bytes = np.concatenate(sub_data_list_bytes)
            sub_data_address = np.cumsum(sub_data_address)
        else:
            raise TypeError('indices should be a int or list of int, '
                            f'but got {type(indices)}')
        return sub_data_list_bytes, sub_data_address  # type: ignore

    def _get_unserialized_subdata(self, indices: Union[List[int], int]) -> \
            list:
        """Get subset of data information list.

        Args:
            indices (int or list of int): If ``type(indices)`` is int, indices
                represents the first few data of data information.
                If  `type(indices)`` is list, indices represents the target
                data information index which consist of subset data
                information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: subset of data information list.
        """
        if isinstance(indices, int) and indices > 0:
            sub_data_list = self.data_list[:indices]
        elif isinstance(indices, list):
            subdata_list = []
            for idx in indices:
                subdata_list.append(self.data_list[idx])
            sub_data_list = subdata_list
        else:
            raise TypeError('indices should be a int or list of int, '
                            f'but got {type(indices)}')
        return sub_data_list

    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_list`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.

        Hold memory using serialized objects, and data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Serialized result and corresponding
            address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        serialized_data_infos_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in serialized_data_infos_list],
                                  dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        # TODO Check if np.concatenate is necessary
        serialized_data_infos = np.concatenate(serialized_data_infos_list)

        return serialized_data_infos, data_address

    def _rand_another(self) -> int:
        """Get random index.

        Returns:
            int: Random index from 0 to ``len(self)-1``
        """
        return np.random.randint(0, len(self))

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)

    @force_full_init
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    def _copy_without_annotation(self, memo=dict()):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            if key in ['data_list', 'data_address', 'data_list_bytes']:
                continue
            super(BaseDataset, other).__setattr__(key,
                                                  copy.deepcopy(value, memo))

        return other
