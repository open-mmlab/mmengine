# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import torch


class BaseDataElement:
    """A base data structure interface of OpenMMlab.

    Data elements refer to predicted results or ground truth labels on a
    task, such as predicted bboxes, instance masks, semantic
    segmentation masks, etc. Because groundtruth labels and predicted results
    often have similar properties (for example, the predicted bboxes and the
    groundtruth bboxes), MMEngine uses the same abstract data interface to
    encapsulate predicted results and groundtruth labels, and it is recommended
    to use different name conventions to distinguish them, such as using
    ``gt_instances`` and ``pred_instances`` to distinguish between labels and
    predicted results. Additionally, we distinguish data elements at instance
    level, pixel level, and label level. Each of these types has its own
    characteristics. Therefore, MMEngine defines the base class
    ``BaseDataElement``, and implement ``InstanceData``, ``PixelData``, and
    ``LabelData`` inheriting from ``BaseDataElement`` to represent different
    types of ground truth labels or predictions.
    They are used as interfaces between different commopenets.

    The attributes in ``BaseDataElement`` are divided into two parts,
    the ``metainfo`` and the ``data`` respectively.

      - ``metainfo``: Usually contains the
        information about the image such as filename,
        image_shape, pad_shape, etc. The attributes can be accessed or
        modified by dict-like or object-like operations, such as
        ``.``(for data access and modification) , ``in``, ``del``,
        ``pop(str)``, ``get(str)``, ``metainfo_keys()``,
        ``metainfo_values()``, ``metainfo_items()``, ``set_metainfo()``(for
        set or change key-value pairs in metainfo).

      - ``data``: Annotations or model predictions are
        stored. The attributes can be accessed or modified by
        dict-like or object-like operations, such as
        ``.`` , ``in``, ``del``, ``pop(str)`` ``get(str)``, ``data_keys()``,
        ``data_values()``, ``data_items()``. Users can also apply tensor-like
        methods to all obj:``torch.Tensor`` in the ``data_fileds``,
        such as ``.cuda()``, ``.cpu()``, ``.numpy()``, , ``.to()``
        ``to_tensor()``, ``.detach()``, ``.numpy()``

    Args:
        meta_info (dict, optional): A dict contains the meta information
            of single image. such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        data (dict, optional): A dict contains annotations of single image or
            model predictions. Defaults to None.

    Examples:
        >>> from mmengine.data import BaseDataElement
        >>> gt_instances = BaseDataElement()
        >>> bboxes = torch.rand((5, 4))
        >>> scores = torch.rand((5,))
        >>> img_id = 0
        >>> img_shape = (800, 1333)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=img_shape),
        ...     data=dict(bboxes=bboxes, scores=scores))
        >>> gt_instances = BaseDataElement(dict(img_id=img_id,
        ...                                     img_shape=(H, W)))

        >>> # new
        >>> gt_instances1 = gt_instance.new(
        ...                     metainfo=dict(img_id=1, img_shape=(640, 640)),
        ...                     data=dict(bboxes=torch.rand((5, 4)),
        ...                               scores=torch.rand((5,))))
        >>> gt_instances2 = gt_instances1.new()

        >>> # add and process property
        >>> gt_instances = BaseDataElement()
        >>> gt_instances.set_metainfo(dict(img_id=9, img_shape=(100, 100))
        >>> assert 'img_shape' in gt_instances.metainfo_keys()
        >>> assert 'img_shape' in gt_instances
        >>> assert 'img_shape' not in gt_instances.data_keys()
        >>> assert 'img_shape' in gt_instances.keys()
        >>> print(gt_instances.img_shape)
        >>> gt_instances.scores = torch.rand((5,))
        >>> assert 'scores' in gt_instances.data_keys()
        >>> assert 'scores' in gt_instances
        >>> assert 'scores' in gt_instances.keys()
        >>> assert 'scores' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.scores)
        >>> gt_instances.bboxes = torch.rand((5, 4))
        >>> assert 'bboxes' in gt_instances.data_keys()
        >>> assert 'bboxes' in gt_instances
        >>> assert 'bboxes' in gt_instances.keys()
        >>> assert 'bboxes' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.bboxes)

        >>> # delete and change property
        >>> gt_instances = BaseDataElement(
        ...  metainfo=dict(img_id=0, img_shape=(640, 640)),
        ...  data=dict(bboxes=torch.rand((6, 4)), scores=torch.rand((6,))))
        >>> gt_instances.img_shape = (1280, 1280)
        >>> gt_instances.img_shape  # (1280, 1280)
        >>> gt_instances.bboxes = gt_instances.bboxes * 2
        >>> gt_instances.get('img_shape', None)  # (640, 640)
        >>> gt_instances.get('bboxes', None)    # 6x4 tensor
        >>> del gt_instances.img_shape
        >>> del gt_instances.bboxes
        >>> assert 'img_shape' not in gt_instances
        >>> assert 'bboxes' not in gt_instances
        >>> gt_instances.pop('img_shape', None)  # None
        >>> gt_instances.pop('bboxes', None)  # None

        >>> # Tensor-like
        >>> cuda_instances = gt_instances.cuda()
        >>> cuda_instances = gt_instancess.to('cuda:0')
        >>> cpu_instances = cuda_instances.cpu()
        >>> cpu_instances = cuda_instances.to('cpu')
        >>> fp16_instances = cuda_instances.to(
        ...  device=None, dtype=torch.float16, non_blocking=False, copy=False,
        ...  memory_format=torch.preserve_format)
        >>> cpu_instances = cuda_instances.detach()
        >>> np_instances = cpu_instances.numpy()

        >>> # print
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> instance_data = BaseDataElement(metainfo=img_meta)
        >>> instance_data.det_labels = torch.LongTensor([0, 1, 2, 3])
        >>> instance_data.det_scores = torch.Tensor([0.01, 0.1, 0.2, 0.3])
        >>> print(results)
        <BaseDataElement(
        META INFORMATION
        img_shape: (800, 1196, 3)
        pad_shape: (800, 1216, 3)
        DATA FIELDS
        shape of det_labels: torch.Size([4])
        shape of det_scores: torch.Size([4])
        ) at 0x7f84acd10f90>
    """

    def __init__(self,
                 metainfo: Optional[dict] = None,
                 data: Optional[dict] = None) -> None:

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if data is not None:
            self.set_data(data)

    def set_metainfo(self, metainfo: dict) -> None:
        """Set or change key-value pairs in ``metainfo_field`` by parameter
        ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        assert isinstance(
            metainfo,
            dict), f'metainfo should be a ``dict`` but got {type(metainfo)}'
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            if k in self._data_fields:
                raise AttributeError(f'`{k}` is used in data,'
                                     'which is immutable. If you want to'
                                     'change the key in data, please use'
                                     'set_data')
            self._metainfo_fields.add(k)
            self.__dict__[k] = v

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter
        ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data,
                          dict), f'meta should be a `dict` but got {data}'
        for k, v in data.items():
            self.__setattr__(k, v)

    def new(self,
            metainfo: dict = None,
            data: dict = None) -> 'BaseDataElement':
        """Return a new data element with same type. If ``metainfo`` and
        ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new result will overwrite it
        with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
                Defaults to None.
            data (dict, optional): A dict contains annotations of image or
                model predictions. Defaults to None.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if data is not None:
            new_data.set_data(data)
        else:
            new_data.set_data(dict(self.data_items()))
        return new_data

    def data_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        return list(self._data_fields)

    def metainfo_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return list(self._metainfo_fields)

    def data_values(self) -> list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.data_keys()]

    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.data_keys()

    def values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.data_values()

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: an iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.keys():
            yield (k, getattr(self, k))

    def data_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: an iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.data_keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: an iterator object whose element is (key, value) tuple
            pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    def __setattr__(self, name: str, val: Any):
        """setattr is only used to set data."""
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, val)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable. ')
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f'`{name}` is used in meta information.'
                    'if you want to change the key in metainfo, please use'
                    'set_metainfo(dict(name=val))')

            self._data_fields.add(name)
            super().__setattr__(name, val)

    def __delattr__(self, item: str):
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a '
                                 'private attribute, which is immutable. ')
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __setitem__ = __setattr__
    __delitem__ = __delattr__

    def get(self, *args) -> Any:
        """get property in data and metainfo as the same as python."""
        assert len(args) < 3, '``get`` get more than 2 arguments'
        return self.__dict__.get(*args)

    def pop(self, *args) -> Any:
        """pop property in data and metainfo as the same as python."""
        assert len(args) < 3, '``pop`` get more than 2 arguments'
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f'{args[0]} is not contained in metainfo or data')

    def __contains__(self, item: str) -> bool:
        return item in self._data_fields or \
                    item in self._metainfo_fields

    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.data_items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        for k, v in self.metainfo_items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    # Tensor-like methods
    def cpu(self) -> 'BaseDataElement':
        """Convert all tensors to CPU in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        for k, v in self.metainfo_items():
            if isinstance(v, torch.Tensor):
                v = v.cpu()
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    # Tensor-like methods
    def cuda(self) -> 'BaseDataElement':
        """Convert all tensors to GPU in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        for k, v in self.metainfo_items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    # Tensor-like methods
    def detach(self) -> 'BaseDataElement':
        """Detach all tensors in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        for k, v in self.metainfo_items():
            if isinstance(v, torch.Tensor):
                v = v.detach()
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    # Tensor-like methods
    def numpy(self) -> 'BaseDataElement':
        """Convert all tensor to np.narray in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        for k, v in self.metainfo_items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    def to_tensor(self) -> 'BaseDataElement':
        """Convert all np.narray to tensor in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data = {k: v}
                new_data.set_data(data)
        for k, v in self.metainfo_items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    def __repr__(self) -> str:
        repr = '\n  META INFORMATION \n'
        for k, v in self.metainfo_items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                repr += f'shape of {k}: {v.shape} \n'
            else:
                repr += f'{k}: {v} \n'
        repr += '\n  DATA FIELDS \n'
        for k, v in self.data_items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                repr += f'shape of {k}: {v.shape} \n'
            else:
                repr += f'{k}: {v} \n'
        classname = self.__class__.__name__
        return f'<{classname}({repr}\n) at {hex(id(self))}>'
