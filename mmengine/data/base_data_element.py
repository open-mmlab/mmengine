# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Iterator, Optional, Tuple, Type

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
        ``.`` , ``in``, ``del``, ``pop(str)`` ``get(str)``, ``keys()``,
        ``values()``, ``items()``. Users can also apply tensor-like
        methods to all obj:``torch.Tensor`` in the ``data_fileds``,
        such as ``.cuda()``, ``.cpu()``, ``.numpy()``, , ``.to()``
        ``to_tensor()``, ``.detach()``.

    Args:
        metainfo (dict, optional): A dict contains the meta information
            of single image. such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        kwargs (dict, optional): A dict contains annotations of single image or
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
        ...     bboxes=bboxes, scores=scores)
        >>> gt_instances = BaseDataElement(
        ...                    metainfo=dict(img_id=img_id,
        ...                                  img_shape=(H, W)))

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
        >>> assert 'img_shape' not in gt_instances.keys()
        >>> assert 'img_shape' in gt_instances.all_keys()
        >>> print(gt_instances.img_shape)
        >>> gt_instances.scores = torch.rand((5,))
        >>> assert 'scores' in gt_instances.keys()
        >>> assert 'scores' in gt_instances
        >>> assert 'scores' in gt_instances.all_keys()
        >>> assert 'scores' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.scores)
        >>> gt_instances.bboxes = torch.rand((5, 4))
        >>> assert 'bboxes' in gt_instances.keys()
        >>> assert 'bboxes' in gt_instances
        >>> assert 'bboxes' in gt_instances.all_keys()
        >>> assert 'bboxes' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.bboxes)

        >>> # delete and change property
        >>> gt_instances = BaseDataElement(
        ...  metainfo=dict(img_id=0, img_shape=(640, 640)),
        ...  bboxes=torch.rand((6, 4)), scores=torch.rand((6,)))
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
        >>> metainfo = dict(img_shape=(800, 1196, 3))
        >>> gt_instances = BaseDataElement(
        >>>     metainfo=metainfo, det_labels=torch.LongTensor([0, 1, 2, 3]))
        >>> sample = BaseDataElement(metainfo=metainfo,
        ...                          gt_instances=gt_instances)
        >>> print(sample)
        <BaseDataElement(
            META INFORMATION
            img_shape: (800, 1196, 3)
            DATA FIELDS
            gt_instances: <BaseDataElement(
                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    DATA FIELDS
                    det_labels: tensor([0, 1, 2, 3])
                ) at 0x7f0ec5eadc70>
        ) at 0x7f0fea49e130>

        >>> # inheritance
        >>> class DetDataSample(BaseDataElement):
        ...     @property
        ...     def proposals(self):
        ...         return self._proposals
        ...     @proposals.setter
        ...     def proposals(self, value):
        ...         self.set_field(value, '_proposals', dtype=BaseDataElement)
        ...     @proposals.deleter
        ...     def proposals(self):
        ...         del self._proposals
        ...     @property
        ...     def gt_instances(self):
        ...         return self._gt_instances
        ...     @gt_instances.setter
        ...     def gt_instances(self, value):
        ...         self.set_field(value, '_gt_instances',
        ...                        dtype=BaseDataElement)
        ...     @gt_instances.deleter
        ...     def gt_instances(self):
        ...         del self._gt_instances
        ...     @property
        ...     def pred_instances(self):
        ...         return self._pred_instances
        ...     @pred_instances.setter
        ...     def pred_instances(self, value):
        ...         self.set_field(value,'_pred_instances',
        ...                        dtype=BaseDataElement)
        ...     @pred_instances.deleter
        ...     def pred_instances(self):
        ...         del self._pred_instances
        >>> det_sample = DetDataSample()
        >>> proposals = BaseDataElement(bboxes=torch.rand((5, 4)))
        >>> det_sample.proposals = proposals
        >>> assert 'proposals' in det_sample
        >>> assert det_sample.proposals == proposals
        >>> del det_sample.proposals
        >>> assert 'proposals' not in det_sample
        >>> with self.assertRaises(AssertionError):
        ...     det_sample.proposals = torch.rand((5, 4))
    """

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

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
            self.set_field(name=k, value=v, field_type='metainfo', dtype=None)

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
            self.set_field(name=k, value=v, field_type='data', dtype=None)

    def update(self, instance: 'BaseDataElement') -> None:
        """The update() method updates the BaseDataElement with the elements
        from another BaseDataElement object.

        Args:
            instance (BaseDataElement): Another BaseDataElement object for
            update the current object.
        """
        assert isinstance(
            instance, BaseDataElement
        ), f'instance should be a `BaseDataElement` but got {type(instance)}'
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

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
        Returns:
            BaseDataElement: a new data element with same type.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if data is not None:
            new_data.set_data(data)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        """Deep copy the current data element.

        Returns:
            BaseDataElement: the copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
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

    def values(self) -> list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: an iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: an iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: an iterator object whose element is (key, value) tuple
            pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    def __setattr__(self, name: str, value: Any):
        """setattr is only used to set data."""
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable. ')
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f'`{name}` is used in meta information.'
                    'if you want to change the key in metainfo, please use'
                    '`set_metainfo(dict(name=value))`')

            self.set_field(
                name=name, value=value, field_type='data', dtype=None)

    def __delattr__(self, item: str):
        """delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
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

    def get(self, key, default=None) -> Any:
        """get property in data and metainfo as the same as python."""
        return self.__dict__.get(key, default)

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
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(self,
                  value: Any,
                  name: str,
                  dtype: Optional[Type] = None,
                  field_type: str = 'data') -> None:
        """Special method for set union field, used as property.setter
        functions."""
        assert field_type in ['metainfo', 'data']
        if dtype is not None:
            assert isinstance(
                value,
                dtype), f'{value} should be a {dtype} but got {type(value)}'

        super().__setattr__(name, value)
        if field_type == 'metainfo':
            self._metainfo_fields.add(name)
        else:
            self._data_fields.add(name)

    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cpu(self) -> 'BaseDataElement':
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cuda(self) -> 'BaseDataElement':
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def detach(self) -> 'BaseDataElement':
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def numpy(self) -> 'BaseDataElement':
        """Convert all tensor to np.narray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> 'BaseDataElement':
        """Convert all np.narray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        """Convert BaseDataElement to dict."""
        return {
            k: v.to_dict() if isinstance(v, BaseDataElement) else v
            for k, v in self.items()
        }

    def __repr__(self) -> str:

        def _addindent(s_: str, num_spaces: int) -> str:
            """This func is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)  # type: ignore
            s = first + '\n' + s  # type: ignore
            return s  # type: ignore

        def dump(obj: Any) -> str:
            """represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str .
            """
            _repr = ''
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f'\n{k}: {_addindent(dump(v), 4)}'
            elif isinstance(obj, BaseDataElement):
                _repr += '\n\n    META INFORMATION'
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += '\n\n    DATA FIELDS'
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f'<{classname}({_repr}\n) at {hex(id(obj))}>'
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)
