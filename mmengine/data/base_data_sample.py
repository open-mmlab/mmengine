# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Iterator, Optional, Tuple, Type

import numpy as np
import torch

from .base_data_element import BaseDataElement


class BaseDataSample:
    """A base data structure interface of OpenMMlab.

    A sample data consists of input data (such as an image) and its annotations
    and predictions. In general, an image can have multiple types of
    annotations and/or predictions at the same time (for example, both
    pixel-level semantic segmentation annotations and instance-level detection
    bboxes annotations). To facilitate data access of multitask, MMEngine
    defines ``BaseDataSample`` as the base class for sample data encapsulation.
    **The attributes of ``BaseDataSample`` will be various types of data
    elements**, and the codebases in OpenMMLab need to implement their own
    xxxDataSample such as ClsDataSample, DetDataSample, SegDataSample based on
    ``BaseDataSample`` to encapsulate all relevant data, as a data
    interface between dataset, model, visualizer, and evaluator components.

    These attributes in ``BaseDataElement`` are divided into two parts,
    the ``metainfo`` and the ``data`` respectively.

        - ``metainfo``: Usually contains the
          information about the image such as filename,
          image_shape, pad_shape, etc. The attributes can be accessed or
          modified by dict-like or object-like operations, such as
          ``.``(only for data access) , ``in``, ``del``, ``pop(str)``,
          ``get(str)``, ``metainfo_keys()``, ``metainfo_values()``,
          ``metainfo_items()``, ``set_metainfo()``(for set or change value
          in metainfo).

        - ``data``: Annotations or model predictions are
          stored. The attributes can be accessed or modified by
          dict-like or object-like operations, such as
          ``.`` , ``in``, ``del``, ``pop(str)`` ``get(str)``, ``data_keys()``,
          ``data_values()``, ``data_items()``. Users can also apply tensor-like
          methods to all obj:``torch.Tensor`` in the ``data_fileds``,
          such as ``.cuda()``, ``.cpu()``, ``.numpy()``, , ``.to()``,
          ``to_tensor()``, ``.detach()``, ``.numpy()``

    Args:
        meta_info (dict, optional): A dict contains the meta information
            of a sample. such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        data (dict, optional): A dict contains annotations of a sample or
            model predictions. Defaults to None.

    Examples:
        >>> from mmengine.data import BaseDataElement, BaseDataSample
        >>> gt_instances = BaseDataSample()

        >>> bboxes = torch.rand((5, 4))
        >>> scores = torch.rand((5,))
        >>> img_id = 0
        >>> img_shape = (800, 1333)
        >>> gt_instances = BaseDataElement(
                metainfo=dict(img_id=img_id, img_shape=img_shape),
                data=dict(bboxes=bboxes, scores=scores))
        >>> data = dict(gt_instances=gt_instances)
        >>> sample = BaseDataSample(
                        metainfo=dict(img_id=img_id, img_shape=img_shape),
                        data=data)
        >>> sample = BaseDataSample(dict(img_id=img_id,
                                          img_shape=(H, W)))
        # new
        >>> data1 = dict(bboxes=torch.rand((5, 4)),
                      scores=torch.rand((5,)))
        >>> metainfo1 = dict(img_id=1, img_shape=(640, 640)),
        >>> gt_instances1 = BaseDataElement(
                metainfo=metainfo1,
                data=data1)
        >>> sample1 = sample.new(
                            metainfo=metainfo1
                            data=dict(gt_instances1=gt_instances1)),

        >>> gt_instances2 = gt_instances1.new()

        # property add and access
        >>> sample = BaseDataSample()
        >>> gt_instances = BaseDataElement(
                metainfo=dict(img_id=9, img_shape=(100, 100)),
                data=dict(bboxes=torch.rand((5, 4)), scores=torch.rand((5,)))
        >>> sample.set_metainfo(dict(img_id=9, img_shape=(100, 100))
        >>> assert 'img_shape' in sample.metainfo_keys()
        >>> assert 'img_shape' in sample
        >>> assert 'img_shape' not in sample.data_keys()
        >>> assert 'img_shape' in sample.keys()
        >>> print(sample.img_shape)

        >>> gt_instances.gt_instances = gt_instances
        >>> assert 'gt_instances' in sample.data_keys()
        >>> assert 'gt_instances' in sample
        >>> assert 'gt_instances' in sample.keys()
        >>> assert 'gt_instances' not in sample.metainfo_keys()
        >>> print(sample.gt_instances)

        >>> pred_instances = BaseDataElement(
                metainfo=dict(img_id=9, img_shape=(100, 100)),
                data=dict(bboxes=torch.rand((5, 4)), scores=torch.rand((5,))
        >>> sample.pred_instances = pred_instances
        >>> assert 'pred_instances' in sample.data_keys()
        >>> assert 'pred_instances' in sample
        >>> assert 'pred_instances' in sample.keys()
        >>> assert 'pred_instances' not in sample.metainfo_keys()
        >>> print(sample.pred_instances)

        # property delete and change
        >>> metainfo=dict(img_id=0, img_shape=(640, 640)
        >>> gt_instances = BaseDataElement(
             metainfo=metainfo)，
             data=dict(bboxes=torch.rand((6, 4)), scores=torch.rand((6,))))
        >>> sample = BaseDataSample(metainfo=metainfo,
                                    data=dict(gt_instances=gt_instances))
        >>> sample.img_shape = (1280, 1280)
        >>> sample.img_shape  # (1280, 1280)
        >>> sample.gt_instances = gt_instances
        >>> sample.get('img_shape', None)  # (640， 640)
        >>> sample.get('gt_instances', None)
        >>> del sample.img_shape
        >>> del sample.gt_instances
        >>> assert 'img_shape' not in sample
        >>> assert 'gt_instances' not in sample
        >>> sample.pop('img_shape', None)  # None
        >>> sample.pop('gt_instances', None)  # None

        # Tensor-like
        >>> cuda_sample = gt_instasamplences.cuda()
        >>> cuda_sample = gt_sample.to('cuda:0')
        >>> cpu_sample = cuda_sample.cpu()
        >>> cpu_sample = cuda_sample.to('cpu')
        >>> fp16_sample = cuda_sample.to(
             device=None, dtype=torch.float16, non_blocking=False, copy=False,
             memory_format=torch.preserve_format)
        >>> cpu_sample = cuda_sample.detach()
        >>> np_sample = cpu_sample.numpy()

        # print
        >>> metainfo = dict(img_shape=(800, 1196, 3))
        >>> gt_instances = BaseDataElement(
             metainfo=metainfo,
             data=dict(det_labels=torch.LongTensor([0, 1, 2, 3])))

        >>> data = dict(gt_instances=gt_instances)
        >>> sample = BaseDataSample(metainfo=metainfo, data=data)
        >>> print(sample)
        <BaseDataSample('
          META INFORMATION '
        img_shape: (800, 1196, 3) '
          DATA FIELDS '
        gt_instances:<BaseDataElement('
          META INFORMATION '
        img_shape: (800, 1196, 3) '
          DATA FIELDS '
        shape of det_labels: torch.Size([4]) '
        ) at 0x7f9705daecd0>'
        ) at 0x7f981e41c550>'

        # inheritance
        >>> class DetDataSample(BaseDataSample):
        >>>     proposals = property(
        >>>         fget=partial(BaseDataSample.get_field, name='_proposals'),
        >>>         fset=partial(
        >>>             BaseDataSample.set_field,
        >>>             name='_proposals',
        >>>             dtype=BaseDataElement),
        >>>         fdel=partial(BaseDataSample.del_field, name='_proposals'),
        >>>         doc='Region proposals of an image')
        >>>     gt_instances = property(
        >>>         fget=partial(BaseDataSample.get_field,
                                 name='_gt_instances'),
        >>>         fset=partial(
        >>>             BaseDataSample.set_field,
        >>>             name='_gt_instances',
        >>>             dtype=BaseDataElement),
        >>>         fdel=partial(BaseDataSample.del_field,
                                 name='_gt_instances'),
        >>>         doc='Ground truth instances of an image')
        >>>     pred_instances = property(
        >>>         fget=partial(
        >>>             BaseDataSample.get_field, name='_pred_instances'),
        >>>         fset=partial(
        >>>             BaseDataSample.set_field,
        >>>             name='_pred_instances',
        >>>             dtype=BaseDataElement),
        >>>         fdel=partial(
        >>>             BaseDataSample.del_field, name='_pred_instances'),
        >>>         doc='Predicted instances of an image')

        >>> det_sample = DetDataSample()
        >>> proposals = BaseDataElement(data=dict(bboxes=torch.rand((5, 4))))
        >>> det_sample.proposals = proposals
        >>> assert 'proposals' in det_sample
        >>> assert det_sample.proposals == proposals
        >>> del det_sample.proposals
        >>> assert 'proposals' not in det_sample
        >>> with self.assertRaises(AssertionError):
                det_sample.proposals = torch.rand((5, 4))
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
            metainfo, dict), f'meta should be a ``dict`` but got {metainfo}'
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
                model predictions. Defaults to None.
        """
        assert isinstance(data,
                          dict), f'meta should be a ``dict`` but got {data}'
        for k, v in data.items():
            self.set_field(name=k, value=v, field_type='data', dtype=None)

    def new(self,
            metainfo: Optional[dict] = None,
            data: Optional[dict] = None) -> 'BaseDataSample':
        """Return a new data element with same type. If ``metainfo`` and
        ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new results will overwrite
        it with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image. such as ``img_shape``, ``scale_factor``, etc.
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

    def __setattr__(self, name: str, value: Any):
        """setattr is only used to set data."""
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f'``{name}`` is used in meta information.'
                    'If you want to change the key in metainfo, please use'
                    'set_metainfo(dict(name=val))')

            self.set_field(
                name=name, value=value, field_type='data', dtype=None)

    def __delattr__(self, item: str):
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a '
                                 'private attribute, which is immutable. ')
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        else:
            self._data_fields.remove(item)

    # dict-like methods
    __setitem__ = __setattr__
    __delitem__ = __delattr__

    def get(self, *args) -> Any:
        """get property in data and metainfo as the same as python."""
        assert len(args) < 3, f'``get`` get more than 2 arguments {args}'
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

    def get_field(self, name: str) -> Any:
        """Special method for get union field, used as property.getter
        functions."""
        return getattr(self, name)

    # It's must to keep the parameters order ``value``, ``name``, for
    # ``partial(BaseDataSample.set_field,
    #          name='_proposals', dtype=BaseDataElement)``
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

    def del_field(self, name: str) -> None:
        """Special method for deleting union field, used as property.deleter
        functions."""
        self.__delattr__(name)

    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'BaseDataSample':
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
    def cpu(self) -> 'BaseDataSample':
        """Convert all tensors to CPU in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        for k, v in self.metainfo_items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cpu()
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    # Tensor-like methods
    def cuda(self) -> 'BaseDataSample':
        """Convert all tensors to GPU in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)

        for k, v in self.metainfo_items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.cuda()
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    # Tensor-like methods
    def detach(self) -> 'BaseDataSample':
        """Detach all tensors in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        for k, v in self.metainfo_items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach()
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    # Tensor-like methods
    def numpy(self) -> 'BaseDataSample':
        """Convert all tensor　to np.narray in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        for k, v in self.metainfo_items():
            if isinstance(v, (torch.Tensor, BaseDataElement)):
                v = v.detach().cpu().numpy()
                metainfo = {k: v}
                new_data.set_metainfo(metainfo)
        return new_data

    def to_tensor(self) -> 'BaseDataSample':
        """Convert all np.narray to tensor in metainfo and data."""
        new_data = self.new()
        for k, v in self.data_items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data.update({k: v})
            elif isinstance(v, (BaseDataElement, BaseDataSample)):
                v = v.to_tensor()
                data.update({k: v})
            new_data.set_data(data)
        for k, v in self.metainfo_items():
            data = {}
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                data.update({k: v})
            elif isinstance(v, (BaseDataElement, BaseDataSample)):
                v = v.to_tensor()
                data.update({k: v})
            new_data.set_metainfo(data)
        return new_data

    def __repr__(self) -> str:
        _repr = '\n  META INFORMATION \n'
        for k, v in self.metainfo_items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                _repr += f'shape of {k}: {v.shape} \n'
            elif isinstance(v, (BaseDataElement, BaseDataSample)):
                _repr += f'{k}:{repr(v)}\n'
            else:
                _repr += f'{k}: {v} \n'
        _repr += '\n  DATA FIELDS \n'
        for k, v in self.data_items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                _repr += f'shape of {k}: {v.shape} \n'
            elif isinstance(v, (BaseDataElement, BaseDataSample)):
                _repr += f'{k}:{repr(v)}\n'
            else:
                _repr += f'{k}: {v} \n'
        classname = self.__class__.__name__
        return f'<{classname}({_repr}\n) at {hex(id(self))}>'
