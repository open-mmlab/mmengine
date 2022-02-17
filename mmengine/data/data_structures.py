# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any

import numpy as np
import torch


class BaseDataElement:

    def __init__(self, metainfo: dict = None, data: dict = None):

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if data is not None:
            self.set_data(data)

    def set_metainfo(self, metainfo: dict):
        """Add meta information.

        Args:
            metainfo (dict): A dict contains the meta information
                of image. such as `img_shape`, `scale_factor`, etc.
                Default: None.
        """
        assert isinstance(metainfo,
                          dict), f'meta should be a `dict` but get {metainfo}'
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            if k in self._data_fields:
                raise AttributeError(f'`{k}` is used in data,'
                                     f'which is immutable. if you want to'
                                     f'change the key in data, please use'
                                     f'set_data')
            self._metainfo_fields.add(k)
            self.__dict__[k] = v

    def set_data(self, data: dict):
        """Update a dict to `data_fields`.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions. Default: None.
        """
        assert isinstance(data,
                          dict), f'meta should be a `dict` but get {data}'
        for k, v in data.items():
            self.__setattr__(k, v)

    def new(self,
            metainfo: dict = None,
            data: dict = None) -> 'BaseDataElement':
        """Return a new results with same image meta information.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image. such as `img_shape`, `scale_factor`, etc.
                Default: None.
            data (dict, optional): A dict contains annotations of image or
                model predictions. Default: None.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            metainfo = dict(self.metainfo_items())
            new_data.set_metainfo(metainfo)
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
        return [key for key in self._data_fields]

    def metainfo_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return [key for key in self._metainfo_fields]

    def data_values(self) -> list:
        """
        Returns:
            list: Contains all values in data_fields.
        """
        return [getattr(self, k) for k in self.data_keys()]

    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo_fields.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def keys(self) -> list:
        return self.metainfo_keys() + self.data_keys()

    def values(self) -> list:
        return self.metainfo_values() + self.data_values()

    def items(self) -> list:
        items = list()
        for k in self.keys():
            items.append((k, getattr(self, k)))
        return items

    def data_items(self) -> list:
        items = list()
        for k in self.data_keys():
            items.append((k, getattr(self, k)))
        return items

    def metainfo_items(self) -> list:
        items = list()
        for k in self.metainfo_keys():
            items.append((k, getattr(self, k)))
        return items

    def __setattr__(self, name: str, val: Any):
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, val)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')
        else:
            if name in self._metainfo_fields:
                raise AttributeError(f'`{name}` is used in meta information,'
                                     f'which is immutable. if you want to'
                                     f'change the key in metainfo, please use'
                                     f'set_metainfo(dict(name=val))')

            self._data_fields.add(name)
            super().__setattr__(name, val)

    def __delattr__(self, item: str):

        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a '
                                 f'private attribute, which is immutable. ')
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __setitem__ = __setattr__
    __delitem__ = __delattr__

    def get(self, *args) -> Any:
        assert len(args) < 3, '`get` get more than 2 arguments'
        return self.__dict__.get(*args)

    def pop(self, *args) -> Any:
        assert len(args) < 3, '`pop` get more than 2 arguments'
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
        """Apply same name function to all tensors in data_fields."""
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
        """Apply same name function to all tensors in data_fields."""
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
        """Apply same name function to all tensors in data_fields."""
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
        """Apply same name function to all tensors in data_fields."""
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
        """Apply same name function to all tensors in data_fields."""
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
        repr += '\n   DATA FIELDS \n'
        for k, v in self.data_items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                repr += f'shape of {k}: {v.shape} \n'
            else:
                repr += f'{k}: {v} \n'
        classname = self.__class__.__name__
        return f'<{classname}({repr}\n) at {hex(id(self))}>'
