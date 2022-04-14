# Copyright (c) OpenMMLab. All rights reserved.
import itertools
from typing import List, Union

import numpy as np
import torch

from .base_data_element import BaseDataElement

IndexType = Union[str, slice, int, torch.LongTensor, torch.cuda.LongTensor,
                  torch.BoolTensor, torch.cuda.BoolTensor, np.long, np.bool]


# Modified from
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/data_structures/instance_data.py # noqa
class InstanceData(BaseDataElement):
    """Data structure for instance-level annnotations or predictions.

    Subclass of :class:`BaseDataElement`. All value in `data_fields`
    should have the same length. This design refer to
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501

    Examples:
        >>> from mmengine.data import InstanceData
        >>> import numpy as np
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> instance_data = InstanceData(metainfo=img_meta)
        >>> 'img_shape' in instance_data
        True
        >>> instance_data.det_labels = torch.LongTensor([2, 3])
        >>> instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
        >>> instance_data.bboxes = torch.rand((2, 4))
        >>> len(instance_data)
        4
        >>> print(instance_data)
        <InstanceData(

            META INFORMATION
            pad_shape: (800, 1196, 3)
            img_shape: (800, 1216, 3)

            DATA FIELDS
            det_labels: tensor([2, 3])
            det_scores: tensor([0.8, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                [0.8101, 0.3105, 0.5123, 0.6263]])
            ) at 0x7fb492de6280>
        >>> sorted_results = instance_data[instance_data.det_scores.sort().indices]
        >>> sorted_results.det_scores
        tensor([0.7000, 0.8000])
        >>> print(instance_data[instance_data.det_scores > 0.75])
        <InstanceData(

            META INFORMATION
            pad_shape: (800, 1216, 3)
            img_shape: (800, 1196, 3)

            DATA FIELDS
            det_labels: tensor([0])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188]])
            det_scores: tensor([0.8000])
        ) at 0x7fb5cf6e2790>
        >>> instance_data[instance_data.det_scores > 0.75].det_labels
        tensor([0])
        >>> instance_data[instance_data.det_scores > 0.75].det_scores
        tensor([0.8000])
    """

    def __setattr__(self, name: str, value: Union[torch.Tensor, np.ndarray,
                                                  list]):
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(
                    f'{name} has been used as a '
                    f'private attribute, which is immutable. ')

        else:
            assert isinstance(value, (torch.Tensor, np.ndarray, list)), \
                f'Can set {type(value)}, only support' \
                f' {(torch.Tensor, np.ndarray, list)}'

            if len(self) > 0:
                assert len(value) == len(self), f'the length of ' \
                                                f'values {len(value)} is ' \
                                                f'not consistent with' \
                                                f' the length of this ' \
                                                f':obj:`InstanceData` ' \
                                                f'{len(self)} '
            super().__setattr__(name, value)

    def __getitem__(self, item: IndexType) -> 'InstanceData':
        """
        Args:
            item (str, obj:`slice`,
                obj`torch.LongTensor`, obj:`torch.BoolTensor`):
                get the corresponding values according to item.

        Returns:
            obj:`InstanceData`: Corresponding values.
        """
        assert len(self) > 0, ' This is a empty instance'

        assert isinstance(
            item, (str, slice, int, torch.LongTensor, torch.cuda.LongTensor,
                   torch.BoolTensor, torch.cuda.BoolTensor, np.bool, np.long))

        if isinstance(item, str):
            return getattr(self, item)

        if type(item) == int:
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.new(data={})
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, 'Only support to get the' \
                                    ' values along the first dimension.'
            if isinstance(item, (torch.BoolTensor, torch.cuda.BoolTensor)):
                assert len(item) == len(self), f'The shape of the' \
                                               f' input(BoolTensor)) ' \
                                               f'{len(item)} ' \
                                               f' does not match the shape ' \
                                               f'of the indexed tensor ' \
                                               f'in results_filed ' \
                                               f'{len(self)} at ' \
                                               f'first dimension. '

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(v, list):
                    r_list = []
                    # convert to indexes from boolTensor
                    if isinstance(item,
                                  (torch.BoolTensor, torch.cuda.BoolTensor)):
                        indexes = torch.nonzero(item).view(-1)
                    else:
                        indexes = item
                    for index in indexes:
                        r_list.append(v[index])
                    new_data[k] = r_list
        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type:ignore

    @staticmethod
    def cat(instances_list: List['InstanceData']) -> 'InstanceData':
        """Concat the instances of all :obj:`InstanceData` in the list.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`InstanceData`]): A list
                of :obj:`InstanceData`.

        Returns:
            obj:`InstanceData`
        """
        assert all(
            isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [
            instances.all_keys() for instances in instances_list
        ]
        assert len(set([len(field_keys) for field_keys in field_keys_list])) \
               == 1 and len(set(itertools.chain(*field_keys_list))) \
               == len(field_keys_list[0]), 'There are different keys in ' \
                                           '`instances_list`, which may ' \
                                           'cause the cat operation ' \
                                           'to fail. Please make sure all ' \
                                           'elements in `instances_list` ' \
                                           'have the exact same key '

        new_data = instances_list[0].new(data={})
        for k in instances_list[0].keys():
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                values = np.concatenate(values, axis=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            else:
                raise ValueError(
                    f'Can not concat the {k} which is a {type(v0)}')
            new_data[k] = values
        return new_data  # type:ignore

    def __len__(self) -> int:
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0
