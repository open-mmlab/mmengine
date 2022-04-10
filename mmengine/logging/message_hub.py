# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Any, Union

import numpy as np
import torch

from mmengine.utils import ManagerMixin
from mmengine.visualization.utils import check_type
from .log_buffer import LogBuffer


class MessageHub(ManagerMixin):
    """Message hub for component interaction. MessageHub is created and
    accessed in the same way as ManagerMixin.

    ``MessageHub`` will record log information and runtime information. The
    log information refers to the learning rate, loss, etc. of the model
    when training a model, which will be stored as ``LogBuffer``. The runtime
    information refers to the iter times, meta information of runner etc.,
    which will be overwritten by next update.

    Args:
        name (str): Name of message hub, for global access. Defaults to ''.
    """

    def __init__(self, name: str = ''):
        self._log_buffers: OrderedDict = OrderedDict()
        self._runtime_info: OrderedDict = OrderedDict()
        super().__init__(name)

    def update_log(self, key: str, value: Union[int, float], count: int = 1) \
            -> None:
        """Update log buffer.

        Args:
            key (str): Key of ``LogBuffer``.
            value (int or float): Value of log.
            count (int): Accumulation times of log, defaults to 1. `count`
                will be used in smooth statistics.
        """
        if key in self._log_buffers:
            self._log_buffers[key].update(value, count)
        else:
            self._log_buffers[key] = LogBuffer([value], [count])

    def update_log_vars(self, log_dict: dict) -> None:
        """Update :attr:`_log_buffers` with a dict.

        Args:
            log_dict (str): Used for batch updating :attr:`_log_buffers`.

        Examples:
            >>> message_hub = MessageHub.get_instance('mmengine')
            >>> log_dict = dict(a=1, b=2, c=3)
            >>> message_hub.update_log_vars(log_dict)
            >>> # The default count of  `a`, `b` and `c` is 1.
            >>> log_dict = dict(a=1, b=2, c=dict(value=1, count=2))
            >>> message_hub.update_log_vars(log_dict)
            >>> # The count of `c` is 2.
        """
        assert isinstance(log_dict, dict), ('`log_dict` must be a dict!, '
                                            f'but got {type(log_dict)}')
        for log_name, log_val in log_dict.items():
            if isinstance(log_val, dict):
                assert 'value' in log_val, \
                    f'value must be defined in {log_val}'
                count = log_val.get('count', 1)
                value = self._get_valid_value(log_name, log_val['value'])
            else:
                value = self._get_valid_value(log_name, log_val)
                count = 1
            self.update_log(log_name, value, count)

    def update_info(self, key: str, value: Any) -> None:
        """Update runtime information.

        Args:
            key (str): Key of runtime information.
            value (Any): Value of runtime information.
        """
        self._runtime_info[key] = value

    @property
    def log_buffers(self) -> OrderedDict:
        """Get all ``LogBuffer`` instances.

        Note:
            Considering the large memory footprint of ``log_buffers`` in the
            post-training, ``MessageHub.log_buffers`` will not return the
            result of ``copy.deepcopy``.

        Returns:
            OrderedDict: All ``LogBuffer`` instances.
        """
        return self._log_buffers

    @property
    def runtime_info(self) -> OrderedDict:
        """Get all runtime information.

        Returns:
            OrderedDict: A copy of all runtime information.
        """
        return copy.deepcopy(self._runtime_info)

    def get_log(self, key: str) -> LogBuffer:
        """Get ``LogBuffer`` instance by key.

        Note:
            Considering the large memory footprint of ``log_buffers`` in the
            post-training, ``MessageHub.get_log`` will not return the
            result of ``copy.deepcopy``.

        Args:
            key (str): Key of ``LogBuffer``.

        Returns:
            LogBuffer: Corresponding ``LogBuffer`` instance if the key exists.
        """
        if key not in self.log_buffers:
            raise KeyError(f'{key} is not found in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return self._log_buffers[key]

    def get_info(self, key: str) -> Any:
        """Get runtime information by key.

        Args:
            key (str): Key of runtime information.

        Returns:
            Any: A copy of corresponding runtime information if the key exists.
        """
        if key not in self.runtime_info:
            raise KeyError(f'{key} is not found in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return copy.deepcopy(self._runtime_info[key])

    def _get_valid_value(self, key: str,
                         value: Union[torch.Tensor, np.ndarray, int, float])\
            -> Union[int, float]:
        """Convert value to python built-in type.

        Args:
            key (str): name of log.
            value (torch.Tensor or np.ndarray or int or float): value of log.

        Returns:
            float or int: python built-in type value.
        """
        if isinstance(value, np.ndarray):
            assert value.size == 1
            value = value.item()
        elif isinstance(value, torch.Tensor):
            assert value.numel() == 1
            value = value.item()
        else:
            check_type(key, value, (int, float))
        return value
