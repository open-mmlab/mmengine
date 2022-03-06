# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from typing import Any, Union

from .base_global_accsessible import BaseGlobalAccessible
from .log_buffer import LogBuffer


class MessageHub(BaseGlobalAccessible):
    """Message hub for component interaction. MessageHub is created and
    accessed in the same way as BaseGlobalAccessible.

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
