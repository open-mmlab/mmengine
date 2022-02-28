# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Any, Union

from .base_global_accsessible import BaseGlobalAccessible
from .log_buffer import LogBuffer


class MessageHub(BaseGlobalAccessible):
    """Message hub for component interaction. MessageHub is created and
    accessed in the same way as BaseGlobalAccessible.

    ``MessageHub`` will record log information and runtime information. The
    log information refers to the learning rate, loss, etc. of the model
    when training a model, which will stored as ``LogBuffer``. The runtime
    information refers to the iter times, meta information of runner etc.,
    which will be overwritten by next update.

    Args:
        name (str): The name of message hub, for global access. Defaults to ''.
    """

    def __init__(self, name=''):
        self._log_buffers = OrderedDict()
        self._runtime = OrderedDict()
        super().__init__(name)

    def update_log(self, key: str, value: Union[int, float], count=1) -> None:
        """Update the ``LogBuffer`` of the specified key.

        Args:
            key (str): The key of ``LogBuffer``.
            value (Union[int, float]): The value of log.
            count (int): The accumulation times of log, defaults to 1. count
                will be used in smooth statistics.
        """
        if key in self._log_buffers:
            self._log_buffers[key].update(value, count)
        else:
            self._log_buffers[key] = LogBuffer([value], [count])

    def update_info(self, key: str, value: Any) -> None:
        """Update the runtime information of the specified key.

        Args:
            key (str): The key of runtime information.
            value (Any): The value of the runtime information.
        """
        self._runtime[key] = value

    @property
    def log_buffers(self) -> OrderedDict:
        """Get all ``LogBuffer`` instances.

        Returns:
            OrderedDict: All ``LogBuffer`` instances.
        """
        return self._log_buffers

    @property
    def runtime_info(self) -> OrderedDict:
        """Get all runtime information.

        Returns:
            OrderedDict: All runtime information.
        """
        return self._runtime

    def get_log(self, key: str) -> LogBuffer:
        """Get ``LogBuffer`` instance by key.

        Args:
            key: The key of LogBuffer.

        Returns:
            LogBuffer: Corresponding ``LogBuffer`` instance if the key exists.
        """
        if key not in self.log_buffers:
            raise KeyError(f'{key} is not found in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return self._log_buffers[key]

    def get_info(self, key) -> Any:
        """Get runtime information by key.

        Args:
            key: The key of runtime information.

        Returns:
            Any: Corresponding runtime information if the key exists.
        """
        if key not in self.runtime_info:
            raise KeyError(f'{key} is not found in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return self._runtime[key]
