# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Any, Optional, Union

import numpy as np
import torch

from mmengine.utils import ManagerMixin
from mmengine.visualization.utils import check_type
from .history_buffer import HistoryBuffer


class MessageHub(ManagerMixin):
    """Message hub for component interaction. MessageHub is created and
    accessed in the same way as ManagerMixin.

    ``MessageHub`` will record log information and runtime information. The
    log information refers to the learning rate, loss, etc. of the model
    during training phase, which will be stored as ``HistoryBuffer``. The
    runtime information refers to the iter times, meta information of
    runner etc., which will be overwritten by next update.

    Args:
        name (str): Name of message hub used to get corresponding instance
            globally.
        log_scalars (OrderedDict, optional): Each key-value pair in the
            dictionary is the name of the log information such as "loss", "lr",
            "metric" and their corresponding values. The type of value must be
            HistoryBuffer. Defaults to None.
        runtime_info (OrderedDict, optional): Each key-value pair in the
            dictionary is the name of the runtime information and their
            corresponding values. Defaults to None.
        resumed_keys (OrderedDict, optional): Each key-value pair in the
            dictionary decides whether the key in :attr:`_log_scalars` and
            :attr:`_runtime_info` will be serialized.

    Note:
        Key in :attr:`_resumed_keys` belongs to :attr:`_log_scalars` or
        :attr:`_runtime_info`. The corresponding value cannot be set
        repeatedly.

    Examples:
        >>> # create empty `MessageHub`.
        >>> message_hub1 = MessageHub()
        >>> log_scalars = OrderedDict(loss=HistoryBuffer())
        >>> runtime_info = OrderedDict(task='task')
        >>> resumed_keys = dict(loss=True)
        >>> # create `MessageHub` from data.
        >>> message_hub2 = MessageHub(
        >>>     name='name',
        >>>     log_scalars=log_scalars,
        >>>     runtime_info=runtime_info,
        >>>     resumed_keys=resumed_keys)
    """

    def __init__(self,
                 name: str,
                 log_scalars: Optional[OrderedDict] = None,
                 runtime_info: Optional[OrderedDict] = None,
                 resumed_keys: Optional[OrderedDict] = None):
        super().__init__(name)
        self._log_scalars = log_scalars if log_scalars is not None else \
            OrderedDict()
        self._runtime_info = runtime_info if runtime_info is not None else \
            OrderedDict()
        self._resumed_keys = resumed_keys if resumed_keys is not None else \
            OrderedDict()

        assert isinstance(self._log_scalars, OrderedDict)
        assert isinstance(self._runtime_info, OrderedDict)
        assert isinstance(self._resumed_keys, OrderedDict)

        for value in self._log_scalars.values():
            assert isinstance(value, HistoryBuffer), \
                ("The type of log_scalars'value must be HistoryBuffer, but "
                 f'got {type(value)}')

        for key in self._resumed_keys.keys():
            assert key in self._log_scalars or key in self._runtime_info, \
                ('Key in `resumed_keys` must contained in `log_scalars` or '
                 f'`runtime_info`, but got {key}')

    def update_scalar(self,
                      key: str,
                      value: Union[int, float, np.ndarray, torch.Tensor],
                      count: int = 1,
                      resumed: bool = True) -> None:
        """Update :attr:_log_scalars.

        Update ``HistoryBuffer`` in :attr:`_log_scalars`. If corresponding key
        ``HistoryBuffer`` has been created, ``value`` and ``count`` is the
        argument of ``HistoryBuffer.update``, Otherwise, ``update_scalar``
        will create an ``HistoryBuffer`` with value and count via the
        constructor of ``HistoryBuffer``.

        Examples:
            >>> message_hub = MessageHub
            >>> # create loss `HistoryBuffer` with value=1, count=1
            >>> message_hub.update_scalar('loss', 1)
            >>> # update loss `HistoryBuffer` with value
            >>> message_hub.update_scalar('loss', 3)
            >>> message_hub.update_scalar('loss', 3, resumed=False)
            AssertionError: loss used to be true, but got false now. resumed
            keys cannot be modified repeatedly'

        Note:
            resumed cannot be set repeatedly for the same key.

        Args:
            key (str): Key of ``HistoryBuffer``.
            value (torch.Tensor or np.ndarray or int or float): Value of log.
            count (torch.Tensor or np.ndarray or int or float): Accumulation
                times of log, defaults to 1. `count` will be used in smooth
                statistics.
            resumed (str): Whether the corresponding ``HistoryBuffer``
                could be resumed. Defaults to True.
        """
        self._set_resumed_keys(key, resumed)
        checked_value = self._get_valid_value(key, value)
        assert isinstance(count, int), (
            f'The type of count must be int. but got {type(count): {count}}')
        if key in self._log_scalars:
            self._log_scalars[key].update(checked_value, count)
        else:
            self._log_scalars[key] = HistoryBuffer([checked_value], [count])

    def update_scalars(self, log_dict: dict, resumed: bool = True) -> None:
        """Update :attr:`_log_scalars` with a dict.

        ``update_scalars`` iterates through each pair of log_dict key-value,
        and calls ``update_scalar``. If type of value is dict, the value should
        be ``dict(value=xxx) or dict(value=xxx, count=xxx)``. Item in
        ``log_dict`` has the same resume option.

        Args:
            log_dict (str): Used for batch updating :attr:`_log_scalars`.
            resumed (bool): Whether all ``HistoryBuffer`` referred in
                log_dict should be resumed. Defaults to True.

        Examples:
            >>> message_hub = MessageHub.get_instance('mmengine')
            >>> log_dict = dict(a=1, b=2, c=3)
            >>> message_hub.update_scalars(log_dict)
            >>> # The default count of  `a`, `b` and `c` is 1.
            >>> log_dict = dict(a=1, b=2, c=dict(value=1, count=2))
            >>> message_hub.update_scalars(log_dict)
            >>> # The count of `c` is 2.
        """
        assert isinstance(log_dict, dict), ('`log_dict` must be a dict!, '
                                            f'but got {type(log_dict)}')
        for log_name, log_val in log_dict.items():
            self._set_resumed_keys(log_name, resumed)
            if isinstance(log_val, dict):
                assert 'value' in log_val, \
                    f'value must be defined in {log_val}'
                count = self._get_valid_value(log_name,
                                              log_val.get('count', 1))
                checked_value = self._get_valid_value(log_name,
                                                      log_val['value'])
            else:
                count = 1
                checked_value = self._get_valid_value(log_name, log_val)
            assert isinstance(count,
                              int), ('The type of count must be int. but got '
                                     f'{type(count): {count}}')
            self.update_scalar(log_name, checked_value, count)

    def update_info(self, key: str, value: Any, resumed: bool = True) -> None:
        """Update runtime information.

        The key corresponding runtime information will be overwritten each
        time calling ``update_info``.

        Note:
            resumed cannot be set repeatedly for the same key.

        Examples:
            >>> message_hub = MessageHub()
            >>> message_hub.update_info('iter', 100)

        Args:
            key (str): Key of runtime information.
            value (Any): Value of runtime information.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        self._set_resumed_keys(key, resumed)
        self._resumed_keys[key] = resumed
        self._runtime_info[key] = value

    def _set_resumed_keys(self, key: str, resumed: bool) -> None:
        """Set corresponding resumed keys.

        This method is called by ``update_scalar``, ``update_scalars`` and
        ``update_info`` to set the corresponding key is true or false in
        :attr:`_resumed_keys`.

        Args:
            key (str): Key of :attr:`_log_scalrs` or :attr:`_runtime_info`.
            resumed (bool): Whether the corresponding ``HistoryBuffer``
                could be resumed.
        """
        if key not in self._resumed_keys:
            self._resumed_keys[key] = resumed
        else:
            assert self._resumed_keys[key] == resumed, \
                f'{key} used to be {self._resumed_keys[key]}, but got ' \
                '{resumed} now. resumed keys cannot be modified repeatedly'

    @property
    def log_scalars(self) -> OrderedDict:
        """Get all ``HistoryBuffer`` instances.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will return a reference of
            history buffer rather than a copy.

        Returns:
            OrderedDict: All ``HistoryBuffer`` instances.
        """
        return self._log_scalars

    @property
    def runtime_info(self) -> OrderedDict:
        """Get all runtime information.

        Returns:
            OrderedDict: A copy of all runtime information.
        """
        # return copy.deepcopy(self._runtime_info)
        return self._runtime_info

    def get_scalar(self, key: str) -> HistoryBuffer:
        """Get ``HistoryBuffer`` instance by key.

        Note:
            Considering the large memory footprint of history buffers in the
            post-training, :meth:`get_scalar` will not return a reference of
            history buffer rather than a copy.

        Args:
            key (str): Key of ``HistoryBuffer``.

        Returns:
            HistoryBuffer: Corresponding ``HistoryBuffer`` instance if the
            key exists.
        """
        if key not in self.log_scalars:
            raise KeyError(f'{key} is not found in Messagehub.log_buffers: '
                           f'instance name is: {MessageHub.instance_name}')
        return self.log_scalars[key]

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

        # TODO： There are restrictions on objects that can be saved
        # return copy.deepcopy(self._runtime_info[key])
        return self._runtime_info[key]

    def _get_valid_value(self, key: str,
                         value: Union[torch.Tensor, np.ndarray, int, float]) \
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
        return value  # type: ignore

    def __getstate__(self):
        for key in list(self._log_scalars.keys()):
            assert key in self._resumed_keys, (
                f'Cannot found {key} in {self}._resumed_keys, '
                'please make sure you do not change the _resumed_keys '
                'outside the class')
            if not self._resumed_keys[key]:
                self._log_scalars.pop(key)

        for key in list(self._runtime_info.keys()):
            assert key in self._resumed_keys, (
                f'Cannot found {key} in {self}._resumed_keys, '
                'please make sure you do not change the _resumed_keys '
                'outside the class')
            if not self._resumed_keys[key]:
                self._runtime_info.pop(key)
        return self.__dict__
