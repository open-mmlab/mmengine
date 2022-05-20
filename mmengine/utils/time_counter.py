# Copyright (c) OpenMMLab. All rights reserved.
import time
import warnings
from typing import Optional, Union

import torch

from mmengine.dist.utils import master_only
from mmengine.logging import MMLogger, print_log


class TimeCounter:
    """A tool for counting function or method average time. You can use it as a
    decorator or context manager to calculate the average time.

    Args:
        log_interval (int): The interval of logging. Defaults to 1.
        warmup_interval (int): The interval of warmup. Defaults to 1.
        with_sync (bool): Whether to synchronize cuda. Defaults to True.
        tag (str, optional): Function tag. Used to distinguish between
            different functions or methods being called. Defaults to None.
        logger (MMLogger, optional): Formatted logger used to record messages.
                Defaults to None.

        Examples:
            >>> import time
            >>> from mmengine.utils import TimeCounter
            >>> @TimeCounter()
            ... def fun1():
            ...     time.sleep(0.1)
            ... fun1()
            [fun1]-1-times per count: 100.0 ms

            >>> @@TimeCounter(log_interval=2, tag='fun')
            ... def fun2():
            ...    time.sleep(0.2)
            >>> for _ in range(3):
            ...    fun2()
            [fun]-2-times per count: 200.0 ms

            >>> with TimeCounter(tag='fun3'):
            ...      time.sleep(0.3)
            [fun3]-1-times per count: 300.0 ms
    """

    def __init__(self,
                 log_interval: int = 1,
                 warmup_interval: int = 1,
                 with_sync: bool = True,
                 tag: Optional[str] = None,
                 logger: Optional[MMLogger] = None):
        assert warmup_interval >= 1
        self.log_interval = log_interval
        self.warmup_interval = warmup_interval
        self.with_sync = with_sync
        self.with_sync = with_sync
        self.tag = tag
        self.logger = logger

        self.__count = 0
        self.__pure_inf_time = 0.
        self.__start_time = 0.

    @master_only
    def __call__(self, fn):
        if self.tag is None:
            self.tag = fn.__name__

        def wrapper(*args, **kwargs):
            self.__count += 1

            if self.with_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            result = fn(*args, **kwargs)

            if self.with_sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            self.print_time(elapsed)

            return result

        return wrapper

    @master_only
    def __enter__(self):
        if self.tag is None:
            self.tag = 'default'
            warnings.warn('In order to clearly distinguish printing '
                          'information in different contexts, please specify '
                          'the tag parameter')

        self.__count += 1

        if self.with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.__start_time = time.perf_counter()

    @master_only
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.__start_time
        self.print_time(elapsed)

    def print_time(self, elapsed: Union[int, float]) -> None:
        """print times per count."""
        if self.__count >= self.warmup_interval:
            self.__pure_inf_time += elapsed

            if self.__count % self.log_interval == 0:
                times_per_count = 1000 * self.__pure_inf_time / (
                    self.__count - self.warmup_interval + 1)
                print_log(
                    f'[{self.tag}]-{self.__count}-times per count: '
                    f'{times_per_count:.1f} ms', self.logger)
