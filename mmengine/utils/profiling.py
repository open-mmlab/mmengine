# Copyright (c) OpenMMLab. All rights reserved.
import time
from contextlib import contextmanager
from typing import Callable, Optional

import torch

from mmengine.dist.utils import master_only
from mmengine.logging import MMLogger


@master_only
def decorate_timer(func: Callable,
                   log_interval: int = 1,
                   warmup_interval: int = 1,
                   with_sync: bool = True,
                   name: Optional[str] = None,
                   logger: Optional[MMLogger] = None):
    """A tool for counting function or method average time. You can use it as a
    decorator on a function that wants to calculate the average time.

    Args:
        func (Callable): Decorated function.
        log_interval (int): The interval of logging. Defaults to 1.
        warmup_interval (int): The interval of warmup. Defaults to 1.
        with_sync (bool): Whether to synchronize cuda. Defaults to True.
        name (str, optional): Function alias. Use func name if None.
            Defaults to None.
        logger (MMLogger, optional): Formatted logger used to record messages.
            Defaults to None.

    Examples:
        >>> from mmengine.utils import decorate_timer
        >>> @decorate_timer
        ... def fun():
        ...    time.sleep(1)

        >>> from functools import partial
        >>> @decorate_timer(log_interval=2, name='fun_1')
        ... def fun():
        ...    time.sleep(1)
    """
    assert warmup_interval >= 1

    if name is None:
        name = func.__name__

    func.__count = 0  # type: ignore
    func.__pure_inf_time = 0  # type: ignore

    def wrapper(*args, **kwargs):
        __count = func.__count
        __pure_inf_time = func.__pure_inf_time

        __count += 1
        func.__count = __count

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        if with_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time

        if __count >= warmup_interval:
            __pure_inf_time += elapsed
            func.__pure_inf_time = __pure_inf_time

            if __count % log_interval == 0:
                times_per_count = 1000 * __pure_inf_time / (
                    __count - warmup_interval + 1)

                if logger is not None:
                    logger.info(f'[{name}]-{__count}-times per count: '
                                f'{times_per_count:.1f} ms')
                else:
                    print(
                        f'[{name}]-{__count}-times per count: '
                        f'{times_per_count:.1f} ms',
                        flush=True)

        return result

    return wrapper


__function_dict: dict = dict()


@master_only
@contextmanager
def context_timer(name,
                  log_interval=1,
                  warmup_interval=1,
                  with_sync=True,
                  logger=None):
    """A tool for counting function or method average time. You can use it as a
    context manager to calculate the average time.

    Args:
        log_interval (int): The interval of logging. Defaults to 1.
        warmup_interval (int): The interval of warmup. Defaults to 1.
        with_sync (bool): Whether to synchronize cuda. Defaults to True.
        name (str, optional): Function alias. Use func name if None.
            Defaults to None.
        logger (MMLogger, optional): Formatted logger used to record messages.
            Defaults to None.

    Examples:
        >>> from mmdet.utils import context_timer
        >>> def fun():
        ...     with context_timer(name='func1'):
        ...         time.sleep(1)

        >>> def fun():
        ...     with context_timer(name='func2', log_interval=2):
        ...         time.sleep(1)
    """
    assert warmup_interval >= 1

    if name in __function_dict:
        __count = __function_dict[name]['count']
        __pure_inf_time = __function_dict[name]['pure_inf_time']
        __log_interval = __function_dict[name]['log_interval']
        __warmup_interval = __function_dict[name]['warmup_interval']
        __with_sync = __function_dict[name]['with_sync']
        __logger = __function_dict[name]['logger']
    else:
        __count = 0
        __pure_inf_time = 0
        __log_interval = log_interval
        __warmup_interval = warmup_interval
        __with_sync = with_sync
        __logger = logger
        __function_dict[name] = dict(
            count=__count,
            pure_inf_time=__pure_inf_time,
            log_interval=__log_interval,
            warmup_interval=__warmup_interval,
            with_sync=__with_sync,
            logger=__logger)

    __count += 1
    __function_dict[name]['count'] = __count

    if __with_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    yield

    if with_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    if __count >= __warmup_interval:
        __pure_inf_time += elapsed
        __function_dict[name]['pure_inf_time'] = __pure_inf_time

        if __count % log_interval == 0:
            times_per_count = 1000 * __pure_inf_time / (
                __count - __warmup_interval + 1)
            if __logger is not None:
                __logger.info(f'[{name}]-{__count}-times per count: '
                              f'{times_per_count:.1f} ms')
            else:
                print(
                    f'[{name}]-{__count}-times per count: '
                    f'{times_per_count:.1f} ms',
                    flush=True)
