# Copyright (c) OpenMMLab. All rights reserved.
import threading
import time
from collections.abc import Iterable
from multiprocessing import Pool
from typing import List

from rich.console import Console
from rich.progress import Progress

from .timer import Timer


class RichProgressBar:
    """use rich to enhance progress bar."""

    def __init__(self):
        self.bar = Progress()
        self.console = Console()
        self.tasks = []
        self.colors = []

        self.bar.start()

    def __del__(self):
        self.bar.stop()

    def write(self, msg, color='blue'):
        self.console.print(msg, style=color)

    def add_multi_task(self, total, color='blue', description='Process...'):
        assert total > 0
        self.colors.append(color)
        self.tasks.append(
            self.bar.add_task(f'[{color}]{description}', total=total))

    def add_single_task(self, total, color='blue', description='Process...'):
        self.colors.append(color)
        if total > 0:
            self.tasks.append(
                self.bar.add_task(f'[{color}]{description}', total=total))
            self.infinite = False
        else:
            self.write('completed: 0, elapsed: 0s', self.colors[0])
            self.timer = Timer()
            self.infinite = True
            self.completed = 0
            self.tasks.append(0)

    def is_task_finish(self, task_id):
        completed = self.bar.tasks[task_id].completed
        total = self.bar.tasks[task_id].total
        return completed == total

    def finished(self):
        return self.bar.finished

    def update(self, task_id=0, advance=1):
        assert advance > 0

        if len(self.tasks) == 1:
            assert task_id == 0
            if not self.infinite:
                self.bar.advance(self.tasks[task_id], advance=advance)

                if self.is_task_finish(task_id):
                    self.bar.stop()
            else:
                self.completed += advance
                elapsed = self.timer.since_start()
                if elapsed > 0:
                    fps = self.completed / elapsed
                else:
                    fps = float('inf')
                self.write(
                    f'completed: {self.completed},'
                    f' elapsed: {int(elapsed + 0.5)}s,'
                    f' {fps:.1f} tasks/s', self.colors[task_id])
        else:
            assert task_id >= 0 and task_id < len(self.tasks)
            self.bar.update(self.tasks[task_id], advance=advance)
            time.sleep(0.01)

            if self.bar.finished:
                self.bar.stop()

    def reset(self):
        for i in range(len(self.tasks)):
            self.bar.remove_task(self.tasks[i])
        self.tasks.clear()
        self.colors.clear()


def track_single_progress(func,
                          tasks,
                          description='Process...',
                          color='blue',
                          **kwargs):
    """Track the progress of tasks execution with a progress bar.

    Tasks are done with a simple for-loop.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        description (str): The description of progress bar.
        color (str): The color of progress bar.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = RichProgressBar()
    prog_bar.add_single_task(task_num, color=color, description=description)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    return results


def track_multi_progress(funcs, tasks, descriptions, colors, params=None):
    """Track multi progress of tasks execution with progress bar.

    Tasks are done with a while-loop and a simple for-loop.

    Args:
        funcs (list): Functions apply for each task.
        tasks (list): A list of tasks.
        descriptions (str or list): The descriptions of each progress bar.
        colors (str or list): The colors of each progress bar.
        params (list): The funcs` parameters.

    Returns:
        list: The task results.
    """
    assert isinstance(funcs, List)
    assert isinstance(tasks, List)
    if params is None:
        params = [dict() for i in range(len(funcs))]

    if isinstance(descriptions, str):
        descriptions = [descriptions] * len(tasks)
    if isinstance(colors, str):
        colors = [colors] * len(tasks)

    assert len(funcs) == len(tasks) == len(params) == len(descriptions) == len(
        colors)

    prog_bar = RichProgressBar()
    for i in range(len(tasks)):
        total = len(tasks[i])
        prog_bar.add_multi_task(total, colors[i], descriptions[i])

    result = [[] for i in range(len(tasks))]
    idx = 0
    while not prog_bar.finished():
        for task_id in range(len(tasks)):
            if not prog_bar.is_task_finish(task_id=task_id):
                result[task_id].append(funcs[task_id](tasks[task_id][idx],
                                                      **params[task_id]))
                prog_bar.update(task_id, 1)
            else:
                continue
        idx += 1
    return result


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_single_parallel_progress(func,
                                   tasks,
                                   nproc,
                                   description='process...',
                                   color='blue',
                                   initializer=None,
                                   initargs=None,
                                   chunksize=1,
                                   skip_first=False,
                                   keep_order=True):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        description (str): The description of progress bar.
        color (str): The color of progress bar.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = RichProgressBar()
    prog_bar.add_single_task(task_num, description=description, color=color)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                continue
        prog_bar.update()
    pool.close()
    pool.join()
    return results


def track_single_iter_progress(tasks, description='Process..', color='blue'):
    """Track the progress of tasks iteration or enumeration with a progress
    bar.

    Tasks are yielded with a simple for-loop.

    Args:
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        description (str): The description of progress bar.
        color (str): The color of progress bar.

    Yields:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = RichProgressBar()
    prog_bar.add_single_task(task_num, description=description, color=color)
    for task in tasks:
        prog_bar.update(0, 1)
        yield task


class MultiThread(threading.Thread):

    def __init__(self, task, func, idx, update_func, param=None):
        super().__init__()
        self.task = task
        self.func = func
        self.idx = idx
        self.updata_func = update_func
        self.result = []
        self.param = param

    def run(self) -> None:
        for i in range(len(self.task)):
            self.result.append(self.func(self.task[i], **self.param))
            self.updata_func(self.idx)

    def get_value(self):
        return self.result


def track_multi_parallel_progress(funcs,
                                  tasks,
                                  descriptions,
                                  colors,
                                  params=None):
    """Track multi progress of tasks execution with progress bar and
    MultiThread.

    After accepting a task, threads will be created based on the number of
     tasks and tasks will be executed in parallel.

    Args:
        funcs (list): Functions apply for each task.
        tasks (list): A list of tasks.
        descriptions (str or list): The descriptions of each progress bar.
        colors (str or list): The colors of each progress bar.
        params (list): The funcs` parameters.

    Returns:
        list: The task results.
    """
    assert isinstance(funcs, List)
    assert isinstance(tasks, List)
    if params is None:
        params = [dict() for i in range(len(funcs))]

    if isinstance(descriptions, str):
        descriptions = [descriptions] * len(tasks)
    if isinstance(colors, str):
        colors = [colors] * len(tasks)

    assert len(funcs) == len(tasks) == len(params) == len(descriptions) == len(
        colors)

    prog_bar = RichProgressBar()
    for i in range(len(tasks)):
        total = len(tasks[i])
        prog_bar.add_multi_task(total, colors[i], descriptions[i])

    process = []
    for i in range(len(tasks)):
        proc = MultiThread(tasks[i], funcs[i], i, prog_bar.update, params[i])
        proc.start()
        process.append(proc)

    for i in range(len(process)):
        process[i].join()

    result = []
    for i in range(len(process)):
        result.append(process[i].get_value())

    return result
