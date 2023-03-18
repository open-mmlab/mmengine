# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from multiprocessing import Pool

from rich.console import Console
from rich.progress import Progress

from .timer import Timer


class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self,
                 task_num=0,
                 start=True,
                 description='Process...',
                 color='blue'):
        self.task_num = task_num
        self.description = description
        self.color = color
        self.bar = Progress()
        self.console = Console()
        self.completed = 0
        if start:
            self.start()

    def __del__(self):
        self.bar.stop()

    def write(self, msg):
        self.console.print(msg, style=self.color)

    def start(self):
        if self.task_num > 0:
            self.task = self.bar.add_task(
                f'[{self.color}]{self.description}', total=self.task_num)
            self.bar.start()
        else:
            self.write('completed: 0, elapsed: 0s')
            self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks

        if self.task_num > 0:
            self.bar.advance(self.task, advance=num_tasks)

            if self.completed == self.task_num:
                self.bar.stop()
        else:
            elapsed = self.timer.since_start()
            if elapsed > 0:
                fps = self.completed / elapsed
            else:
                fps = float('inf')
            self.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')


def track_progress(func,
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
    prog_bar = ProgressBar(task_num, description=description, color=color)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    prog_bar.write('\n')
    return results


def init_pool(process_num, initializer=None, initargs=None):
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)


def track_parallel_progress(func,
                            tasks,
                            nproc,
                            description='Process',
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
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(
        task_num, start, description=description, color=color)
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
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.write('\n')
    pool.close()
    pool.join()
    return results


def track_iter_progress(tasks, description='Process', color='blue'):
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
    prog_bar = ProgressBar(task_num, description=description, color=color)
    for task in tasks:
        yield task
        prog_bar.update()
    prog_bar.write('\n')
