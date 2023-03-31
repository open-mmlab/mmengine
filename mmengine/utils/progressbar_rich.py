# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from multiprocessing import Pool

import rich
from rich.progress import Progress

from ..utils import is_seq_of
from .timer import Timer


class RichProgressBar:
    """use rich to enhance progress bar."""

    def __init__(self):
        self.bar = Progress()
        self.tasks = []
        self.descriptions = []
        self.colors = []
        self.timer = None
        self.completed = 0
        self.infinite = False

        self.bar.start()

    def __del__(self):
        self.bar.stop()

    def write(self, msg, color='blue'):
        rich.print(f'[{color}]{msg}')

    def add_task(self, total=None, color='blue', description='Process...'):
        if total is not None:
            assert not self.infinite, (
                'The prior task is an infinite task (total is None),'
                ' RichProgressBar can only accept one infinite task')

            if total <= 0:
                raise ValueError(
                    'Total only exists if it is greater than zero or None.')
            self.tasks.append(
                self.bar.add_task(
                    f'[{color}]{description}_0/{total}', total=total))
            self.colors.append(color)
            self.descriptions.append(description)
            task_id = len(self.tasks) - 1
        else:
            assert not self.tasks, (
                'Since the total argument is None, this task is considered '
                'infinite. RichProgressBar should not have any other tasks '
                'added to it when an infinite task is present.')
            self.write('completed: 0, elapsed: 0s', color)
            self.timer = Timer()
            self.infinite = True
            self.completed = 0
            self.tasks.append(0)
            self.colors.append(color)
            self.descriptions.append(description)
            task_id = 0
        return task_id

    def update(self, task_id=0, advance=1):
        if advance <= 0:
            raise ValueError('advance should greater than zero.')

        if self.infinite:
            self.completed += advance
            elapsed = self.timer.since_start()
            if elapsed > 0:
                fps = self.completed / elapsed
            else:
                fps = float('inf')
            self.write(
                f'completed: {self.completed}, '
                f'elapsed: {int(elapsed + 0.5)}s, '
                f'{fps:.1f} tasks/s', self.colors[task_id])
        else:
            assert task_id >= -1 and task_id < len(self.tasks), (
                'The task_id must be within a valid range')

            if task_id == -1:
                task_id = len(self.tasks) - 1

            completed = self.bar.tasks[task_id].completed + advance
            total = self.bar.tasks[task_id].total
            self.bar.update(
                task_id,
                advance=advance,
                description=f'[{self.colors[task_id]}]'
                f'{self.descriptions[task_id]}'
                f'_{completed}/{total}')

            if self.bar.finished:
                self.bar.stop()


def track_single_progress(func,
                          tasks,
                          task_num=None,
                          description='Process...',
                          color='blue'):
    """Track the progress of tasks execution with a progress bar.

    Tasks are done with a simple for-loop.

    Args:
        func (callable): The function to be applied to each task.
        tasks (tuple[Iterable]): A tuple of tasks.
        task_num (int): Number of tasks. Default is None.
        description (str): The description of progress bar.
        color (str): The color of progress bar.

    Returns:
        list: The task results.
    """
    assert is_seq_of(tasks, Iterable)
    assert len({len(arg) for arg in tasks}) == 1, 'args must have same length'
    if task_num is not None:
        assert task_num == len(
            tasks[0]), ('task_num should be same as arg length')

    prog_bar = RichProgressBar()
    prog_bar.add_task(task_num, color=color, description=description)
    results = []
    for task in zip(*tasks):
        results.append(func(*task))
        prog_bar.update()
    return results


def worker(params):
    func = params[0]
    param = params[1]
    result = func(*param)
    return result


def track_single_parallel_progress(func,
                                   tasks,
                                   task_num=None,
                                   nproc=1,
                                   description='process...',
                                   color='blue',
                                   chunksize=1,
                                   skip_first=False,
                                   keep_order=True):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (tuple[Iterable]): A tuple of tasks.
        task_num (int): Number of tasks. Default is None.
        nproc (int): Process (worker) number. Default is 1.
        description (str): The description of progress bar.
        color (str): The color of progress bar.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    assert is_seq_of(tasks, Iterable)
    assert len({len(arg) for arg in tasks}) == 1, 'args must have same length'
    if task_num is not None:
        assert task_num == len(
            tasks[0]), ('task_num should be same as arg length')

    input_param = []
    param = list(zip(*tasks))
    for i in range(len(tasks[0])):
        input_param.append([func, param[i]])

    pool = Pool(nproc)
    if task_num is not None:
        task_num -= nproc * chunksize * int(skip_first)
    prog_bar = RichProgressBar()
    prog_bar.add_task(task_num, description=description, color=color)
    results = []
    if keep_order:
        gen = pool.imap(worker, input_param, chunksize)
    else:
        gen = pool.imap_unordered(worker, input_param, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) <= nproc * chunksize:
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
        assert len(tasks) == 2, (
            '"tasks" must be composed of two elements (task, task_num)')
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
    prog_bar.add_task(task_num, description=description, color=color)
    for task in tasks:
        prog_bar.update(0, 1)
        yield task
