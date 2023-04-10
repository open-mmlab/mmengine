# Copyright (c) OpenMMLab. All rights reserved.
import atexit
from collections.abc import Iterable
from functools import partial
from multiprocessing import Pool
from typing import Callable, Tuple

import rich
from rich.progress import Progress

from ..utils import is_seq_of
from .timer import Timer


class RichProgressBar:
    """use rich to enhance progress bar."""

    running_progress = False
    bar = Progress()

    def __init__(self):
        if RichProgressBar.running_progress:
            RichProgressBar.bar.stop()
            RichProgressBar.bar = Progress()

        RichProgressBar.running_progress = True
        self.tasks = []
        self.descriptions = []
        self.colors = []
        self.timer = None
        self.completed = 0
        self.infinite = False

        self.bar.start()
        atexit.register(self.cleanup)

    def cleanup(self):
        self.bar.stop()
        RichProgressBar.running_progress = False

    @staticmethod
    def write(msg: str, color: str = 'blue'):
        rich.print(f'[{color}]{msg}')

    def add_task(self,
                 total: int = None,
                 color: str = 'blue',
                 description: str = 'Process...') -> int:
        if total is not None:
            assert not self.infinite, (
                'The prior task is an infinite task (total is None), '
                'RichProgressBar can only accept one infinite task')

            if total <= 0:
                raise ValueError(
                    'Total only exists if it is greater than zero or None.')
            self.tasks.append(
                self.bar.add_task(
                    f'[{color}]{description}_0/{total}', total=total))
            self.colors.append(color)
            self.descriptions.append(description)
            task_id = self.bar.tasks[-1].id
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

    def update(self, task_id: int = 0, advance: int = 1):
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


def worker(params, function):
    result = function(*params)
    return result


def track_progress_rich(func: Callable,
                        tasks: Tuple[list],
                        nproc: int = 1,
                        description: str = 'process...',
                        color: str = 'blue',
                        chunksize: int = 1,
                        skip_first: bool = False,
                        keep_order: bool = True) -> list:
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (tuple[list]): A tuple of tasks.
        nproc (int): Process (worker) number, if nuproc is 1, use single
            process. Default is 1.
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
    assert is_seq_of(tasks,
                     Iterable), 'The content of tasks must be iterable object'
    assert nproc > 0, 'nproc must be a positive number'

    if not hasattr(tasks[0], '__len__'):
        task_num = None
    else:
        assert len(
            {len(arg)
             for arg in tasks}
        ) == 1, 'args must have the same length, ' \
                'please check each argument in your ' \
                'tasks has the same length'
        task_num = len(tasks[0])

    prog_bar = RichProgressBar()

    if nproc == 1:
        prog_bar.add_task(task_num, color=color, description=description)
        results = []
        for task in zip(*tasks):
            results.append(func(*task))
            prog_bar.update()
        return results
    else:
        param = list(zip(*tasks))
        work = partial(worker, function=func)

        pool = Pool(nproc)
        if task_num is not None:
            task_num -= nproc * chunksize * int(skip_first)

        prog_bar.add_task(task_num, description=description, color=color)
        results = []
        if keep_order:
            gen = pool.imap(work, param, chunksize)
        else:
            gen = pool.imap_unordered(work, param, chunksize)
        for result in gen:
            results.append(result)
            if skip_first:
                if len(results) <= nproc * chunksize:
                    continue
            prog_bar.update()
        pool.close()
        pool.join()

        return results
