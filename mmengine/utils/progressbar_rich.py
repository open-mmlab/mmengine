# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from functools import partial
from multiprocessing import Pool
from typing import Callable, Tuple

import rich
from rich.progress import Progress

from ..utils import is_seq_of
from .timer import Timer


class RichProgressBar:
    """Use rich to enhance progress bar.

    This class uses the rich library to enhance the progressbar,
    Do not update two RichProgressbars at the same time.

    Examples:
        >>> import mmengine
        >>> import time
        >>> bar = mmengine.RichProgressBar()
        >>> bar.add_task(10)
        >>> for i in range(10):
        >>>     bar.update()
        >>>     time.sleep(1)
    """

    def __init__(self, *args, **kwargs):
        self.bar = Progress(*args, **kwargs)
        self.tasks = []
        self.descriptions = []
        self.colors = []
        self.timer = None
        self.completed = 0
        self.infinite = False

    @staticmethod
    def write(msg: str, color: str = 'blue') -> None:
        """Write the massage.

        Args:
            msg (str): Output massage.
            color (str, optional): Text color.
        """
        rich.print(f'[{color}]{msg}')

    def add_task(self,
                 total: int = None,
                 color: str = 'blue',
                 description: str = 'Process...') -> int:
        """Adding tasks to RichProgressbar.

        Args:
            total (int, optional): Number of total steps, When the input is
                None, it indicates a task with unknown length. Defaults to
                None.
            color (str): Color of progress bar. Defaults to "blue".
            description (str): Description of progress bar.
                Defaults to "Process...".
        Returns:
            int: Added task`s id.
        """
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

    def update(self, task_id: int = 0, advance: int = 1) -> None:
        """update progressbar.

        Args:
            task_id (int): Task ID that needs to be updated. Defaults to 0.
            advance (int): Update step size. Defaults to 1.
        """
        if advance <= 0:
            raise ValueError('advance should be greater than zero.')

        if self.infinite:
            if task_id != 0:
                raise ValueError('In Infinite mode, task_ ID must be 0.')

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
            if task_id < 0 or task_id > len(self.tasks):
                raise ValueError('The task_id must be within a valid range')

            # Activate self.bar before all tasks have been updated.
            if all(task.completed == 0 for task in self.bar.tasks):
                self.bar.start()

            completed = self.bar.tasks[task_id].completed + advance
            total = self.bar.tasks[task_id].total
            self.bar.update(
                task_id,
                advance=advance,
                description=f'[{self.colors[task_id]}]'
                f'{self.descriptions[task_id]}'
                f'_{completed}/{total}')

            # After all tasks are completed, deactivate self.bar.
            if all(task.finished for task in self.bar.tasks):
                self.bar.stop()


def worker(params, function):
    """Used for multithreaded functions."""
    result = function(*params)
    return result


def track_progress_rich(func: Callable,
                        tasks: Tuple[list],
                        nproc: int = 1,
                        description: str = 'Process...',
                        color: str = 'blue',
                        chunksize: int = 1,
                        skip_first: bool = False,
                        keep_order: bool = True) -> list:
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (Tuple[list]): A tuple of tasks.
        nproc (int): Process (worker) number, if nuproc is 1,
            use single process. Defaults to 1.
        description (str): The description of progress bar.
            Defaults to "Process".
        color (str): The color of progress bar. Defaults to "blue".
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
            Defaults to 1.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer. Defaults to False.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used. Defaults to True.

    Returns:
        list: The task results.
    """
    if not is_seq_of(tasks, Iterable):
        raise ValueError('The content of tasks must be iterable object')

    if nproc <= 0:
        raise ValueError('nproc must be a positive number')

    if not hasattr(tasks[0], '__len__'):
        task_num = None
    else:
        if len({len(arg) for arg in tasks}) != 1:
            raise ValueError('args must have the same length, '
                             'please check each argument in your '
                             'tasks has the same length')

        task_num = len(tasks[0])

    prog_bar = RichProgressBar()

    # Use single process when nproc is 1, else use multiprocess.
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

        try:
            for result in gen:
                results.append(result)
                if skip_first:
                    if len(results) <= nproc * chunksize:
                        continue
                prog_bar.update()
        except Exception as e:
            prog_bar.bar.stop()
            raise e

        pool.close()
        pool.join()

        return results
