# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing
import time
from collections.abc import Iterable
from multiprocessing import Pool
from typing import List

import rich
from rich.progress import Progress

from .timer import Timer


class RichProgressBar:
    """use rich to enhance progress bar."""

    def __init__(self):
        self.bar = Progress()
        self.tasks = []
        self.descriptions = []
        self.colors = []
        self.bar.start()

    def __del__(self):
        self.bar.stop()

    def write(self, msg, color='blue'):
        rich.print(f'[{color}]{msg}')

    def add_multi_tasks(self, total, color='blue', description='Process...'):
        if total <= 0:
            raise ValueError('total should be greater than 0')
        self.descriptions.append(description)
        self.colors.append(color)
        self.tasks.append(
            self.bar.add_task(
                f'[{color}]{description}_0/{total}', total=total))

    def add_single_task(self, total, color='blue', description='Process...'):
        assert len(self.tasks) == 0
        self.descriptions.append(description)
        self.colors.append(color)
        if total is not None:
            if total <= 0:
                raise ValueError(
                    'Total only exists if it is greater than zero or None.')
            self.tasks.append(
                self.bar.add_task(
                    f'[{color}]{description}_0/{total}', total=total))
            self.infinite = False
        else:
            self.write('completed: 0, elapsed: 0s', self.colors[0])
            self.timer = Timer()
            self.infinite = True
            self.completed = 0
            self.tasks.append(0)

    def _is_task_finish(self, task_id):
        completed = self.bar.tasks[task_id].completed
        total = self.bar.tasks[task_id].total
        return completed == total

    def _finished(self):
        return self.bar.finished

    def update(self, task_id=0, advance=1):
        if advance <= 0:
            raise ValueError('advance should greater than zero.')

        if len(self.tasks) == 1:
            if task_id != 0:
                raise ValueError('The ID of a single task can only be 0.')
            if not self.infinite:
                completed = self.bar.tasks[task_id].completed + 1
                total = self.bar.tasks[task_id].total
                self.bar.update(
                    task_id,
                    advance=advance,
                    description=f'[{self.colors[task_id]}]'
                    f'{self.descriptions[task_id]}'
                    f'_{completed}/{total}')

                if self._is_task_finish(task_id):
                    self.bar.stop()
            else:
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
            assert task_id >= 0 and task_id < len(self.tasks)
            completed = self.bar.tasks[task_id].completed + 1
            total = self.bar.tasks[task_id].total
            self.bar.update(
                task_id,
                advance=advance,
                description=f'[{self.colors[task_id]}]'
                f'{self.descriptions[task_id]}'
                f'_{completed}/{total}')

            if self._finished():
                self.bar.stop()


def track_single_progress(func,
                          tasks,
                          task_num=None,
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
        tasks = tasks[0]
        if task_num is not None:
            assert task_num == tasks[1]
    elif isinstance(tasks, Iterable):
        if task_num is not None:
            assert task_num == len(tasks)
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


def worker(func, task, send_end, param=dict()):
    result = []
    for i in range(len(task)):
        result.append(func(task[i], **param))
    send_end.send(result)


def track_multi_parallel_progress(funcs,
                                  tasks,
                                  description,
                                  color='blue',
                                  num_process=3,
                                  params=None):
    """Track multi progress of tasks execution with progress bar and
    MultiProgress.

    After accepting a task, progresses will be created based on the number of
     tasks and tasks will be executed in parallel.

    Args:
        funcs (list): Functions apply for each task.
        tasks (list): A list of tasks.
        description (str): The description of progress bar.
        color (str): The colors of progress bar.
        num_process (int): The number of maximum processes.
        params (list): The funcs` parameters.

    Returns:
        list: The task results.
    """
    assert isinstance(funcs, List)
    assert isinstance(tasks, List)
    if params is None:
        params = [dict() for i in range(len(funcs))]

    assert len(funcs) == len(tasks) == len(params)

    num_tasks = len(tasks)

    prog_bar = RichProgressBar()
    prog_bar.add_single_task(num_tasks, color=color, description=description)

    pool = Pool(processes=num_process)

    process_list = []
    pipe_list = []
    for i in range(num_tasks):
        recv_end, send_end = multiprocessing.Pipe(False)
        p = pool.apply_async(worker, (funcs[i], tasks[i], send_end, params[i]))
        process_list.append(p)
        pipe_list.append(recv_end)
    pool.close()

    finished = [0 for i in range(num_tasks)]
    while True:
        time.sleep(0.01)
        for i in range(num_tasks):
            if process_list[i].ready() and finished[i] == 0:
                finished[i] = 1
                prog_bar.update()
        if sum(finished) == num_tasks:
            break

    result = [pipe.recv() for pipe in pipe_list]
    return result
