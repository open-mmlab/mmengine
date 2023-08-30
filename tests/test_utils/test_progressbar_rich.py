# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine.utils import track_progress_rich


def foo():
    return 1


def foo1(x):
    return x


def foo2(x, y):
    return x, y


def test_progressbar_rich_exception():
    tasks = [1] * 10
    # Valid func
    with pytest.raises(TypeError):
        track_progress_rich(1, tasks)
    # invalid task
    with pytest.raises(TypeError):
        track_progress_rich(foo1, 1)
    # mismatched task number
    with pytest.raises(ValueError):
        track_progress_rich(foo1, tasks, task_num=9)
    # invalid proc
    with pytest.raises(ValueError):
        track_progress_rich(foo1, tasks, nproc=0)
    # empty tasks and task_num is None
    with pytest.raises(ValueError):
        track_progress_rich(foo1, nproc=0)


@pytest.mark.parametrize('nproc', [1, 2])
def test_progressbar_rich(nproc):
    # empty tasks
    results = track_progress_rich(foo, nproc=nproc, task_num=10)
    assert results == [1] * 10
    # Ordered results
    # foo1
    tasks_ = [i for i in range(10)]
    for tasks in (tasks_, iter(tasks_)):
        results = track_progress_rich(foo1, tasks, nproc=nproc)
        assert results == tasks_
    # foo2
    tasks_ = [(i, i + 1) for i in range(10)]
    for tasks in (tasks_, iter(tasks_)):
        results = track_progress_rich(foo2, tasks, nproc=nproc)
        assert results == tasks_
