# Copyright (c) OpenMMLab. All rights reserved.
import time

import pytest

import mmengine


class TestRichProgressBar:

    def test_init(self):
        prog_bar = mmengine.RichProgressBar()
        assert mmengine.RichProgressBar.running_progress is True
        assert len(prog_bar.tasks) == 0
        assert len(prog_bar.colors) == 0
        assert len(prog_bar.descriptions) == 0
        assert prog_bar.infinite is False
        assert prog_bar.timer is None
        assert prog_bar.completed == 0

    def test_add(self):
        # single task
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_task(100)
        assert len(prog_bar.bar.tasks) == 1
        assert prog_bar.bar.tasks[0].total == 100
        assert prog_bar.infinite is False
        # multi tasks
        prog_bar = mmengine.RichProgressBar()
        for i in range(5):
            prog_bar.add_task(10)
        assert len(prog_bar.bar.tasks) == 5
        # without total task num
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_task(None)
        assert prog_bar.infinite is True
        # test assert
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_task(None)
        with pytest.raises(AssertionError):
            prog_bar.add_task(10)
        with pytest.raises(AssertionError):
            prog_bar.add_task(None)
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_task(10)
        with pytest.raises(AssertionError):
            prog_bar.add_task(None)

    def test_update(self):
        # single task
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_task(10)
        for i in range(10):
            prog_bar.update()
        assert prog_bar.bar.tasks[0].finished is True
        # without total task num
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_task(None)
        for i in range(10):
            prog_bar.update()
        assert prog_bar.completed == 10
        # multi task
        prog_bar = mmengine.RichProgressBar()
        task_ids = []
        for i in range(10):
            task_ids.append(prog_bar.add_task(10))
        for i in range(10):
            for idx in task_ids:
                prog_bar.update(idx)
        assert prog_bar.bar.finished is True
        for idx in task_ids:
            assert prog_bar.bar.tasks[idx].finished is True


def add(x, y):
    time.sleep(1)
    return x + y


def test_track_progress():
    ret = mmengine.track_progress_rich(add, ([1, 2, 3], [4, 5, 6]))
    assert ret == [5, 7, 9]


def test_track_parallel_progress():
    results = mmengine.track_progress_rich(
        add, ([1, 2, 3], [4, 5, 6]), nproc=3)
    assert results == [5, 7, 9]


def test_track_parallel_progress_skip_first():
    results = mmengine.track_progress_rich(
        add, ([1, 2, 3, 4], [2, 3, 4, 5]), nproc=2, skip_first=True)
    assert results == [3, 5, 7, 9]


def test_track_parallel_progress_iterator():
    results = mmengine.track_progress_rich(
        add, (iter([1, 2, 3, 4]), iter([2, 3, 4, 5])), nproc=3)
    assert results == [3, 5, 7, 9]
