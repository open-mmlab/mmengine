# Copyright (c) OpenMMLab. All rights reserved.
import time

import mmengine


class TestRichProgressBar:

    def test_start(self):
        # single task
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_single_task(100)
        assert len(prog_bar.bar.tasks) == 1
        assert prog_bar.bar.tasks[0].total == 100
        assert prog_bar.infinite is False
        del prog_bar
        # multi tasks
        prog_bar = mmengine.RichProgressBar()
        for i in range(5):
            prog_bar.add_multi_tasks(10)
        assert len(prog_bar.bar.tasks) == 5
        del prog_bar
        # without total task num
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_single_task(None)
        assert prog_bar.infinite is True
        del prog_bar

    def test_update(self):
        # single task
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_single_task(10)
        for i in range(10):
            prog_bar.update()
        assert prog_bar.bar.tasks[0].finished is True
        del prog_bar
        # without total task num
        prog_bar = mmengine.RichProgressBar()
        prog_bar.add_single_task(None)
        for i in range(10):
            prog_bar.update()
        assert prog_bar.completed == 10
        del prog_bar
        # multi task
        prog_bar = mmengine.RichProgressBar()
        for i in range(10):
            prog_bar.add_multi_tasks(10)
        for i in range(10):
            for task_id in range(10):
                prog_bar.update(task_id)
        assert prog_bar.bar.finished is True
        for i in range(10):
            assert prog_bar.bar.tasks[i].finished is True
        del prog_bar


def sleep_1s(num):
    time.sleep(1)
    return num


def add(x, y):
    time.sleep(1)
    return x + y


def test_track_single_progress_with_task_num():
    ret = mmengine.track_single_progress(
        add, ([1, 2, 3], [4, 5, 6]), task_num=3)
    assert ret == [5, 7, 9]


def test_track_single_progress_without_task_num():
    ret = mmengine.track_single_progress(add, ([1, 2, 3], [4, 5, 6]))
    assert ret == [5, 7, 9]


def test_track_single_iter_progress():
    ret = []
    for num in mmengine.track_single_iter_progress([1, 2, 3]):
        ret.append(sleep_1s(num))
    assert ret == [1, 2, 3]


def test_track_single_enum_progress():
    ret = []
    count = []
    for i, num in enumerate(mmengine.track_single_iter_progress([1, 2, 3])):
        ret.append(sleep_1s(num))
        count.append(i)
    assert ret == [1, 2, 3]
    assert count == [0, 1, 2]


def test_track_single_parallel_progress_list():
    results = mmengine.track_single_parallel_progress(sleep_1s, [1, 2, 3, 4],
                                                      2)
    assert results == [1, 2, 3, 4]


def test_track_single_parallel_progress_list_with_static_param():
    results = mmengine.track_single_parallel_progress(
        add, [1, 2, 3, 4], 2, static_params={'y': 1})
    assert results == [2, 3, 4, 5]


def test_track_single_parallel_progress_iterator():
    results = mmengine.track_single_parallel_progress(
        sleep_1s, ((i for i in [1, 2, 3, 4]), 4), 2)
    assert results == [1, 2, 3, 4]


def test_track_multi_parallel_progress():
    ret = mmengine.track_multi_parallel_progress([sleep_1s for i in range(2)],
                                                 [[1, 2, 3], [1, 2]],
                                                 description='task',
                                                 color='blue')
    assert ret[0] == [1, 2, 3] and ret[1] == [1, 2]
