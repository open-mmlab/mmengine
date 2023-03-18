# Copyright (c) OpenMMLab. All rights reserved.
import time

import mmengine


class TestProgressBar:

    def test_start(self):
        # without total task num
        prog_bar = mmengine.ProgressBar()
        assert len(prog_bar.bar.tasks) == 0
        del prog_bar
        prog_bar = mmengine.ProgressBar(start=False)
        assert len(prog_bar.bar.tasks) == 0
        prog_bar.start()
        assert len(prog_bar.bar.tasks) == 0
        del prog_bar
        # with total task num
        prog_bar = mmengine.ProgressBar(10)
        assert len(prog_bar.bar.tasks) == 1
        del prog_bar
        prog_bar = mmengine.ProgressBar(10, start=False)
        assert len(prog_bar.bar.tasks) == 0
        prog_bar.start()
        assert len(prog_bar.bar.tasks) == 1

    def test_update(self):
        # without total task num
        prog_bar = mmengine.ProgressBar()
        time.sleep(1)
        prog_bar.update()
        assert prog_bar.completed == 1
        del prog_bar
        # with total task num
        prog_bar = mmengine.ProgressBar(10)
        time.sleep(1)
        prog_bar.update()
        assert prog_bar.bar.tasks[0].completed == 1

    def test_description(self):
        # without description and color
        prog_bar = mmengine.ProgressBar(10)
        assert prog_bar.description == 'Process...' \
               and prog_bar.color == 'blue'
        del prog_bar
        # with description
        prog_bar = mmengine.ProgressBar(10, description='mmengine')
        assert prog_bar.description == 'mmengine' and prog_bar.color == 'blue'
        del prog_bar
        # with color
        prog_bar = mmengine.ProgressBar(10, color='red')
        assert prog_bar.description == 'Process...' and prog_bar.color == 'red'
        del prog_bar
        # with description and color
        prog_bar = mmengine.ProgressBar(
            10, description='mmengine', color='red')
        assert prog_bar.description == 'mmengine' and prog_bar.color == 'red'


def sleep_1s(num):
    time.sleep(1)
    return num


def test_track_progress_list():
    ret = mmengine.track_progress(sleep_1s, [1, 2, 3])
    assert ret == [1, 2, 3]


def test_track_progress_iterator():
    ret = mmengine.track_progress(sleep_1s, ((i for i in [1, 2, 3]), 3))
    assert ret == [1, 2, 3]


def test_track_iter_progress():
    ret = []
    for num in mmengine.track_iter_progress([1, 2, 3]):
        ret.append(sleep_1s(num))
    assert ret == [1, 2, 3]


def test_track_enum_progress():
    ret = []
    count = []
    for i, num in enumerate(mmengine.track_iter_progress([1, 2, 3])):
        ret.append(sleep_1s(num))
        count.append(i)
    assert ret == [1, 2, 3]
    assert count == [0, 1, 2]


def test_track_parallel_progress_list():
    results = mmengine.track_parallel_progress(sleep_1s, [1, 2, 3, 4], 2)
    assert results == [1, 2, 3, 4]


def test_track_parallel_progress_iterator():
    results = mmengine.track_parallel_progress(sleep_1s,
                                               ((i for i in [1, 2, 3, 4]), 4),
                                               2)
    assert results == [1, 2, 3, 4]
