# Copyright (c) OpenMMLab. All rights reserved.
import os
import platform
import time
from io import StringIO
from unittest import skipIf
from unittest.mock import patch

import mmengine


def reset_string_io(io):
    io.truncate(0)
    io.seek(0)


class TestProgressBar:

    def test_start(self):
        out = StringIO()
        bar_width = 20
        # without total task num
        prog_bar = mmengine.ProgressBar(bar_width=bar_width, file=out)
        assert out.getvalue() == 'completed: 0, elapsed: 0s'
        reset_string_io(out)
        prog_bar = mmengine.ProgressBar(
            bar_width=bar_width, start=False, file=out)
        assert out.getvalue() == ''
        reset_string_io(out)
        prog_bar.start()
        assert out.getvalue() == 'completed: 0, elapsed: 0s'
        # with total task num
        reset_string_io(out)
        prog_bar = mmengine.ProgressBar(10, bar_width=bar_width, file=out)
        assert out.getvalue() == f'[{" " * bar_width}] 0/10, elapsed: 0s, ETA:'
        reset_string_io(out)
        prog_bar = mmengine.ProgressBar(
            10, bar_width=bar_width, start=False, file=out)
        assert out.getvalue() == ''
        reset_string_io(out)
        prog_bar.start()
        assert out.getvalue() == f'[{" " * bar_width}] 0/10, elapsed: 0s, ETA:'

    @skipIf(
        platform.system() != 'Linux',
        reason='Only test `TestProgressBar.test_update` in Linux')
    def test_update(self):
        out = StringIO()
        bar_width = 20
        # without total task num
        prog_bar = mmengine.ProgressBar(bar_width=bar_width, file=out)
        time.sleep(1)
        reset_string_io(out)
        prog_bar.update()
        assert out.getvalue() == 'completed: 1, elapsed: 1s, 1.0 tasks/s'
        reset_string_io(out)
        # with total task num
        prog_bar = mmengine.ProgressBar(10, bar_width=bar_width, file=out)
        time.sleep(1)
        reset_string_io(out)
        prog_bar.update()
        assert out.getvalue() == f'\r[{">" * 2 + " " * 18}] 1/10, 1.0 ' \
                                 'task/s, elapsed: 1s, ETA:     9s'

    @skipIf(
        platform.system() != 'Linux',
        reason='Only test `TestProgressBar.test_adaptive_length` in Linux')
    def test_adaptive_length(self):
        with patch.dict('os.environ', {'COLUMNS': '80'}):
            out = StringIO()
            bar_width = 20
            prog_bar = mmengine.ProgressBar(10, bar_width=bar_width, file=out)
            time.sleep(1)
            reset_string_io(out)
            prog_bar.update()
            assert len(out.getvalue()) == 66

            os.environ['COLUMNS'] = '30'
            reset_string_io(out)
            prog_bar.update()
            assert len(out.getvalue()) == 48

            os.environ['COLUMNS'] = '60'
            reset_string_io(out)
            prog_bar.update()
            assert len(out.getvalue()) == 60


def sleep_1s(num):
    time.sleep(1)
    return num


def return_itself(num):
    return num


def test_track_progress():
    # tasks is a list
    out = StringIO()
    ret = mmengine.track_progress(sleep_1s, [1, 2, 3], bar_width=3, file=out)
    if platform == 'Linux':
        assert out.getvalue() == (
            '[   ] 0/3, elapsed: 0s, ETA:'
            '\r[>  ] 1/3, 1.0 task/s, elapsed: 1s, ETA:     2s'
            '\r[>> ] 2/3, 1.0 task/s, elapsed: 2s, ETA:     1s'
            '\r[>>>] 3/3, 1.0 task/s, elapsed: 3s, ETA:     0s\n')
    assert ret == [1, 2, 3]

    # tasks is an iterable object
    ret = mmengine.track_progress(
        return_itself, ((i for i in [1, 2, 3]), 3), bar_width=3, file=out)
    assert ret == [1, 2, 3]

    # tasks is a range object
    ret = mmengine.track_progress(
        return_itself, range(1, 4), bar_width=3, file=out)
    assert ret == [1, 2, 3]


def test_track_iter_progress():
    out = StringIO()
    ret = []
    for num in mmengine.track_iter_progress([1, 2, 3], bar_width=3, file=out):
        ret.append(num)

    assert ret == [1, 2, 3]

    ret = []
    count = []
    for i, num in enumerate(
            mmengine.track_iter_progress([1, 2, 3], bar_width=3, file=out)):
        ret.append(num)
        count.append(i)
    assert ret == [1, 2, 3]
    assert count == [0, 1, 2]

    # tasks is a range object
    res = mmengine.track_iter_progress(range(1, 4), bar_width=3, file=out)
    assert list(res) == [1, 2, 3]


def test_track_parallel_progress():
    # tasks is a list
    out = StringIO()
    ret = mmengine.track_parallel_progress(
        return_itself, [1, 2, 3, 4], 2, bar_width=4, file=out)
    assert ret == [1, 2, 3, 4]

    # tasks is an iterable object
    ret = mmengine.track_parallel_progress(
        return_itself, ((i for i in [1, 2, 3, 4]), 4),
        2,
        bar_width=4,
        file=out)
    assert ret == [1, 2, 3, 4]

    # tasks is a range object
    ret = mmengine.track_parallel_progress(
        return_itself, range(1, 5), 2, bar_width=4, file=out)
    assert ret == [1, 2, 3, 4]
