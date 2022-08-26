# Copyright (c) OpenMMLab. All rights reserved.
import time

import pytest

import mmengine


def test_timer_init():
    timer = mmengine.Timer(start=False)
    assert not timer.is_running
    timer.start()
    assert timer.is_running
    timer = mmengine.Timer()
    assert timer.is_running


def test_timer_run():
    timer = mmengine.Timer()
    time.sleep(1)
    assert abs(timer.since_start() - 1) < 1e-2
    time.sleep(1)
    assert abs(timer.since_last_check() - 1) < 1e-2
    assert abs(timer.since_start() - 2) < 1e-2
    timer = mmengine.Timer(False)
    with pytest.raises(mmengine.TimerError):
        timer.since_start()
    with pytest.raises(mmengine.TimerError):
        timer.since_last_check()


def test_timer_context(capsys):
    with mmengine.Timer():
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert abs(float(out) - 1) < 1e-2
    with mmengine.Timer(print_tmpl='time: {:.1f}s'):
        time.sleep(1)
    out, _ = capsys.readouterr()
    assert out == 'time: 1.0s\n'
