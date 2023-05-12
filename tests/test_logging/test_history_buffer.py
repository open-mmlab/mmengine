# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import pytest

from mmengine.logging import HistoryBuffer

array_method = [np.array, lambda x: x]
try:
    import torch
except ImportError:
    pass
else:
    array_method.append(torch.tensor)


@HistoryBuffer.register_statistics
def custom_statistics(self):
    return -1


class TestLoggerBuffer:

    def test_init(self):
        log_buffer = HistoryBuffer()
        assert log_buffer.max_length == 1000000
        log_history, counts = log_buffer.data
        assert len(log_history) == 0
        assert len(counts) == 0
        # test the length of array exceed `max_length`
        logs = np.random.randint(1, 10, log_buffer.max_length + 1)
        counts = np.random.randint(1, 10, log_buffer.max_length + 1)
        log_buffer = HistoryBuffer(logs, counts)
        log_history, count_history = log_buffer.data

        assert len(log_history) == log_buffer.max_length
        assert len(count_history) == log_buffer.max_length
        assert logs[1] == log_history[0]
        assert counts[1] == count_history[0]

        # The different lengths of `log_history` and `count_history` will
        # raise error
        with pytest.raises(AssertionError):
            HistoryBuffer([1, 2], [1])

    @pytest.mark.parametrize('array_method', array_method)
    def test_update(self, array_method):
        # test `update` method
        log_buffer = HistoryBuffer()
        log_history = array_method([1, 2, 3, 4, 5])
        count_history = array_method([5, 5, 5, 5, 5])
        for i in range(len(log_history)):
            log_buffer.update(float(log_history[i]), float(count_history[i]))

        recorded_history, recorded_count = log_buffer.data
        for a, b in zip(log_history, recorded_history):
            assert float(a) == float(b)
        for a, b in zip(count_history, recorded_count):
            assert float(a) == float(b)

        # test the length of `array` exceed `max_length`
        max_array = array_method([[-1] + [1] * (log_buffer.max_length - 1)])
        max_count = array_method([[-1] + [1] * (log_buffer.max_length - 1)])
        log_buffer = HistoryBuffer(max_array, max_count)
        log_buffer.update(1)
        log_history, count_history = log_buffer.data
        assert log_history[0] == 1
        assert count_history[0] == 1
        assert len(log_history) == log_buffer.max_length
        assert len(count_history) == log_buffer.max_length
        # Update an iterable object will raise a type error, `log_val` and
        # `count` should be single value
        with pytest.raises(TypeError):
            log_buffer.update(array_method([1, 2]))

    @pytest.mark.parametrize('statistics_method, log_buffer_type',
                             [(np.min, 'min'), (np.max, 'max')])
    def test_max_min(self, statistics_method, log_buffer_type):
        log_history = np.random.randint(1, 5, 20)
        count_history = np.ones(20)
        log_buffer = HistoryBuffer(log_history, count_history)
        assert statistics_method(log_history[-10:]) == \
               getattr(log_buffer, log_buffer_type)(10)
        assert statistics_method(log_history) == \
               getattr(log_buffer, log_buffer_type)()

    def test_mean(self):
        log_history = np.random.randint(1, 5, 20)
        count_history = np.ones(20)
        log_buffer = HistoryBuffer(log_history, count_history)
        assert np.sum(log_history[-10:]) / \
               np.sum(count_history[-10:]) == \
               log_buffer.mean(10)
        assert np.sum(log_history) / \
               np.sum(count_history) == \
               log_buffer.mean()

    def test_current(self):
        log_history = np.random.randint(1, 5, 20)
        count_history = np.ones(20)
        log_buffer = HistoryBuffer(log_history, count_history)
        assert log_history[-1] == log_buffer.current()
        # test get empty array
        log_buffer = HistoryBuffer()
        with pytest.raises(ValueError):
            log_buffer.current()

    def test_statistics(self):
        log_history = np.array([1, 2, 3, 4, 5])
        count_history = np.array([1, 1, 1, 1, 1])
        log_buffer = HistoryBuffer(log_history, count_history)
        assert log_buffer.statistics('mean') == 3
        assert log_buffer.statistics('min') == 1
        assert log_buffer.statistics('max') == 5
        assert log_buffer.statistics('current') == 5
        # Access unknown method will raise an error.
        with pytest.raises(KeyError):
            log_buffer.statistics('unknown')

    def test_register_statistics(self):
        log_buffer = HistoryBuffer()
        assert log_buffer.statistics('custom_statistics') == -1
