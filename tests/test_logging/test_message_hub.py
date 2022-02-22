import pytest
from mmengine import MessageHub, MeanLogBuffer, MaxLogBuffer, MinLogBuffer,\
    CurrentLogBuffer
import numpy as np


class TestMessageHub:

    def test_init(self):
        message_hub = MessageHub('name')
        assert message_hub.name == 'name'
        assert len(message_hub.log_buffers) == 0
        assert len(message_hub.log_buffers) == 0

    def test_get_message_hub(self):
        # test `message_hub` can create by name.
        message_hub = MessageHub.get_message_hub('name1')
        assert message_hub.name == 'name1'
        # test default get root `message_hub`.
        message_hub = MessageHub.get_message_hub()
        assert message_hub.name == 'root'
        # test default get latest `message_hub`.
        message_hub = MessageHub.get_message_hub(current=True)
        assert message_hub.name == 'name1'
        message_hub.mark = -1
        # test get latest `message_hub` repeatedly.
        message_hub = MessageHub.get_message_hub('name2')
        assert message_hub.name == 'name2'
        message_hub = MessageHub.get_message_hub(current=True)
        assert message_hub.name == 'name2'
        # test get root repeatedly.
        message_hub = MessageHub.get_message_hub()
        assert message_hub.name == 'root'
        # test get name1 repeatedly
        message_hub = MessageHub.get_message_hub('name1')
        assert message_hub.mark == -1

    def test_update_log(self):
        message_hub = MessageHub.get_message_hub()
        # empty buffer
        # test get target `LogBuffer` by name
        message_hub.update_log('mean_value', 1, log_type='mean')
        assert isinstance(message_hub._log_buffers['mean_value'],
                          MeanLogBuffer)
        message_hub.update_log('min_value', 1, log_type='min')
        assert isinstance(message_hub._log_buffers['min_value'],
                          MinLogBuffer)
        message_hub.update_log('max_value', 1, log_type='max')
        assert isinstance(message_hub._log_buffers['max_value'],
                          MaxLogBuffer)
        message_hub.update_log('current_value', 1, log_type='current')
        assert isinstance(message_hub._log_buffers['current_value'],
                          CurrentLogBuffer)
        message_hub.update_log('current_value', 1)
        assert isinstance(message_hub._log_buffers['current_value'],
                          CurrentLogBuffer)
        # unmatched string will raise a key error
        with pytest.raises(KeyError):
            message_hub.update_log('mean_value', 1, log_type='unknown')

    def test_update_runtime(self):
        message_hub = MessageHub.get_message_hub()
        # test runtime value can be overwritten.
        message_hub.update_runtime('key', 2)
        assert message_hub.runtime['key'] == 2
        message_hub.update_runtime('key', 1)
        assert message_hub.runtime['key'] == 1

    def test_get_log_buffers(self):
        message_hub = MessageHub.get_message_hub()
        # Get undefined key will raise error
        with pytest.raises(KeyError):
            message_hub.get_log_buffer('unknown')
        # test get log_buffer as wished
        log_history = np.array([1, 2, 3, 4, 5])
        count = np.array([1, 1, 1, 1, 1])
        for i in range(len(log_history)):
            message_hub.update_log('test_value', log_history[i], count[i])
        recorded_history, recorded_count = \
            message_hub.get_log_buffer('test_value').data
        assert (log_history == recorded_history).all()
        assert (recorded_count == count).all()

    def test_get_runtime(self):
        message_hub = MessageHub.get_message_hub()
        with pytest.raises(KeyError):
            message_hub.get_runtime('unknown')
        recorded_dict = dict(a=1, b=2)
        message_hub.update_runtime('test_value', recorded_dict)
        assert message_hub.get_runtime('test_value') == recorded_dict

    def test_get_all_data(self):
        message_hub = MessageHub.get_message_hub()
        message_hub.update_log('name1', 1)
        message_hub.update_log('name2', 1)
        message_hub.update_runtime('name3', 1)
        message_hub.update_runtime('name4', 2)
        # test get `log_buffers` and  `runtime` as wished
        log_buffers = message_hub.log_buffers
        runtime = message_hub.runtime
        assert 'name1' in log_buffers
        assert 'name2' in log_buffers
        assert 'name3' in runtime
        assert 'name4' in runtime




