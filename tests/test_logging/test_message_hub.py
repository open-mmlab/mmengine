import pytest
from mmengine import MessageHub
import numpy as np


class TestMessageHub:

    def test_init(self):
        message_hub = MessageHub('name')
        assert message_hub.instance_name == 'name'
        assert len(message_hub.log_buffers) == 0
        assert len(message_hub.log_buffers) == 0

    def test_update_log(self):
        message_hub = MessageHub.create_instance()
        # test create target `LogBuffer` by name
        message_hub.update_log('name', 1)
        log_buffer = message_hub.log_buffers['name']
        assert (log_buffer._log_history == np.array([1])).all()
        # test update target `LogBuffer` by name
        message_hub.update_log('name', 1)
        assert (log_buffer._log_history == np.array([1, 1])).all()
        # unmatched string will raise a key error

    def test_update_info(self):
        message_hub = MessageHub.create_instance()
        # test runtime value can be overwritten.
        message_hub.update_info('key', 2)
        assert message_hub.runtime_info['key'] == 2
        message_hub.update_info('key', 1)
        assert message_hub.runtime_info['key'] == 1

    def test_get_log_buffers(self):
        message_hub = MessageHub.create_instance()
        # Get undefined key will raise error
        with pytest.raises(KeyError):
            message_hub.get_log('unknown')
        # test get log_buffer as wished
        log_history = np.array([1, 2, 3, 4, 5])
        count = np.array([1, 1, 1, 1, 1])
        for i in range(len(log_history)):
            message_hub.update_log('test_value', float(log_history[i]),
                                   int(count[i]))
        recorded_history, recorded_count = \
            message_hub.get_log('test_value').data
        assert (log_history == recorded_history).all()
        assert (recorded_count == count).all()

    def test_get_runtime(self):
        message_hub = MessageHub.create_instance()
        with pytest.raises(KeyError):
            message_hub.get_info('unknown')
        recorded_dict = dict(a=1, b=2)
        message_hub.update_info('test_value', recorded_dict)
        assert message_hub.get_info('test_value') == recorded_dict




