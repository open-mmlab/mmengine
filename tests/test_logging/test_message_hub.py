import pytest
from mmengine import MessageHub


class TestMessageHub:

    def test_init(self):
        # Message hub cannot build from constructor
        with pytest.raises(NotImplementedError):
            MessageHub()

    def test_get_message_hub(self):
        # test `message_hub` can create by name.
        message_hub = MessageHub.get_message_hub('name1')
        assert message_hub.name == 'name1'
        # test default get root `message_hub`.
        message_hub = MessageHub.get_message_hub()
        assert message_hub.name == 'root'
        # test default get latest `message_hub`.
        message_hub = MessageHub.get_message_hub(latest=True)
        assert message_hub.name == 'name1'
        # test default get latest `message_hub` repeatedly.
        message_hub = MessageHub.get_message_hub('name2')
        assert message_hub.name == 'name2'

    def test_update_log(self):
        message_hub = MessageHub.get_message_hub()
        message_hub.update()


