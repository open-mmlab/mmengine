# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from collections import OrderedDict

import numpy as np
import pytest
import torch

from mmengine import MessageHub


class TestMessageHub:

    def test_init(self):
        message_hub = MessageHub('name')
        assert message_hub.instance_name == 'name'
        assert len(message_hub.log_scalars) == 0
        assert len(message_hub.log_scalars) == 0
        # The type of log_scalars's value must be `HistoryBuffer`.
        with pytest.raises(AssertionError):
            MessageHub('hello', log_scalars=OrderedDict(a=1))
        # `Resumed_keys`
        with pytest.raises(AssertionError):
            MessageHub(
                'hello',
                runtime_info=OrderedDict(iter=1),
                resumed_keys=OrderedDict(iters=False))

    def test_update_scalar(self):
        message_hub = MessageHub.get_instance('mmengine')
        # test create target `HistoryBuffer` by name
        message_hub.update_scalar('name', 1)
        log_buffer = message_hub.log_scalars['name']
        assert (log_buffer._log_history == np.array([1])).all()
        # test update target `HistoryBuffer` by name
        message_hub.update_scalar('name', 1)
        assert (log_buffer._log_history == np.array([1, 1])).all()
        # unmatched string will raise a key error

    def test_update_info(self):
        message_hub = MessageHub.get_instance('mmengine')
        # test runtime value can be overwritten.
        message_hub.update_info('key', 2)
        assert message_hub.runtime_info['key'] == 2
        message_hub.update_info('key', 1)
        assert message_hub.runtime_info['key'] == 1

    def test_get_scalar(self):
        message_hub = MessageHub.get_instance('mmengine')
        # Get undefined key will raise error
        with pytest.raises(KeyError):
            message_hub.get_scalar('unknown')
        # test get log_buffer as wished
        log_history = np.array([1, 2, 3, 4, 5])
        count = np.array([1, 1, 1, 1, 1])
        for i in range(len(log_history)):
            message_hub.update_scalar('test_value', float(log_history[i]),
                                      int(count[i]))
        recorded_history, recorded_count = \
            message_hub.get_scalar('test_value').data
        assert (log_history == recorded_history).all()
        assert (recorded_count == count).all()

    def test_get_runtime(self):
        message_hub = MessageHub.get_instance('mmengine')
        with pytest.raises(KeyError):
            message_hub.get_info('unknown')
        recorded_dict = dict(a=1, b=2)
        message_hub.update_info('test_value', recorded_dict)
        assert message_hub.get_info('test_value') == recorded_dict

    def test_get_scalars(self):
        message_hub = MessageHub.get_instance('mmengine')
        log_dict = dict(
            loss=1,
            loss_cls=torch.tensor(2),
            loss_bbox=np.array(3),
            loss_iou=dict(value=1, count=2))
        message_hub.update_scalars(log_dict)
        loss = message_hub.get_scalar('loss')
        loss_cls = message_hub.get_scalar('loss_cls')
        loss_bbox = message_hub.get_scalar('loss_bbox')
        loss_iou = message_hub.get_scalar('loss_iou')
        assert loss.current() == 1
        assert loss_cls.current() == 2
        assert loss_bbox.current() == 3
        assert loss_iou.mean() == 0.5

        with pytest.raises(AssertionError):
            loss_dict = dict(error_type=[])
            message_hub.update_scalars(loss_dict)

        with pytest.raises(AssertionError):
            loss_dict = dict(error_type=dict(count=1))
            message_hub.update_scalars(loss_dict)

    def test_getstate(self):
        message_hub = MessageHub.get_instance('name')
        # update log_scalars.
        message_hub.update_scalar('loss', 0.1)
        message_hub.update_scalar('lr', 0.1, resumed=False)
        # update runtime information
        message_hub.update_info('iter', 1, resumed=True)
        message_hub.update_info('feat', [1, 2, 3], resumed=False)
        obj = pickle.dumps(message_hub)
        instance = pickle.loads(obj)

        with pytest.raises(KeyError):
            instance.get_info('feat')
        with pytest.raises(KeyError):
            instance.get_info('lr')

        instance.get_info('iter')
        instance.get_scalar('loss')

    def test_get_instance(self):
        # Test get root mmengine message hub.
        MessageHub._instance_dict = OrderedDict()
        root_logger = MessageHub.get_current_instance()
        assert id(MessageHub.get_instance('mmengine')) == id(root_logger)
        # Test original `get_current_instance` function.
        MessageHub.get_instance('mmdet')
        assert MessageHub.get_current_instance().instance_name == 'mmdet'
