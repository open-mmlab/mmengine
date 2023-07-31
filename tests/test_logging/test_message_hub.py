# Copyright (c) OpenMMLab. All rights reserved.
import pickle
from collections import OrderedDict

import numpy as np
import pytest

from mmengine.logging import HistoryBuffer, MessageHub
from mmengine.utils import is_installed


class NoDeepCopy:

    def __deepcopy__(self, memodict={}):
        raise NotImplementedError


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
        # Update scalar with int.
        message_hub.update_scalar('name', 1)
        log_buffer = message_hub.log_scalars['name']
        assert (log_buffer._log_history == np.array([1])).all()

        # Update scalar with np.ndarray.
        message_hub.update_scalar('name', np.array(1))
        assert (log_buffer._log_history == np.array([1, 1])).all()

        # Update scalar with np.int
        message_hub.update_scalar('name', np.int32(1))
        assert (log_buffer._log_history == np.array([1, 1, 1])).all()

    def test_update_info(self):
        message_hub = MessageHub.get_instance('mmengine')
        # test runtime value can be overwritten.
        message_hub.update_info('key', 2)
        assert message_hub.runtime_info['key'] == 2
        message_hub.update_info('key', 1)
        assert message_hub.runtime_info['key'] == 1

    def test_pop_info(self):
        message_hub = MessageHub.get_instance('mmengine')
        message_hub.update_info('pop_key', 'pop_info')
        assert message_hub.runtime_info['pop_key'] == 'pop_info'
        assert message_hub.pop_info('pop_key') == 'pop_info'

        assert message_hub.pop_info('not_existed_key', 'info') == 'info'

    def test_update_infos(self):
        message_hub = MessageHub.get_instance('mmengine')
        # test runtime value can be overwritten.
        message_hub.update_info_dict({'a': 2, 'b': 3})
        assert message_hub.runtime_info['a'] == 2
        assert message_hub.runtime_info['b'] == 3
        assert message_hub._resumed_keys['a']
        assert message_hub._resumed_keys['b']

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
        assert message_hub.get_info('unknown') is None
        recorded_dict = dict(a=1, b=2)
        message_hub.update_info('test_value', recorded_dict)
        assert message_hub.get_info('test_value') == recorded_dict

    @pytest.mark.skipif(not is_installed('torch'), reason='requires torch')
    def test_get_scalars(self):
        import torch
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

    def test_state_dict(self):
        message_hub = MessageHub.get_instance('test_state_dict')
        # update log_scalars.
        message_hub.update_scalar('loss', 0.1)
        message_hub.update_scalar('lr', 0.1, resumed=False)
        # update runtime information
        message_hub.update_info('iter', 1, resumed=True)
        message_hub.update_info('tensor', [1, 2, 3], resumed=False)
        no_copy = NoDeepCopy()
        message_hub.update_info('no_copy', no_copy, resumed=True)
        state_dict = message_hub.state_dict()

        assert state_dict['log_scalars']['loss'].data == (np.array([0.1]),
                                                          np.array([1]))
        assert 'lr' not in state_dict['log_scalars']
        assert state_dict['runtime_info']['iter'] == 1
        assert 'tensor' not in state_dict['runtime_info']
        assert state_dict['runtime_info']['no_copy'] is no_copy

    def test_load_state_dict(self, capsys):
        message_hub1 = MessageHub.get_instance('test_load_state_dict1')
        # update log_scalars.
        message_hub1.update_scalar('loss', 0.1)
        message_hub1.update_scalar('lr', 0.1, resumed=False)
        # update runtime information
        message_hub1.update_info('iter', 1, resumed=True)
        message_hub1.update_info('tensor', [1, 2, 3], resumed=False)
        state_dict = message_hub1.state_dict()

        # Resume from state_dict
        message_hub2 = MessageHub.get_instance('test_load_state_dict2')
        message_hub2.load_state_dict(state_dict)
        assert message_hub2.get_scalar('loss').data == (np.array([0.1]),
                                                        np.array([1]))
        assert message_hub2.get_info('iter') == 1

        # Test resume from `MessageHub` instance.
        message_hub3 = MessageHub.get_instance('test_load_state_dict3')
        message_hub3.load_state_dict(state_dict)
        assert message_hub3.get_scalar('loss').data == (np.array([0.1]),
                                                        np.array([1]))
        assert message_hub3.get_info('iter') == 1

        # Test resume custom state_dict
        state_dict = OrderedDict()
        state_dict['log_scalars'] = dict(a=1, b=HistoryBuffer())
        state_dict['runtime_info'] = dict(c=1, d=NoDeepCopy(), e=1)
        state_dict['resumed_keys'] = dict(
            a=True, b=True, c=True, e=False, f=True)

        message_hub4 = MessageHub.get_instance('test_load_state_dict4')
        message_hub4.load_state_dict(state_dict)
        assert 'a' not in message_hub4.log_scalars and 'b' in \
               message_hub4.log_scalars
        assert 'c' in message_hub4.runtime_info and \
               state_dict['runtime_info']['d'] is \
               message_hub4.runtime_info['d']
        assert message_hub4._resumed_keys == OrderedDict(
            b=True, c=True, e=False)

    def test_getstate(self):
        message_hub = MessageHub.get_instance('name')
        # update log_scalars.
        message_hub.update_scalar('loss', 0.1)
        message_hub.update_scalar('lr', 0.1, resumed=False)
        # update runtime information
        message_hub.update_info('iter', 1, resumed=True)
        message_hub.update_info('tensor', [1, 2, 3], resumed=False)
        obj = pickle.dumps(message_hub)
        instance = pickle.loads(obj)

        assert instance.get_info('feat') is None
        assert instance.get_info('lr') is None

        instance.get_info('iter')
        instance.get_scalar('loss')

    def test_get_instance(self):
        # Test get root mmengine message hub.
        MessageHub._instance_dict = OrderedDict()
        message_hub = MessageHub.get_current_instance()
        assert id(MessageHub.get_instance('mmengine')) == id(message_hub)
        # Test original `get_current_instance` function.
        MessageHub.get_instance('mmdet')
        assert MessageHub.get_current_instance().instance_name == 'mmdet'
