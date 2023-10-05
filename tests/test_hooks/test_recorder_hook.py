# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os.path as osp
import tempfile
from unittest.mock import Mock

import torch
import torch.nn as nn
from parameterized import parameterized

from mmengine.hooks import RecorderHook
from mmengine.logging import MMLogger
from mmengine.model import BaseModel
from mmengine.testing import RunnerTestCase


class ToyModel(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, data_sample, mode='tensor'):
        labels = torch.stack(data_sample)
        inputs = torch.stack(inputs)
        outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


def get_mock_runner():
    runner = Mock()
    runner.train_loop = Mock()
    runner.train_loop.stop_training = False
    return runner


class TestRecorderHook(RunnerTestCase):

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # `FileHandler` should be closed in Windows, otherwise we cannot
        # delete the temporary directory
        logging.shutdown()
        MMLogger._instance_dict.clear()
        self.temp_dir.cleanup()

    def test_init(self):
        # Test recorders
        with self.assertRaisesRegex(ValueError, 'recorders not initialized'):
            RecorderHook(recorders=None, save_dir=self.temp_dir)
        with self.assertRaisesRegex(ValueError, 'recorders not initialized'):
            RecorderHook(recorders=[], save_dir=self.temp_dir)

        hook = RecorderHook(
            recorders=[dict(type='FunctionRecorder', target='x')])
        self.assertEqual(len(hook.recorders), 1)
        self.assertEqual(hook.recorders[0].target, 'x')

        self.assertEqual(hook.recorders[0].model, 'runner_model')
        self.assertEqual(hook.recorders[0].method, 'forward')

        hook = RecorderHook(recorders=[
            dict(type='AttributeRecorder', target='self.linear1.weight')
        ])
        self.assertEqual(len(hook.recorders), 1)
        self.assertEqual(hook.recorders[0].model, 'runner_model')
        self.assertEqual(hook.recorders[0].method, 'forward')
        self.assertEqual(hook.recorders[0].target, 'linear1.weight')

        hook = RecorderHook(recorders=[
            dict(type='FunctionRecorder', target='x'),
            dict(type='AttributeRecorder', target='self.linear1.weight')
        ])
        self.assertEqual(len(hook.recorders), 2)

        hook = RecorderHook(recorders=[
            dict(
                type='AttributeRecorder',
                model='resnet',
                method='_forward_impl',
                target='x')
        ])
        self.assertEqual(len(hook.recorders), 1)
        self.assertEqual(hook.recorders[0].model, 'resnet')
        self.assertEqual(hook.recorders[0].method, '_forward_impl')
        self.assertEqual(hook.recorders[0].target, 'x')

    def test_before_run(self):
        # test method modification
        runner = Mock()
        base_model = ToyModel()
        origin_forward = base_model.forward
        runner.model = base_model
        runner.work_dir = self.temp_dir.name

        hook = RecorderHook(
            recorders=[dict(type='FunctionRecorder', target='x')])
        hook.before_run(runner)
        self.assertEqual(hook.save_dir, self.temp_dir.name)
        self.assertEqual(hook.base_model, base_model)
        self.assertNotEqual(origin_forward, hook.base_model.forward)

    def test_after_train(self):
        runner = Mock()
        base_model = ToyModel()
        origin_forward = base_model.forward
        runner.model = base_model

        hook = RecorderHook(
            recorders=[dict(type='FunctionRecorder', target='x')])
        hook.before_run(runner)
        self.assertEqual(hook.base_model, base_model)
        self.assertNotEqual(origin_forward, hook.base_model.forward)

        hook.after_train(runner)
        self.assertEqual(origin_forward, hook.base_model.forward)

    @parameterized.expand([['iter'], ['epoch']])
    def test_with_runner(self, training_type):
        common_cfg = getattr(self, f'{training_type}_based_cfg')
        setattr(common_cfg.train_cfg, f'max_{training_type}s', 11)
        recorder_cfg = dict(
            type='RecorderHook', by_epoch=training_type == 'epoch', interval=1)
        common_cfg.default_hooks = dict(recorder=recorder_cfg)

        # Test interval in epoch based training
        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.recorder.recorders = [
            dict(type='FunctionRecorder', target='outputs', index=[0, 1])
        ]
        cfg.default_hooks.recorder.interval = 2
        runner = self.build_runner(cfg)
        runner.train()

        for i in range(1, 11):
            self.assertEqual(
                osp.isfile(
                    osp.join(cfg.work_dir, f'record_{training_type}_{i}.pth')),
                i % 2 == 0)

        record = torch.load(
            osp.join(cfg.work_dir, f'record_{training_type}_10.pth'))
        self.assertEqual(len(record), 2)
        for varname, var in record.items():
            self.assertTrue(varname.startswith('runner_model:forward:outputs'))
            # tensor_list should be a list of tensor
            if training_type == 'epoch':
                self.assertTrue(
                    all(isinstance(item, torch.Tensor) for item in var))
            else:
                self.assertTrue(isinstance(var, torch.Tensor))

        self.clear_work_dir()

        cfg = copy.deepcopy(common_cfg)
        cfg.default_hooks.recorder.recorders = [
            dict(type='AttributeRecorder', target='linear1.weight'),
            dict(type='AttributeRecorder', target='linear2.bias')
        ]

        runner = self.build_runner(cfg)
        runner.train()

        for i in range(1, 11):
            self.assertEqual(
                osp.isfile(
                    osp.join(cfg.work_dir, f'record_{training_type}_{i}.pth')),
                True)

        record = torch.load(
            osp.join(cfg.work_dir, f'record_{training_type}_10.pth'))
        self.assertEqual(len(record), 2)
        for varname, var in record.items():
            self.assertTrue(
                varname.startswith('runner_model:forward:linear1.weight')
                or varname.startswith('runner_model:forward:linear2.bias'))
            if training_type == 'epoch':
                self.assertTrue(
                    all(isinstance(item, torch.Tensor) for item in var))
            else:
                self.assertTrue(isinstance(var, torch.Tensor))
