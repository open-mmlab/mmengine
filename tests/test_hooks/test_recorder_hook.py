# Copyright (c) OpenMMLab. All rights reserved.
from cgitb import Hook
import copy
import os
import sys
from unittest import TestCase
from unittest.mock import Mock

import mmengine
import mmengine.testing as testing
from mmengine import MessageHub
from mmengine.hooks import RecorderHook
from mmengine.hooks.recorder_hook import FuncRewriterRecorder
from mmengine.registry import HOOKS



def func1(a, b):
    c = a + b
    return c * 10


class ToyClass:

    def func2(self, d):
        return d * 10


class ToyRunner:

    def __init__(self) -> None:
        self.cls1 = ToyClass()
        self.cls2 = ToyClass()

    def run(self):
        self.cls1.func2(1)
        self.cls2.func2(2)


def toy_collate_fn(data_batches):
    return data_batches


class TestFuncRewriter(TestCase):

    def setUp(self) -> None:
        sys.path.append(__file__)
        return super().setUp()

    def tearDown(self) -> None:
        sys.path.append(__file__)
        MessageHub._instance_dict.clear()
        return super().tearDown()

    def test_init(self):
        func_rewriter = FuncRewriterRecorder(
            function='test_recorder_hook.func1', target_variable=('a', 'b'))
        func_rewriter.initialize(Mock())
        func1(2, 3)
        print(MessageHub.get_current_instance().runtime_info)
        func_rewriter.deinitialize(Mock())
        func_rewriter.clear()

        func_rewriter = FuncRewriterRecorder(
            function='test_recorder_hook.ToyClass.func2',
            target_variable=('d', ),
        )
        runner = Mock()
        func_rewriter.initialize(runner)
        ToyClass().func2(2)
        print(MessageHub.get_current_instance().runtime_info)
        func_rewriter.deinitialize(Mock())
        func_rewriter.clear()

        func_rewriter = FuncRewriterRecorder(
            function='test_recorder_hook.ToyClass.func2',
            target_variable=('d', ),
            target_instance='cls1')

        runner = ToyRunner()
        func_rewriter.initialize(runner)
        runner.run()
        func_rewriter.deinitialize(Mock())
        func_rewriter.clear()
        print(MessageHub.get_current_instance().runtime_info)


class  TestRecorderHook(testing.RunnerTestCase):
    def setUp(self):
        super().setUp()
        HOOKS.register_module()
        sys.path.append(os.path.dirname(__file__))

    def test_with_runner(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.custom_hooks = [
            dict(
                type='RecorderHook',
                recorders=[
                    dict(
                        type='FuncRewriterRecorder',
                        function=
                        'mmengine.testing.runner_test_case.ToyModel.forward',
                        target_variable=('inputs', 'data_samples'),
                        recorded_name='model'),
                    dict(
                        type='FuncRewriterRecorder',
                        function='torch.nn.Linear.forward',
                        target_instance='linear1',
                        recorded_name='linear1',
                    ),
                    dict(
                        type='AttributeGetterRecorder',
                        target_attributes='linear1.weight',
                        recorded_name='linear1_weight'
                    )
                ],
            )
        ]
        runner = self.build_runner(cfg)
        # TODO add more tests
        runner.train()
