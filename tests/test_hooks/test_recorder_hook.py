# Copyright (c) OpenMMLab. All rights reserved.
import sys
from unittest import TestCase
from unittest.mock import Mock

from mmengine import MessageHub
from mmengine.hooks.recorder_hook import FuncRewriter


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


class TestFuncRewriter(TestCase):

    def setUp(self) -> None:
        sys.path.append(__file__)
        return super().setUp()

    def tearDown(self) -> None:
        sys.path.append(__file__)
        MessageHub._instance_dict.clear()
        return super().tearDown()

    def test_init(self):
        func_rewriter = FuncRewriter(
            function='test_recorder_hook.func1', target_variable=('a', 'b'))
        func_rewriter.patch(Mock())
        func1(2, 3)
        print(MessageHub.get_current_instance().runtime_info)
        func_rewriter.unpatch(Mock())
        func_rewriter.clear()

        func_rewriter = FuncRewriter(
            function='test_recorder_hook.ToyClass.func2',
            target_variable=('d', ),
        )
        runner = Mock()
        func_rewriter.patch(runner)
        ToyClass().func2(2)
        print(MessageHub.get_current_instance().runtime_info)
        func_rewriter.unpatch(Mock())
        func_rewriter.clear()

        func_rewriter = FuncRewriter(
            function='test_recorder_hook.ToyClass.func2',
            target_variable=('d', ),
            target_instance='cls1')

        runner = ToyRunner()
        func_rewriter.patch(runner)
        runner.run()
        func_rewriter.unpatch(Mock())
        func_rewriter.clear()
        print(MessageHub.get_current_instance().runtime_info)
