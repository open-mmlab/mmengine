# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest
from functools import partial

from mmengine.utils import context_timer, decorate_timer


class TestProfiling(unittest.TestCase):

    def test_decorate_timer(self):

        @decorate_timer
        def demo_fun():
            time.sleep(0.1)

        demo_fun()

        @decorate_timer
        def demo_fun():
            time.sleep(0.1)

        for _ in range(10):
            demo_fun()

        @partial(decorate_timer, log_interval=2, with_sync=False)
        def demo_fun():
            time.sleep(0.1)

        demo_fun()

        # warmup_interval must be greater than 0
        with self.assertRaises(AssertionError):

            @partial(decorate_timer, warmup_interval=0)
            def demo_fun():
                time.sleep(0.1)

    def test_context_timer(self):
        with context_timer('func_1'):
            time.sleep(0.1)

        for _ in range(10):
            with context_timer('func_2', log_interval=2, with_sync=False):
                time.sleep(0.1)

        # warmup_interval must be greater than 0
        with self.assertRaises(AssertionError):
            with context_timer('func_1', warmup_interval=0):
                time.sleep(0.1)
