# Copyright (c) OpenMMLab. All rights reserved.
import time
import unittest

from mmengine.utils.dl_utils.time_counter import TimeCounter


class TestTimeCounter(unittest.TestCase):

    def test_decorate_timer(self):

        @TimeCounter()
        def demo_fun():
            time.sleep(0.1)

        demo_fun()

        @TimeCounter()
        def demo_fun():
            time.sleep(0.1)

        for _ in range(10):
            demo_fun()

        @TimeCounter(log_interval=2, with_sync=False, tag='demo_fun1')
        def demo_fun():
            time.sleep(0.1)

        demo_fun()

        # warmup_interval must be greater than 0
        with self.assertRaises(AssertionError):

            @TimeCounter(warmup_interval=0)
            def demo_fun():
                time.sleep(0.1)

    def test_context_timer(self):

        # tag must be specified in context mode
        with self.assertRaises(AssertionError):
            with TimeCounter():
                time.sleep(0.1)

        # warmup_interval must be greater than 0
        with self.assertRaises(AssertionError):
            with TimeCounter(warmup_interval=0, tag='func_1'):
                time.sleep(0.1)

        with TimeCounter(tag='func_1'):
            time.sleep(0.1)

        for _ in range(10):
            with TimeCounter(log_interval=2, with_sync=False, tag='func_2'):
                time.sleep(0.1)
