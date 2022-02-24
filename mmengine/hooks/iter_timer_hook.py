# Copyright (c) OpenMMLab. All rights reserved.
import time

from .hook import HOOKS, Hook


@HOOKS.register_module()
class IterTimerHook(Hook):
    """A hook that logs the time spent during iteration.

    Eg. ``data_time`` for loading data and ``time`` for a model train step.
    """

    def before_epoch(self, runner):
        self.t = time.time()

    def before_iter(self, runner):
        # TODO: update for new logging system
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        # TODO: update for new logging system
        runner.log_buffer.update({'time': time.time() - self.t})
        self.t = time.time()
