# Copyright (c) OpenMMLab. All rights reserved.
import time

from .hook import HOOKS, Hook


@HOOKS.register_module()
class IterTimerHook(Hook):
    """A hook that logs the time spent during iteration.

    Eg. ``data_time`` for loading data and ``time`` for a model train step.
    """

    def before_epoch(self, runner: object) -> None:
        """Record time flag before start a epoch."""
        self.t = time.time()

    def before_iter(self, runner: object, data_batch=None) -> None:
        """Logging time for loading data and update the time flag."""
        # TODO: update for new logging system
        runner.log_buffer.update({  # type: ignore
            'data_time': time.time() - self.t
        })

    def after_iter(self,
                   runner: object,
                   data_batch=None,
                   outputs=None) -> None:
        """Logging time for a iteration and update the time flag."""
        # TODO: update for new logging system
        runner.log_buffer.update({  # type: ignore
            'time': time.time() - self.t
        })
        self.t = time.time()
