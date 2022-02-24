# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import Registry

HOOKS = Registry('hook')


class Hook:

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass

    def after_epoch(self, runner):
        pass

    def before_iter(self, runner, data_batch=None):
        pass

    def after_iter(self, runner, data_batch=None, outputs=None):
        pass

    def before_save_checkpoint(self, runner, checkpoint):
        pass

    def after_load_checkpoint(self, runner, checkpoint):
        pass

    def before_train_epoch(self, runner):
        self.before_epoch(runner)

    def before_val_epoch(self, runner):
        self.before_epoch(runner)

    def before_test_epoch(self, runner):
        self.before_epoch(runner)

    def after_train_epoch(self, runner):
        self.after_epoch(runner)

    def after_val_epoch(self, runner):
        self.after_epoch(runner)

    def after_test_epoch(self, runner):
        self.after_epoch(runner)

    def before_train_iter(self, runner, data_batch=None):
        self.before_iter(runner, data_batch=None)

    def before_val_iter(self, runner, data_batch=None):
        self.before_iter(runner, data_batch=None)

    def before_test_iter(self, runner, data_batch=None):
        self.before_iter(runner, data_batch=None)

    def after_train_iter(self, runner, data_batch=None, outputs=None):
        self.after_iter(runner, data_batch=None, outputs=None)

    def after_val_iter(self, runner, data_batch=None, outputs=None):
        self.after_iter(runner, data_batch=None, outputs=None)

    def after_test_iter(self, runner, data_batch=None, outputs=None):
        self.after_iter(runner, data_batch=None, outputs=None)

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter + 1 == len(runner.data_loader)

    def is_last_epoch(self, runner):
        return runner.epoch + 1 == runner._max_epochs

    def is_last_iter(self, runner):
        return runner.iter + 1 == runner._max_iters
