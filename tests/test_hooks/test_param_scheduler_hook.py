# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import Mock

import pytest

from mmengine.hooks import ParamSchedulerHook
from mmengine.optim import _ParamScheduler
from mmengine.runner import BaseLoop


class TestParamSchedulerHook:
    error_msg = ('runner.param_schedulers should be list of ParamScheduler or '
                 'a dict containing list of ParamScheduler')

    def test_after_train_iter(self):
        # runner.param_schedulers should be a list or dict
        with pytest.raises(TypeError, match=self.error_msg):
            hook = ParamSchedulerHook()
            runner = Mock()
            scheduler = Mock()
            scheduler.step = Mock()
            scheduler.by_epoch = False
            runner.param_schedulers = scheduler
            hook.after_train_iter(runner, 0)
            scheduler.step.assert_called()

        # runner.param_schedulers is a list of schedulers
        hook = ParamSchedulerHook()
        runner = Mock()
        scheduler = Mock()
        scheduler.step = Mock()
        scheduler.by_epoch = False
        runner.param_schedulers = [scheduler]
        hook.after_train_iter(runner, 0)
        scheduler.step.assert_called()

        # runner.param_schedulers is a dict containing list of schedulers
        scheduler1 = Mock()
        scheduler1.step = Mock()
        scheduler1.by_epoch = False
        scheduler2 = Mock()
        scheduler2.step = Mock()
        scheduler2.by_epoch = False
        runner.param_schedulers = dict(key1=[scheduler1], key2=[scheduler2])
        hook.after_train_epoch(runner)
        hook.after_train_iter(runner, 0)
        scheduler1.step.assert_called()
        scheduler2.step.assert_called()

    def test_after_train_epoch(self):
        # runner.param_schedulers should be a list or dict
        with pytest.raises(TypeError, match=self.error_msg):
            hook = ParamSchedulerHook()
            runner = Mock()
            scheduler = Mock()
            scheduler.step = Mock()
            scheduler.by_epoch = True
            runner.param_schedulers = scheduler
            hook.after_train_epoch(runner)
            scheduler.step.assert_called()

        # runner.param_schedulers is a list of schedulers
        hook = ParamSchedulerHook()
        runner = Mock()
        scheduler = Mock()
        scheduler.step = Mock()
        scheduler.by_epoch = True
        runner.param_schedulers = [scheduler]
        hook.after_train_epoch(runner)
        scheduler.step.assert_called()

        # runner.param_schedulers is a dict containing list of schedulers
        scheduler1 = Mock()
        scheduler1.step = Mock()
        scheduler1.by_epoch = True
        scheduler2 = Mock()
        scheduler2.step = Mock()
        scheduler2.by_epoch = True
        runner.param_schedulers = dict(key1=[scheduler1], key2=[scheduler2])
        hook.after_train_iter(runner, 0)
        hook.after_train_epoch(runner)
        scheduler1.step.assert_called()
        scheduler2.step.assert_called()

    def test_after_val_epoch(self):
        metrics = dict(loss=1.0)

        # mock super BaseLoop class
        class MockBaseLoop(BaseLoop):

            def __init__(self):
                pass

            def run(self):
                pass

        # mock super _ParamScheduler class
        class MockParamScheduler(_ParamScheduler):

            def __init__(self):
                pass

            def _get_value(self):
                pass

        # runner.param_schedulers is None
        hook = ParamSchedulerHook()
        runner = Mock()
        runner.param_schedulers = None
        assert hook.after_val_epoch(runner, metrics) is None

        # metrics is None
        scheduler = Mock()
        runner.param_schedulers = [scheduler]
        assert hook.after_val_epoch(runner, None) is None

        # runner._train_loop is not built
        hook = ParamSchedulerHook()  # release self._should
        runner._train_loop = None
        assert hook.after_val_epoch(runner, metrics) is None

        # runner.param_schedulers is not built
        hook = ParamSchedulerHook()  # release self._should
        runner._train_loop = MockBaseLoop()
        runner.param_schedulers = dict(key1=[None], key2=[None])
        assert hook.after_val_epoch(runner) is None

        # runner.param_schedulers should be a list or dict
        with pytest.raises(TypeError, match=self.error_msg):
            hook = ParamSchedulerHook()
            runner = Mock()
            runner._train_loop = MockBaseLoop()
            scheduler = Mock()
            scheduler.step = Mock()
            scheduler.by_epoch = True
            scheduler.need_val_args = True
            runner.param_schedulers = scheduler
            hook.after_val_epoch(runner, metrics)
            scheduler.step.assert_called_with(metrics)

        # runner.param_schedulers is a list of schedulers
        hook = ParamSchedulerHook()
        runner = Mock()
        runner._train_loop = MockBaseLoop()
        scheduler = Mock()
        scheduler.step = Mock()
        scheduler.by_epoch = True
        scheduler.need_val_args = True
        runner.param_schedulers = [scheduler]
        hook.after_val_epoch(runner, metrics)
        scheduler.step.assert_called_with(metrics)

        # runner.param_schedulers is a dict containing list of schedulers
        scheduler1 = Mock()
        scheduler1.step = Mock()
        scheduler1.by_epoch = True
        scheduler1.need_val_args = True
        scheduler2 = Mock()
        scheduler2.step = Mock()
        scheduler2.by_epoch = True
        scheduler2.need_val_args = True
        runner.param_schedulers = dict(key1=[scheduler1], key2=[scheduler2])
        hook.after_val_epoch(runner, metrics)
        scheduler1.step.assert_called_with(metrics)
        scheduler2.step.assert_called_with(metrics)
