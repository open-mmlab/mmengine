# Copyright (c) OpenMMLab. All rights reserved.
import copy
from mmengine.hooks import FreezeHook
from mmengine.testing import RunnerTestCase


class TestFreezeHook(RunnerTestCase):

    def setUp(self):
        super().setUp()

    def test_init(self):
        # Test build freeze hook.
        FreezeHook(freeze_epoch=1, freeze_layers=("backbone",))

        with self.assertRaisesRegex(TypeError, 'freeze_epoch must be'):
            FreezeHook(freeze_epoch='100', freeze_layers=("backbone",))

        with self.assertRaisesRegex(ValueError, 'freeze_epoch'):
            FreezeHook(freeze_epoch=0, freeze_layers=("backbone",))

        # freeze_layers should be None or string or tuple of string or list of string.
        with self.assertRaisesRegex(TypeError, 'freeze_layers must be'):
            FreezeHook(freeze_epoch=1, freeze_layers=False)

        with self.assertRaisesRegex(TypeError, 'freeze_layers must be'):
            FreezeHook(freeze_epoch=1, freeze_layers=1)
            
        with self.assertRaisesRegex(TypeError, 'unfreeze_epoch must be'):
            FreezeHook(freeze_epoch=1, freeze_layers="backbone", unfreeze_epoch='100')

        with self.assertRaisesRegex(ValueError, 'unfreeze_epoch'):
            FreezeHook(freeze_epoch=1,freeze_layers="backbone", unfreeze_layers=("backbone",))

        with self.assertRaisesRegex(ValueError, 'unfreeze_epoch'):
            FreezeHook(freeze_epoch=1, freeze_layers="backbone", unfreeze_epoch=2)

        with self.assertRaisesRegex(ValueError, 'unfreeze_epoch'):
            FreezeHook(freeze_epoch=1,freeze_layers="backbone", unfreeze_epoch=1,unfreeze_layers=("backbone",))

        # unfreeze_layers should be None or string or tuple of string or list of string.
        with self.assertRaisesRegex(TypeError, 'unfreeze_layers must be'):
            FreezeHook(freeze_epoch=1, freeze_layers="backbone", unfreeze_layers=False)

        with self.assertRaisesRegex(TypeError, 'unfreeze_layers must be'):
            FreezeHook(freeze_epoch=1, freeze_layers="backbone", unfreeze_layers=1)

        with self.assertRaisesRegex(ValueError, 'unfreeze_layers'):
            FreezeHook(freeze_epoch=1,freeze_layers="backbone", unfreeze_epoch=2)

        with self.assertRaisesRegex(TypeError, 'log_grad must be'):
            FreezeHook(freeze_epoch=1,freeze_layers="backbone", log_grad=1)

    def test_before_train_epoch(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)

        freeze_hook = FreezeHook(freeze_epoch=2, freeze_layers=("linear1","linear2"))
        freeze_hook.before_train_epoch(runner)
        self.assertTrue(runner.model.linear1.weight.requires_grad)
        self.assertTrue(runner.model.linear2.weight.requires_grad)

        runner.train_loop._epoch = 3
        freeze_hook = FreezeHook(freeze_epoch=4, freeze_layers=("linear1","linear2"))
        freeze_hook.before_train_epoch(runner)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertFalse(runner.model.linear2.weight.requires_grad)

        runner.train_loop._epoch = 5
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertFalse(runner.model.linear2.weight.requires_grad)

        runner.train_loop._epoch = 7
        freeze_hook = FreezeHook(freeze_epoch=4, freeze_layers=("linear1","linear2"), unfreeze_epoch=8, unfreeze_layers=("linear2",))
        freeze_hook.before_train_epoch(runner)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertTrue(runner.model.linear2.weight.requires_grad)

        runner.train_loop._epoch = 9
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertTrue(runner.model.linear2.weight.requires_grad)
