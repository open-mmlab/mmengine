# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmengine.hooks import FreezeHook
from mmengine.testing import RunnerTestCase


class TestFreezeHook(RunnerTestCase):

    def setUp(self):
        super().setUp()

    def test_init(self):
        # Test FreezeHook TypeError.
        FreezeHook(freeze_layers='backbone.*', freeze_epoch=0)

        with self.assertRaisesRegex(TypeError, '`freeze_layers`'):
            FreezeHook(freeze_layers=1, freeze_epoch=0)

        with self.assertRaisesRegex(TypeError, '`freeze_layers`'):
            FreezeHook(freeze_layers=(1, 2), freeze_epoch=0)

        with self.assertRaisesRegex(TypeError, '`freeze_layers`'):
            FreezeHook(freeze_layers=('backbone.*', ), freeze_epoch=0)

        with self.assertRaisesRegex(TypeError, '`freeze_iter`'):
            FreezeHook(freeze_layers='backbone.*', freeze_iter='0')

        with self.assertRaisesRegex(TypeError, '`freeze_epoch`'):
            FreezeHook(freeze_layers='backbone.*', freeze_epoch='0')

        with self.assertRaisesRegex(TypeError, '`unfreeze_layers`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_epoch=0,
                unfreeze_layers=1,
                unfreeze_epoch=0,
            )

        with self.assertRaisesRegex(TypeError, '`unfreeze_layers`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_epoch=0,
                unfreeze_layers=(1, 2),
                unfreeze_epoch=0,
            )

        with self.assertRaisesRegex(TypeError, '`unfreeze_layers`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_epoch=0,
                unfreeze_layers=('backbone.*', ),
                unfreeze_epoch=0,
            )

        with self.assertRaisesRegex(TypeError, '`unfreeze_iter`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_iter=0,
                unfreeze_layers='backbone.*',
                unfreeze_iter='0',
            )

        with self.assertRaisesRegex(TypeError, '`unfreeze_epoch`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_epoch=0,
                unfreeze_layers='backbone.*',
                unfreeze_epoch='0',
            )

        with self.assertRaisesRegex(TypeError, '`verbose`'):
            FreezeHook(freeze_layers='backbone.*', freeze_epoch=1, verbose=1)

        # Test FreezeHook ValueError.
        with self.assertRaisesRegex(ValueError, '`freeze_iter`'):
            FreezeHook(freeze_layers='backbone.*', freeze_iter=-1)

        with self.assertRaisesRegex(ValueError, '`freeze_epoch`'):
            FreezeHook(freeze_layers='backbone.*', freeze_epoch=-1)

        with self.assertRaisesRegex(ValueError,
                                    '`freeze_iter` and `freeze_epoch`'):
            FreezeHook(freeze_layers='backbone.*', )

        with self.assertRaisesRegex(ValueError,
                                    '`freeze_iter` and `freeze_epoch`'):
            FreezeHook(
                freeze_layers='backbone.*', freeze_iter=0, freeze_epoch=0)

        with self.assertRaisesRegex(ValueError,
                                    '`unfreeze_iter` and `unfreeze_epoch`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_epoch=0,
                unfreeze_layers='backbone.*')

        with self.assertRaisesRegex(ValueError,
                                    '`unfreeze_iter` and `unfreeze_epoch`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_epoch=0,
                unfreeze_layers='backbone.*',
                unfreeze_iter=1,
                unfreeze_epoch=2,
            )

        with self.assertRaisesRegex(ValueError, '`unfreeze_iter`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_iter=0,
                unfreeze_layers='backbone.*',
                unfreeze_iter=-1,
            )

        with self.assertRaisesRegex(ValueError, '`freeze_iter`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_iter=None,
                unfreeze_layers='backbone.*',
                unfreeze_iter=1,
            )

        with self.assertRaisesRegex(ValueError, '`unfreeze_iter`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_iter=2,
                unfreeze_layers='backbone.*',
                unfreeze_iter=1,
            )

        with self.assertRaisesRegex(ValueError, '`unfreeze_epoch`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_epoch=0,
                unfreeze_layers='backbone.*',
                unfreeze_epoch=-1,
            )

        with self.assertRaisesRegex(ValueError, '`freeze_epoch`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_epoch=None,
                unfreeze_layers='backbone.*',
                unfreeze_epoch=1,
            )

        with self.assertRaisesRegex(ValueError, '`unfreeze_epoch`'):
            FreezeHook(
                freeze_layers='backbone.*',
                freeze_epoch=2,
                unfreeze_layers='backbone.*',
                unfreeze_epoch=1,
            )

    def test_before_train_iter(self):
        cfg = copy.deepcopy(self.iter_based_cfg)
        runner = self.build_runner(cfg)

        freeze_hook = FreezeHook(
            freeze_layers='linear1.*|linear2.*',
            freeze_iter=1,
            unfreeze_layers='linear2.*',
            unfreeze_iter=3,
        )
        # Collect network layers that will be freeze or unfreeze.
        freeze_hook.before_train(runner)
        freeze_hook.before_train_iter(runner, 1)
        self.assertTrue(runner.model.linear1.weight.requires_grad)
        self.assertTrue(runner.model.linear2.weight.requires_grad)

        runner.train_loop._iter = 1
        freeze_hook.before_train_iter(runner, 1)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertFalse(runner.model.linear2.weight.requires_grad)

        runner.train_loop._iter = 2
        freeze_hook.before_train_iter(runner, 1)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertFalse(runner.model.linear2.weight.requires_grad)

        runner.train_loop._iter = 3
        freeze_hook.before_train_iter(runner, 1)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertTrue(runner.model.linear2.weight.requires_grad)

        runner.train_loop._iter = 4
        freeze_hook.before_train_iter(runner, 1)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertTrue(runner.model.linear2.weight.requires_grad)

    def test_before_train_epoch(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = self.build_runner(cfg)

        freeze_hook = FreezeHook(
            freeze_layers='linear1.*|linear2.*',
            freeze_epoch=1,
            unfreeze_layers='linear2.*',
            unfreeze_epoch=3,
        )
        # Collect network layers that will be freeze or unfreeze.
        freeze_hook.before_train(runner)
        freeze_hook.before_train_epoch(runner)
        self.assertTrue(runner.model.linear1.weight.requires_grad)
        self.assertTrue(runner.model.linear2.weight.requires_grad)

        runner.train_loop._epoch = 1
        freeze_hook.before_train_epoch(runner)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertFalse(runner.model.linear2.weight.requires_grad)

        runner.train_loop._epoch = 2
        freeze_hook.before_train_epoch(runner)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertFalse(runner.model.linear2.weight.requires_grad)

        runner.train_loop._epoch = 3
        freeze_hook.before_train_epoch(runner)
        freeze_hook.before_train_epoch(runner)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertTrue(runner.model.linear2.weight.requires_grad)

        runner.train_loop._epoch = 4
        freeze_hook.before_train_epoch(runner)
        freeze_hook.before_train_epoch(runner)
        self.assertFalse(runner.model.linear1.weight.requires_grad)
        self.assertTrue(runner.model.linear2.weight.requires_grad)
