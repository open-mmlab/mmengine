# Copyright (c) OpenMMLab. All rights reserved.
import os

from mmengine import Config
from mmengine.logging import MessageHub, MMLogger
from mmengine.registry import DefaultScope
from mmengine.testing import RunnerTestCase
from mmengine.visualization import Visualizer


class TestRunnerTestCase(RunnerTestCase):

    def test_setup(self):
        self.assertIsInstance(self.epoch_based_cfg, Config)
        self.assertIsInstance(self.iter_based_cfg, Config)
        self.assertIn('MASTER_ADDR', self.dist_cfg)
        self.assertIn('MASTER_PORT', self.dist_cfg)
        self.assertIn('RANK', self.dist_cfg)
        self.assertIn('WORLD_SIZE', self.dist_cfg)
        self.assertIn('LOCAL_RANK', self.dist_cfg)

    def test_tearDown(self):
        self.tearDown()
        self.assertEqual(MMLogger._instance_dict, {})
        self.assertEqual(MessageHub._instance_dict, {})
        self.assertEqual(Visualizer._instance_dict, {})
        self.assertEqual(DefaultScope._instance_dict, {})
        # tearDown should not be called twice.
        self.tearDown = super(RunnerTestCase, self).tearDown

    def test_build_runner(self):
        runner = self.build_runner(self.epoch_based_cfg)
        runner.train()
        runner.val()
        runner.test()

        runner = self.build_runner(self.iter_based_cfg)
        runner.train()
        runner.val()
        runner.test()

    def test_experiment_name(self):
        runner1 = self.build_runner(self.epoch_based_cfg)
        runner2 = self.build_runner(self.epoch_based_cfg)
        self.assertNotEqual(runner1.experiment_name, runner2.experiment_name)

    def test_init_dist(self):
        self.setup_dist_env()
        self.assertEqual(
            str(self.dist_cfg['MASTER_PORT']), os.environ['MASTER_PORT'])
        self.assertEqual(self.dist_cfg['MASTER_ADDR'],
                         os.environ['MASTER_ADDR'])
        self.assertEqual(self.dist_cfg['RANK'], os.environ['RANK'])
        self.assertEqual(self.dist_cfg['LOCAL_RANK'], os.environ['LOCAL_RANK'])
        self.assertEqual(self.dist_cfg['WORLD_SIZE'], os.environ['WORLD_SIZE'])
        fisrt_port = os.environ['MASTER_ADDR']
        self.setup_dist_env()
        self.assertNotEqual(fisrt_port, os.environ['MASTER_PORT'])
