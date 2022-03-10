# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import multiprocessing as mp
import os
import os.path as osp
import platform
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from mmengine.config import Config
from mmengine.data import DefaultSampler
from mmengine.evaluator import BaseEvaluator, build_evaluator
from mmengine.hooks import (Hook, IterTimerHook, OptimizerHook,
                            ParamSchedulerHook)
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.logging import MessageHub, MMLogger
from mmengine.model.wrappers import MMDataParallel, MMDistributedDataParallel
from mmengine.optim.scheduler import MultiStepLR, StepLR
from mmengine.registry import (DATASETS, EVALUATORS, HOOKS, LOOPS,
                               MODEL_WRAPPERS, MODELS, PARAM_SCHEDULERS,
                               Registry)
from mmengine.runner import (BaseLoop, EpochBasedTrainLoop, IterBasedTrainLoop,
                             Runner, TestLoop, ValLoop)
from mmengine.runner.priority import get_priority
from mmengine.utils import is_list_of


@MODELS.register_module()
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, data_batch, return_loss=False):
        input, label = zip(*data_batch)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        input = torch.stack(input).to(device)
        label = torch.stack(label).to(device)
        output = self.linear(input)
        if return_loss:
            loss = (label - output).sum()
            outputs = dict(loss=loss, log_vars=dict(loss=loss.item()))
            return outputs
        else:
            outputs = dict(log_vars=dict(a=1, b=0.5))
            return outputs


@DATASETS.register_module()
class ToyDataset(Dataset):
    META = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


@EVALUATORS.register_module()
class ToyEvaluator(BaseEvaluator):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


class TestRunner(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        full_cfg = dict(
            model=dict(type='ToyModel'),
            work_dir=self.temp_dir,
            train_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            test_dataloader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            optimizer=dict(type='SGD', lr=0.01),
            param_scheduler=dict(type='MultiStepLR', milestones=[1, 2]),
            evaluator=dict(type='ToyEvaluator'),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            val_cfg=dict(interval=1),
            test_cfg=dict(),
            custom_hooks=[],
            default_hooks=dict(
                timer=dict(type='IterTimerHook'),
                checkpoint=dict(type='CheckpointHook', interval=1),
                logger=dict(type='LoggerHook'),
                optimizer=dict(type='OptimizerHook', grad_clip=None),
                param_scheduler=dict(type='ParamSchedulerHook')),
            launcher='none',
            env_cfg=dict(dist_cfg=dict(backend='nccl')),
            logger=dict(log_level='INFO'),
        )
        self.full_cfg = Config(full_cfg)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # test arguments: train_dataloader, train_cfg, optimizer and
        # param_scheduler
        cfg = copy.deepcopy(self.full_cfg)
        cfg.pop('train_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        # all of training related configs are None
        cfg.pop('train_dataloader')
        cfg.pop('optimizer')
        cfg.pop('param_scheduler')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # all of training related configs are not None
        cfg = copy.deepcopy(self.full_cfg)
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # test argument: val_dataloader and val_cfg
        cfg = copy.deepcopy(self.full_cfg)
        cfg.pop('val_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        cfg.pop('val_dataloader')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        cfg = copy.deepcopy(self.full_cfg)
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # test arguments: test_dataloader and test_cfg
        cfg = copy.deepcopy(self.full_cfg)
        cfg.pop('test_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            runner = Runner(**cfg)

        cfg.pop('test_dataloader')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # test argument: evaluator
        cfg = copy.deepcopy(self.full_cfg)
        cfg.pop('evaluator')
        with self.assertRaisesRegex(ValueError,
                                    'evaluator should not be None'):
            Runner(**cfg)

        cfg = copy.deepcopy(self.full_cfg)
        cfg.pop('val_dataloader')
        cfg.pop('val_cfg')
        cfg.pop('test_dataloader')
        cfg.pop('test_cfg')
        with self.assertRaisesRegex(ValueError,
                                    'evaluator should not be None'):
            Runner(**cfg)

        # test env params
        cfg = copy.deepcopy(self.full_cfg)
        runner = Runner(**cfg)

        assert runner.distributed is False
        assert runner.seed is not None
        assert runner.work_dir == self.temp_dir

        # model should be initialized
        self.assertIsInstance(runner.model, (nn.Module, MMDataParallel))

        # lazy initialization
        self.assertIsInstance(runner.train_dataloader, dict)
        self.assertIsInstance(runner.val_dataloader, dict)
        self.assertIsInstance(runner.test_dataloader, dict)
        self.assertIsInstance(runner.optimizer, dict)
        self.assertIsInstance(runner.param_schedulers[0], dict)
        self.assertIsInstance(runner.evaluator, dict)

        # After calling runner.train(),
        # train_dataloader and val_loader should be initialized but
        # test_dataloader should also be dict
        runner.train()
        self.assertIsInstance(runner.train_dataloader, DataLoader)
        self.assertIsInstance(runner.val_dataloader, DataLoader)
        self.assertIsInstance(runner.optimizer, SGD)
        self.assertIsInstance(runner.evaluator, ToyEvaluator)
        self.assertIsInstance(runner.test_dataloader, dict)

        # After calling runner.test(), test_dataloader should be initialized
        runner.test()
        self.assertIsInstance(runner.test_dataloader, DataLoader)

    def test_build_from_cfg(self):
        runner = Runner.build_from_cfg(cfg=self.full_cfg)
        self.assertIsInstance(runner, Runner)

    def test_manually_init(self):
        model = ToyModel()
        optimizer = SGD(
            model.parameters(),
            lr=0.01,
        )

        class ToyHook(Hook):

            def before_train_epoch(self, runner):
                pass

        class ToyHook2(Hook):

            def after_train_epoch(self, runner):
                pass

        toy_hook = ToyHook()
        toy_hook2 = ToyHook2()
        runner = Runner(
            model=model,
            work_dir=self.temp_dir,
            train_dataloader=DataLoader(dataset=ToyDataset()),
            val_dataloader=DataLoader(dataset=ToyDataset()),
            optimizer=optimizer,
            param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
            evaluator=ToyEvaluator(),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            val_cfg=dict(interval=1),
            default_hooks=dict(param_scheduler=toy_hook),
            custom_hooks=[toy_hook2])
        runner.train()
        hook_names = [hook.__class__.__name__ for hook in runner.hooks]
        # test custom hook registered in runner
        assert 'ToyHook2' in hook_names
        # test default hook is replaced
        assert 'ToyHook' in hook_names
        # test other default hooks
        assert 'IterTimerHook' in hook_names

        # TODO
        # cannot run runner.test() when test_dataloader is None
        with self.assertRaisesRegex(AssertionError,
                                    'test dataloader does not exist'):
            runner.test()

        # TODO
        # cannot run runner.train() when optimizer is None
        with self.assertRaisesRegex(AssertionError,
                                    'optimizer does not exist'):
            runner = Runner(
                model=model,
                train_dataloader=DataLoader(dataset=ToyDataset()),
                val_dataloader=DataLoader(dataset=ToyDataset()),
                param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
                evaluator=ToyEvaluator(),
                train_cfg=dict(by_epoch=True, max_epochs=3),
                validation_cfg=dict(interval=1))
            runner.train()

        # cannot run runner.train() when validation_cfg is set
        # but val loader is None
        with self.assertRaisesRegex(AssertionError,
                                    'optimizer does not exist'):
            runner = Runner(
                model=model,
                train_dataloader=DataLoader(dataset=ToyDataset()),
                optimizer=optimizer,
                param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
                train_cfg=dict(by_epoch=True, max_epochs=3),
                validation_cfg=dict(interval=1))
            runner.train()

        # run runner.train() without validation
        runner = Runner(
            model=model,
            train_dataloader=DataLoader(dataset=ToyDataset()),
            optimizer=optimizer,
            param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
            train_cfg=dict(by_epoch=True, max_epochs=3),
            validation_cfg=None)
        runner.train()

    def test_setup_env(self):
        # temporarily store system setting
        sys_start_method = mp.get_start_method(allow_none=True)
        # pop and temp save system env vars
        sys_omp_threads = os.environ.pop('OMP_NUM_THREADS', default=None)
        sys_mkl_threads = os.environ.pop('MKL_NUM_THREADS', default=None)

        # test default multi-processing setting when workers > 1
        cfg = copy.deepcopy(self.full_cfg)
        cfg.train_dataloader.num_workers = 4
        cfg.test_dataloader.num_workers = 4
        cfg.val_dataloader.num_workers = 4
        Runner.build_from_cfg(cfg)
        assert os.getenv('OMP_NUM_THREADS') == '1'
        assert os.getenv('MKL_NUM_THREADS') == '1'
        if platform.system() != 'Windows':
            assert mp.get_start_method() == 'fork'

        # test default multi-processing setting when workers <= 1
        os.environ.pop('OMP_NUM_THREADS')
        os.environ.pop('MKL_NUM_THREADS')
        cfg = copy.deepcopy(self.full_cfg)
        cfg.train_dataloader.num_workers = 0
        cfg.test_dataloader.num_workers = 0
        cfg.val_dataloader.num_workers = 0
        Runner.build_from_cfg(cfg)
        assert 'OMP_NUM_THREADS' not in os.environ
        assert 'MKL_NUM_THREADS' not in os.environ

        # test manually set env var
        os.environ['OMP_NUM_THREADS'] = '3'
        cfg = copy.deepcopy(self.full_cfg)
        cfg.train_dataloader.num_workers = 2
        cfg.test_dataloader.num_workers = 2
        cfg.val_dataloader.num_workers = 2
        Runner.build_from_cfg(cfg)
        assert os.getenv('OMP_NUM_THREADS') == '3'

        # test manually set mp start method
        cfg = copy.deepcopy(self.full_cfg)
        cfg.env_cfg.mp_cfg = dict(mp_start_method='spawn')
        Runner.build_from_cfg(cfg)
        assert mp.get_start_method() == 'spawn'

        # revert setting to avoid affecting other programs
        if sys_start_method:
            mp.set_start_method(sys_start_method, force=True)
        if sys_omp_threads:
            os.environ['OMP_NUM_THREADS'] = sys_omp_threads
        else:
            os.environ.pop('OMP_NUM_THREADS')
        if sys_mkl_threads:
            os.environ['MKL_NUM_THREADS'] = sys_mkl_threads
        else:
            os.environ.pop('MKL_NUM_THREADS')

    def test_logger(self):
        runner = Runner.build_from_cfg(self.full_cfg)
        assert isinstance(runner.logger, MMLogger)
        # test latest logger and runner logger are the same
        assert runner.logger.level == logging.INFO
        assert MMLogger.get_instance(
        ).instance_name == runner.logger.instance_name
        # test latest message hub and runner message hub are the same
        assert isinstance(runner.message_hub, MessageHub)
        assert MessageHub.get_instance(
        ).instance_name == runner.message_hub.instance_name

        # test set log level in cfg
        self.full_cfg.log_cfg.log_level = 'DEBUG'
        runner = Runner.build_from_cfg(self.full_cfg)
        assert runner.logger.level == logging.DEBUG

    @patch('torch.distributed.get_rank', lambda: 0)
    @patch('torch.distributed.is_initialized', lambda: True)
    @patch('torch.distributed.is_available', lambda: True)
    def test_model_wrapper(self):
        # non-distributed model build from config
        runner = Runner.build_from_cfg(self.full_cfg)
        assert isinstance(runner.model, MMDataParallel)

        # non-distributed model build manually
        model = ToyModel()
        runner = Runner(
            model=model, train_cfg=dict(by_epoch=True, max_epochs=3))
        assert isinstance(runner.model, MMDataParallel)

        # distributed model build from config
        cfg = copy.deepcopy(self.full_cfg)
        cfg.launcher = 'pytorch'
        runner = Runner.build_from_cfg(cfg)
        assert isinstance(runner.model, MMDistributedDataParallel)

        # distributed model build manually
        model = ToyModel()
        runner = Runner(
            model=model,
            train_cfg=dict(by_epoch=True, max_epochs=3),
            env_cfg=dict(dist_params=dict(backend='nccl')),
            launcher='pytorch')
        assert isinstance(runner.model, MMDistributedDataParallel)

        # custom model wrapper
        @MODEL_WRAPPERS.register_module()
        class CustomModelWrapper:

            def train_step(self, *inputs, **kwargs):
                pass

            def val_step(self, *inputs, **kwargs):
                pass

        cfg = copy.deepcopy(self.full_cfg)
        cfg.model_wrapper = dict(type='CustomModelWrapper')
        runner = Runner.build_from_cfg(cfg)
        assert isinstance(runner.model, CustomModelWrapper)

    def test_default_scope(self):
        TOY_SCHEDULERS = Registry(
            'parameter scheduler', parent=PARAM_SCHEDULERS, scope='toy')

        @TOY_SCHEDULERS.register_module()
        class ToyScheduler(MultiStepLR):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        self.full_cfg.param_scheduler = dict(
            type='ToyScheduler', milestones=[1, 2])
        self.full_cfg.default_scope = 'toy'

        runner = Runner.build_from_cfg(self.full_cfg)
        runner.train()
        assert isinstance(runner.scheduler[0], ToyScheduler)

    def test_build_model(self):
        runner = Runner.build_from_cfg(self.full_cfg)

        model_cfg = dict(type='ToyModel')
        model = runner.build_model(model_cfg)
        assert isinstance(model, ToyModel)

        # Model does not implement `train_step` method
        @MODELS.register_module()
        class ToyModelv1(nn.Module):

            def __init__(self):
                super().__init__()

        model_cfg = dict(type='ToyModelv1')
        runner.model = None
        with self.assertRaisesRegex(RuntimeError, 'Model should implement'):
            runner.build_model(model_cfg)

    def test_build_optimizer(self):
        runner = Runner.build_from_cfg(self.full_cfg)
        cfg = dict(type='SGD', lr=0.01)
        optimizer = runner.build_optimizer(cfg)
        assert isinstance(optimizer, SGD)

    def test_build_param_scheduler(self):
        runner = Runner.build_from_cfg(self.full_cfg)
        cfg = dict(type='MultiStepLR', milestones=[1, 2])

        # Optimizer should be built before ParamScheduler
        runner.optimizer = None
        with self.assertRaises(RuntimeError):
            param_schedulers = runner.build_param_scheduler(cfg)

        runner.optimizer = runner.build_optimizer(dict(type='SGD', lr=0.01))

        # cfg is a dict
        param_schedulers = runner.build_param_scheduler(cfg)
        assert isinstance(param_schedulers[0], MultiStepLR)

        # cfg is a list of dict
        cfg = [
            dict(type='MultiStepLR', milestones=[1, 2]),
            dict(type='StepLR', step_size=1)
        ]
        param_schedulers = runner.build_param_scheduler(cfg)
        assert len(param_schedulers) == 2
        assert isinstance(param_schedulers[0], MultiStepLR)
        assert isinstance(param_schedulers[1], StepLR)

    def test_build_dataloader(self):
        runner = Runner.build_from_cfg(self.full_cfg)

        cfg = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=1,
            num_workers=0)
        dataloader = runner.build_dataloader(cfg)
        assert isinstance(dataloader, DataLoader)
        assert isinstance(dataloader.dataset, ToyDataset)
        assert isinstance(dataloader.sampler, DefaultSampler)

    def test_build_train_loop(self):
        # Only one of type or by_epoch can exist in cfg
        runner = Runner.build_from_cfg(self.full_cfg)
        cfg = dict(type='EpochBasedTrainLoop', by_epoch=True, max_epochs=3)
        with self.assertRaisesRegex(RuntimeError, 'Only one'):
            runner.build_train_loop(cfg)

        # type in cfg
        cfg = dict(type='EpochBasedTrainLoop', max_epochs=3)
        loop = runner.build_train_loop(cfg)
        assert isinstance(loop, EpochBasedTrainLoop)

        cfg = dict(type='IterBasedTrainLoop', max_iters=3)
        loop = runner.build_train_loop(cfg)
        assert isinstance(loop, IterBasedTrainLoop)

        # by_epoch in cfg
        cfg = dict(by_epoch=True, max_epochs=3)
        loop = runner.build_train_loop(cfg)
        assert isinstance(loop, EpochBasedTrainLoop)

        cfg = dict(by_epoch=False, max_iters=3)
        loop = runner.build_train_loop(cfg)
        assert isinstance(loop, IterBasedTrainLoop)

    def test_build_val_loop(self):
        runner = Runner.build_from_cfg(self.full_cfg)

        @LOOPS.register_module()
        class CustomValLoop(BaseLoop):

            def __init__(self, runner, dataloader, evaluator, interval=1):
                super().__init__(runner, dataloader)
                self._runner = runner

                if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
                    self.evaluator = build_evaluator(evaluator)  # type: ignore
                else:
                    self.evaluator = evaluator

            def run(self) -> None:
                pass

        # type in cfg
        cfg = dict(type='CustomValLoop', interval=1)
        loop = runner.build_val_loop(cfg)
        assert isinstance(loop, CustomValLoop)

        cfg = dict(type='ValLoop', interval=1)
        loop = runner.build_val_loop(cfg)
        assert isinstance(loop, ValLoop)

        # type not in cfg
        cfg = dict(interval=1)
        loop = runner.build_val_loop(cfg)
        assert isinstance(loop, ValLoop)

    def test_build_test_loop(self):
        runner = Runner.build_from_cfg(self.full_cfg)

        @LOOPS.register_module()
        class CustomTestLoop(BaseLoop):

            def __init__(self, runner, dataloader, evaluator):
                super().__init__(runner, dataloader)
                self._runner = runner

                if isinstance(evaluator, dict) or is_list_of(evaluator, dict):
                    self.evaluator = build_evaluator(evaluator)  # type: ignore
                else:
                    self.evaluator = evaluator

            def run(self) -> None:
                pass

        # type in cfg
        cfg = dict(type='CustomTestLoop')
        loop = runner.build_test_loop(cfg)
        assert isinstance(loop, CustomTestLoop)

        cfg = dict(type='TestLoop')
        loop = runner.build_test_loop(cfg)
        assert isinstance(loop, TestLoop)

        # type not in cfg
        cfg = dict()
        loop = runner.build_test_loop(cfg)
        assert isinstance(loop, TestLoop)

    def test_train(self):
        pass

    def test_val(self):
        pass

    def test_test(self):
        pass

    def test_register_hook(self):
        runner = Runner.build_from_cfg(self.full_cfg)
        runner._hooks = []

        # `hook` should be either a Hook object or dict
        # invalid `hook` type
        with self.assertRaisesRegex(
                TypeError, 'hook should be an instance of Hook or dict'):
            runner.register_hook(['string'])

        # `hook` is a dict
        timer_cfg = dict(type='IterTimerHook')
        runner.register_hook(timer_cfg)
        self.assertEqual(len(runner._hooks), 1)
        self.assertTrue(isinstance(runner._hooks[0], IterTimerHook))
        # default priority of `IterTimerHook` is 'NORMAL'
        self.assertEqual(
            get_priority(runner._hooks[0].priority), get_priority('NORMAL'))

        runner._hooks = []
        # `hook` is a dict and contains `priority` field
        # set the priority of `IterTimerHook` as 'BELOW_NORMAL'
        timer_cfg = dict(type='IterTimerHook', priority='BELOW_NORMAL')
        runner.register_hook(timer_cfg)
        self.assertEqual(len(runner._hooks), 1)
        self.assertTrue(isinstance(runner._hooks[0], IterTimerHook))

        self.assertEqual(
            get_priority(runner._hooks[0].priority),
            get_priority('BELOW_NORMAL'))

        # `hook` is a hook object
        optimizer_hook = OptimizerHook()
        runner.register_hook(optimizer_hook)
        self.assertEqual(len(runner._hooks), 2)
        # The priority of `OptimizerHook` is `HIGH` which is greater than
        # `IterTimerHook`, so the first item of `_hooks` should be
        # `OptimizerHook`
        self.assertTrue(isinstance(runner._hooks[0], OptimizerHook))
        self.assertEqual(
            get_priority(runner._hooks[0].priority), get_priority('HIGH'))

        # `priority` argument is not None and it will be set as priority of
        # hook
        param_scheduler_cfg = dict(type='ParamSchedulerHook', priority='LOW')
        runner.register_hook(param_scheduler_cfg, priority='VERY_LOW')
        self.assertEqual(len(runner._hooks), 3)
        self.assertTrue(isinstance(runner._hooks[2], ParamSchedulerHook))
        self.assertEqual(
            get_priority(runner._hooks[2].priority), get_priority('VERY_LOW'))

        # TODO: `priority` is Priority

    def test_default_hooks(self):
        runner = Runner.build_from_cfg(self.full_cfg)

        # register five hooks by default
        runner._hooks = []
        runner.register_default_hooks()
        self.assertEqual(len(runner._hooks), 5)
        # the forth registered hook should be `ParamSchedulerHook`
        self.assertTrue(isinstance(runner._hooks[3], ParamSchedulerHook))

        # remove `ParamSchedulerHook` from default hooks
        runner._hooks = []
        runner.register_default_hooks(hooks=dict(timer=None))
        self.assertEqual(len(runner._hooks), 4)
        # `ParamSchedulerHook` was popped so the forth is `CheckpointHook`
        self.assertTrue(isinstance(runner._hooks[3], CheckpointHook))

        # add a new default hook
        @HOOKS.register_module()
        class ToyHook(Hook):
            priority = 'Lowest'

        runner._hooks = []
        runner.register_default_hooks(hooks=dict(ToyHook=dict(type='ToyHook')))
        self.assertEqual(len(runner._hooks), 6)
        self.assertTrue(isinstance(runner._hooks[6], ToyHook))

    def test_custom_hooks(self):

        @HOOKS.register_module()
        class ToyHook(Hook):
            priority = 'Lowest'

        runner = Runner.build_from_cfg(self.full_cfg)
        self.assertEqual(len(runner._hooks), 5)
        custom_hooks = [dict(type='ToyHook')]
        runner.register_custom_hooks(custom_hooks)
        self.assertEqual(len(runner._hooks), 6)
        self.assertTrue(isinstance(runner._hooks[6], ToyHook))

    def test_register_hooks(self):
        runner = Runner.build_from_cfg(self.full_cfg)
        runner._hooks = []

        @HOOKS.register_module()
        class ToyHook(Hook):
            priority = 'Lowest'

        custom_hooks = [dict(type='ToyHook')]
        runner.register_hooks(custom_hooks=custom_hooks)
        # five default hooks + custom hook (ToyHook)
        self.assertEqual(len(runner._hooks), 6)

    def test_checkpoint(self):
        runner = Runner.build_from_cfg(self.full_cfg)
        runner.train()

        # test `save_checkpoint``
        path = osp.join(self.temp_dir, 'epoch_3.pth')
        runner.save_checkpoint(path)
        assert osp.exists(path)
        assert osp.exists(osp.join(self.temp_dir, 'latest.pth'))
        ckpt = torch.load(path)
        assert ckpt['meta']['epoch'] == 3
        # assert ckpt['meta']['iter'] =
        # assert ckpt['meta']['inner_iter'] =
        assert isinstance(ckpt['optimizer'], dict)
        assert isinstance(ckpt['param_schedulers'], list)

        # test `load_checkpoint`
        runner2 = Runner.build_from_cfg(self.full_cfg)
        runner2.load_checkpoint(path)

        self.assertEqual(runner2.epoch, 0)
        self.assertEqual(runner2.iter, 0)

        # test `resume`
        runner3 = Runner.build_from_cfg(self.full_cfg)
        runner3.resume(path)
        self.assertEqual(runner3.epoch, runner.epoch)
        self.assertEqual(runner3.iter, runner.iter)

    def test_iter_based(self):
        self.full_cfg.train_cfg = dict(by_epoch=False, max_iters=30)

        # test iter and epoch counter of IterBasedTrainLoop
        epoch_results = []
        iter_results = []
        inner_iter_results = []
        iter_targets = [i for i in range(30)]

        @HOOKS.register_module()
        class TestIterHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner):
                iter_results.append(runner.iter)
                inner_iter_results.append(runner.inner_iter)

        self.full_cfg.custom_hooks = [dict(type='TestIterHook', priority=50)]
        runner = Runner.build_from_cfg(self.full_cfg)

        assert isinstance(runner._train_loop, IterBasedTrainLoop)

        runner.train()

        self.assertEqual(len(epoch_results), 1)
        self.assertEqual(epoch_results[0], 0)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(inner_iter_results, iter_targets):
            self.assertEqual(result, target)

    def test_epoch_based(self):
        self.full_cfg.train_cfg = dict(by_epoch=True, max_epochs=3)

        # test iter and epoch counter of EpochBasedTrainLoop
        epoch_results = []
        epoch_targets = [i for i in range(3)]
        iter_results = []
        iter_targets = [i for i in range(10 * 3)]
        inner_iter_results = []
        inner_iter_targets = [i for i in range(10)] * 3  # train and val

        @HOOKS.register_module()
        class TestEpochHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner, data_batch=None):
                iter_results.append(runner.iter)
                inner_iter_results.append(runner.inner_iter)

        self.full_cfg.custom_hooks = [dict(type='TestEpochHook', priority=50)]
        runner = Runner.build_from_cfg(self.full_cfg)

        assert isinstance(runner._train_loop, EpochBasedTrainLoop)

        runner.train()

        for result, target, in zip(epoch_results, epoch_targets):
            self.assertEqual(result, target)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(inner_iter_results, inner_iter_targets):
            self.assertEqual(result, target)

    def test_custom_loop(self):
        # test custom loop with additional hook
        @LOOPS.register_module()
        class CustomTrainLoop(EpochBasedTrainLoop):
            """custom train loop with additional warmup stage."""

            def __init__(self, runner, loader, max_epochs, warmup_loader,
                         max_warmup_iters):
                super().__init__(
                    runner=runner, loader=loader, max_epochs=max_epochs)
                self.warmup_loader = self.runner.build_dataloader(
                    warmup_loader)
                self.max_warmup_iters = max_warmup_iters

            def run(self):
                self.runner.call_hooks('before_run')
                for idx, data_batch in enumerate(self.warmup_loader):
                    self.warmup_iter(data_batch)
                    if idx >= self.max_warmup_iters:
                        break

                self.runner.call_hooks('before_train_epoch')
                while self.runner.iter < self._max_iter:
                    data_batch = next(self.loader)
                    self.run_iter(data_batch)
                self.runner.call_hooks('after_train_epoch')
                self.runner.call_hooks('after_run')

            def warmup_iter(self, data_batch):
                self.runner.call_hooks(
                    'before_warmup_iter', args=dict(data_batch=data_batch))
                outputs = self.runner.model.train_step(data_batch)
                self.runner.call_hooks(
                    'after_warmup_iter',
                    args=dict(data_batch=data_batch, outputs=outputs))

        before_warmup_iter_results = []
        after_warmup_iter_results = []

        @HOOKS.register_module()
        class TestWarmupHook(Hook):
            """test custom train loop."""

            def before_warmup_iter(self, data_batch=None):
                before_warmup_iter_results.append('before')

            def after_warmup_iter(self, data_batch=None, outputs=None):
                after_warmup_iter_results.append('after')

        self.full_cfg.train_cfg = dict(
            type='CustomTrainLoop',
            max_epochs=3,
            warmup_loader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=0),
            max_warmup_iters=5)
        self.full_cfg.custom_hooks = [dict(type='TestWarmupHook', priority=50)]
        runner = Runner.build_from_cfg(self.full_cfg)

        assert isinstance(runner._train_loop, CustomTrainLoop)

        runner.train()

        # test custom hook triggered normally
        self.assertEqual(len(before_warmup_iter_results), 5)
        self.assertEqual(len(after_warmup_iter_results), 5)
        for before, after in zip(before_warmup_iter_results,
                                 after_warmup_iter_results):
            self.assertEqual(before, 'before')
            self.assertEqual(after, 'after')
