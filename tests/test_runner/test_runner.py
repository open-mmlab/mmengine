# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os.path as osp
import shutil
import tempfile
import time
from unittest import TestCase

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from mmengine.config import Config
from mmengine.data import DefaultSampler
from mmengine.evaluator import (BaseEvaluator, ComposedEvaluator,
                                build_evaluator)
from mmengine.hooks import (Hook, IterTimerHook, LoggerHook, OptimizerHook,
                            ParamSchedulerHook)
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.logging import MessageHub, MMLogger
from mmengine.optim.scheduler import MultiStepLR, StepLR
from mmengine.registry import (DATASETS, EVALUATORS, HOOKS, LOOPS,
                               MODEL_WRAPPERS, MODELS, PARAM_SCHEDULERS,
                               Registry)
from mmengine.runner import (BaseLoop, EpochBasedTrainLoop, IterBasedTrainLoop,
                             Runner, TestLoop, ValLoop)
from mmengine.runner.priority import Priority, get_priority
from mmengine.utils import is_list_of
from mmengine.visualization.writer import ComposedWriter


@MODELS.register_module()
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, data_batch, return_loss=False):
        inputs, labels = zip(*data_batch)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = torch.stack(inputs).to(device)
        labels = torch.stack(labels).to(device)
        outputs = self.linear(inputs)
        if return_loss:
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss, log_vars=dict(loss=loss.item()))
            return outputs
        else:
            outputs = dict(log_vars=dict(a=1, b=0.5))
            return outputs


@MODELS.register_module()
class ToyModel1(ToyModel):

    def __init__(self):
        super().__init__()


@MODEL_WRAPPERS.register_module()
class CustomModelWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model


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
class ToyEvaluator1(BaseEvaluator):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


@EVALUATORS.register_module()
class ToyEvaluator2(BaseEvaluator):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


@HOOKS.register_module()
class ToyHook(Hook):
    priority = 'Lowest'

    def before_train_epoch(self, runner):
        pass


@HOOKS.register_module()
class ToyHook2(Hook):
    priority = 'Lowest'

    def after_train_epoch(self, runner):
        pass


@LOOPS.register_module()
class CustomTrainLoop(BaseLoop):

    def __init__(self, runner, dataloader, max_epochs):
        super().__init__(runner, dataloader)
        self._max_epochs = max_epochs

    def run(self) -> None:
        pass


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


def collate_fn(data_batch):
    return data_batch


class TestRunner(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        epoch_based_cfg = dict(
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
            val_evaluator=dict(type='ToyEvaluator1'),
            test_evaluator=dict(type='ToyEvaluator1'),
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
        )
        self.epoch_based_cfg = Config(epoch_based_cfg)
        self.iter_based_cfg = copy.deepcopy(self.epoch_based_cfg)
        self.iter_based_cfg.train_dataloader = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='InfiniteSampler', shuffle=True),
            batch_size=3,
            num_workers=0)
        self.iter_based_cfg.train_cfg = dict(by_epoch=False, max_iters=12)
        self.iter_based_cfg.default_hooks = dict(
            timer=dict(type='IterTimerHook'),
            checkpoint=dict(type='CheckpointHook', interval=1, by_epoch=False),
            logger=dict(type='LoggerHook'),
            optimizer=dict(type='OptimizerHook', grad_clip=None),
            param_scheduler=dict(type='ParamSchedulerHook'))

        time.sleep(1)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # 1. test arguments
        # 1.1 train_dataloader, train_cfg, optimizer and param_scheduler
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.pop('train_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        # all of training related configs are None
        cfg.pop('train_dataloader')
        cfg.pop('optimizer')
        cfg.pop('param_scheduler')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # avoid different runners having same timestamp
        time.sleep(1)

        # all of training related configs are not None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # 1.2 val_dataloader, val_evaluator, val_cfg
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.pop('val_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        time.sleep(1)

        cfg.pop('val_dataloader')
        cfg.pop('val_evaluator')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        time.sleep(1)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # 1.3 test_dataloader, test_evaluator and test_cfg
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.pop('test_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            runner = Runner(**cfg)

        time.sleep(1)

        cfg.pop('test_dataloader')
        cfg.pop('test_evaluator')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        time.sleep(1)

        # 1.4 test env params
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = Runner(**cfg)
        self.assertFalse(runner.distributed)
        self.assertFalse(runner.deterministic)

        time.sleep(1)

        # 1.5 message_hub, logger and writer
        # they are all not specified
        cfg = copy.deepcopy(self.epoch_based_cfg)
        runner = Runner(**cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertIsInstance(runner.writer, ComposedWriter)

        time.sleep(1)

        # they are all specified
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.logger = dict(name='test_logger')
        cfg.message_hub = dict(name='test_message_hub')
        cfg.writer = dict(name='test_writer')
        runner = Runner(**cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertEqual(runner.logger.instance_name, 'test_logger')
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertEqual(runner.message_hub.instance_name, 'test_message_hub')
        self.assertIsInstance(runner.writer, ComposedWriter)
        self.assertEqual(runner.writer.instance_name, 'test_writer')

        assert runner.distributed is False
        assert runner.seed is not None
        assert runner.work_dir == self.temp_dir

        # 2 model should be initialized
        self.assertIsInstance(runner.model,
                              (nn.Module, DistributedDataParallel))
        self.assertEqual(runner.model_name, 'ToyModel')

        # 3. test lazy initialization
        self.assertIsInstance(runner.train_dataloader, dict)
        self.assertIsInstance(runner.val_dataloader, dict)
        self.assertIsInstance(runner.test_dataloader, dict)
        self.assertIsInstance(runner.optimizer, dict)
        self.assertIsInstance(runner.param_schedulers[0], dict)

        # After calling runner.train(),
        # train_dataloader and val_loader should be initialized but
        # test_dataloader should also be dict
        runner.train()

        self.assertIsInstance(runner.train_loop, BaseLoop)
        self.assertIsInstance(runner.train_loop.dataloader, DataLoader)
        self.assertIsInstance(runner.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)
        self.assertIsInstance(runner.val_loop, BaseLoop)
        self.assertIsInstance(runner.val_loop.dataloader, DataLoader)
        self.assertIsInstance(runner.val_loop.evaluator, ToyEvaluator1)

        # After calling runner.test(), test_dataloader should be initialized
        self.assertIsInstance(runner.test_loop, dict)
        runner.test()
        self.assertIsInstance(runner.test_loop, BaseLoop)
        self.assertIsInstance(runner.test_loop.dataloader, DataLoader)
        self.assertIsInstance(runner.test_loop.evaluator, ToyEvaluator1)

        time.sleep(1)
        # 4. initialize runner with objects rather than config
        model = ToyModel()
        optimizer = SGD(
            model.parameters(),
            lr=0.01,
        )
        toy_hook = ToyHook()
        toy_hook2 = ToyHook2()

        train_dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        val_dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        test_dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        runner = Runner(
            model=model,
            work_dir=self.temp_dir,
            train_cfg=dict(by_epoch=True, max_epochs=3),
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            param_scheduler=MultiStepLR(optimizer, milestones=[1, 2]),
            val_cfg=dict(interval=1),
            val_dataloader=val_dataloader,
            val_evaluator=ToyEvaluator1(),
            test_cfg=dict(),
            test_dataloader=test_dataloader,
            test_evaluator=ToyEvaluator1(),
            default_hooks=dict(param_scheduler=toy_hook),
            custom_hooks=[toy_hook2])
        runner.train()
        runner.test()

        # 5. test `dump_config`
        # TODO

    def test_build_from_cfg(self):
        runner = Runner.build_from_cfg(cfg=self.epoch_based_cfg)
        self.assertIsInstance(runner, Runner)

    def test_setup_env(self):
        # TODO
        pass

    def test_logger(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertEqual(runner.experiment_name, runner.logger.instance_name)
        self.assertEqual(runner.logger.level, logging.NOTSET)

        # input is a MMLogger object
        self.assertEqual(
            id(runner.build_logger(runner.logger)), id(runner.logger))

        # input is None
        runner._experiment_name = 'logger_name1'
        logger = runner.build_logger(None)
        self.assertIsInstance(logger, MMLogger)
        self.assertEqual(logger.instance_name, 'logger_name1')

        # input is a dict
        log_cfg = dict(name='logger_name2')
        logger = runner.build_logger(log_cfg)
        self.assertIsInstance(logger, MMLogger)
        self.assertEqual(logger.instance_name, 'logger_name2')

        # input is a dict but does not contain name key
        runner._experiment_name = 'logger_name3'
        log_cfg = dict()
        logger = runner.build_logger(log_cfg)
        self.assertIsInstance(logger, MMLogger)
        self.assertEqual(logger.instance_name, 'logger_name3')

        # input is not a valid type
        with self.assertRaisesRegex(TypeError, 'logger should be'):
            runner.build_logger('invalid-type')

    def test_build_message_hub(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertEqual(runner.message_hub.instance_name,
                         runner.experiment_name)

        # input is a MessageHub object
        self.assertEqual(
            id(runner.build_message_hub(runner.message_hub)),
            id(runner.message_hub))

        # input is a dict
        message_hub_cfg = dict(name='message_hub_name1')
        message_hub = runner.build_message_hub(message_hub_cfg)
        self.assertIsInstance(message_hub, MessageHub)
        self.assertEqual(message_hub.instance_name, 'message_hub_name1')

        # input is a dict but does not contain name key
        runner._experiment_name = 'message_hub_name2'
        message_hub_cfg = dict()
        message_hub = runner.build_message_hub(message_hub_cfg)
        self.assertIsInstance(message_hub, MessageHub)
        self.assertEqual(message_hub.instance_name, 'message_hub_name2')

        # input is not a valid type
        with self.assertRaisesRegex(TypeError, 'message_hub should be'):
            runner.build_message_hub('invalid-type')

    def test_build_writer(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.writer, ComposedWriter)
        self.assertEqual(runner.experiment_name, runner.writer.instance_name)

        # input is a ComposedWriter object
        self.assertEqual(
            id(runner.build_writer(runner.writer)), id(runner.writer))

        # input is a dict
        writer_cfg = dict(name='writer_name1')
        writer = runner.build_writer(writer_cfg)
        self.assertIsInstance(writer, ComposedWriter)
        self.assertEqual(writer.instance_name, 'writer_name1')

        # input is a dict but does not contain name key
        runner._experiment_name = 'writer_name2'
        writer_cfg = dict()
        writer = runner.build_writer(writer_cfg)
        self.assertIsInstance(writer, ComposedWriter)
        self.assertEqual(writer.instance_name, 'writer_name2')

        # input is not a valid type
        with self.assertRaisesRegex(TypeError, 'writer should be'):
            runner.build_writer('invalid-type')

    def test_default_scope(self):
        TOY_SCHEDULERS = Registry(
            'parameter scheduler', parent=PARAM_SCHEDULERS, scope='toy')

        @TOY_SCHEDULERS.register_module()
        class ToyScheduler(MultiStepLR):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        self.epoch_based_cfg.param_scheduler = dict(
            type='ToyScheduler', milestones=[1, 2])
        self.epoch_based_cfg.default_scope = 'toy'

        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.train()
        self.assertIsInstance(runner.param_schedulers[0], ToyScheduler)

    def test_build_model(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.model, ToyModel)

        # input should be a nn.Module object or dict
        with self.assertRaisesRegex(TypeError, 'model should be'):
            runner.build_model('invalid-type')

        # input is a nn.Module object
        _model = ToyModel1()
        model = runner.build_model(_model)
        self.assertEqual(id(model), id(_model))

        # input is a dict
        model = runner.build_model(dict(type='ToyModel1'))
        self.assertIsInstance(model, ToyModel1)

    def test_wrap_model(self):
        # TODO: test on distributed environment
        # custom model wrapper
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.model_wrapper_cfg = dict(type='CustomModelWrapper')
        runner = Runner.build_from_cfg(cfg)
        self.assertIsInstance(runner.model, CustomModelWrapper)

    def test_build_optimizer(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        # input should be an Optimizer object or dict
        with self.assertRaisesRegex(TypeError, 'optimizer should be'):
            runner.build_optimizer('invalid-type')

        # input is an Optimizer object
        _optimizer = SGD(runner.model.parameters(), lr=0.01)
        optimizer = runner.build_optimizer(_optimizer)
        self.assertEqual(id(_optimizer), id(optimizer))

        # input is a dict
        optimizer = runner.build_optimizer(dict(type='SGD', lr=0.01))
        self.assertIsInstance(optimizer, SGD)

    def test_build_param_scheduler(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        # `build_optimizer` should be called before `build_param_scheduler`
        cfg = dict(type='MultiStepLR', milestones=[1, 2])
        runner.optimizer = None
        with self.assertRaisesRegex(RuntimeError, 'should be called before'):
            runner.build_param_scheduler(cfg)

        runner.optimizer = runner.build_optimizer(dict(type='SGD', lr=0.01))
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertIsInstance(param_schedulers, list)
        self.assertEqual(len(param_schedulers), 1)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)

        # input is a ParamScheduler object
        param_scheduler = MultiStepLR(runner.optimizer, milestones=[1, 2])
        param_schedulers = runner.build_param_scheduler(param_scheduler)
        self.assertEqual(id(param_schedulers[0]), id(param_scheduler))

        # input is a list of dict
        cfg = [
            dict(type='MultiStepLR', milestones=[1, 2]),
            dict(type='StepLR', step_size=1)
        ]
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertEqual(len(param_schedulers), 2)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)
        self.assertIsInstance(param_schedulers[1], StepLR)

        # input is a list and some items are ParamScheduler objects
        cfg = [param_scheduler, dict(type='StepLR', step_size=1)]
        param_schedulers = runner.build_param_scheduler(cfg)
        self.assertEqual(len(param_schedulers), 2)
        self.assertIsInstance(param_schedulers[0], MultiStepLR)
        self.assertIsInstance(param_schedulers[1], StepLR)

    def test_build_evaluator(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        # input is a BaseEvaluator or ComposedEvaluator object
        evaluator = ToyEvaluator1()
        self.assertEqual(id(runner.build_evaluator(evaluator)), id(evaluator))

        evaluator = ComposedEvaluator([ToyEvaluator1(), ToyEvaluator2()])
        self.assertEqual(id(runner.build_evaluator(evaluator)), id(evaluator))

        # input is a dict or list of dict
        evaluator = dict(type='ToyEvaluator1')
        self.assertIsInstance(runner.build_evaluator(evaluator), ToyEvaluator1)

        # input is a invalid type
        evaluator = [dict(type='ToyEvaluator1'), dict(type='ToyEvaluator2')]
        self.assertIsInstance(
            runner.build_evaluator(evaluator), ComposedEvaluator)

    def test_build_dataloader(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        cfg = dict(
            dataset=dict(type='ToyDataset'),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=1,
            num_workers=0)
        dataloader = runner.build_dataloader(cfg)
        self.assertIsInstance(dataloader, DataLoader)
        self.assertIsInstance(dataloader.dataset, ToyDataset)
        self.assertIsInstance(dataloader.sampler, DefaultSampler)

    def test_build_train_loop(self):
        # input should be a Loop object or dict
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        with self.assertRaisesRegex(TypeError, 'should be'):
            runner.build_train_loop('invalid-type')

        # Only one of type or by_epoch can exist in cfg
        cfg = dict(type='EpochBasedTrainLoop', by_epoch=True, max_epochs=3)
        with self.assertRaisesRegex(RuntimeError, 'Only one'):
            runner.build_train_loop(cfg)

        # input is a dict and contains type key
        cfg = dict(type='EpochBasedTrainLoop', max_epochs=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, EpochBasedTrainLoop)

        cfg = dict(type='IterBasedTrainLoop', max_iters=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, IterBasedTrainLoop)

        # input is a dict and does not contain type key
        cfg = dict(by_epoch=True, max_epochs=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, EpochBasedTrainLoop)

        cfg = dict(by_epoch=False, max_iters=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, IterBasedTrainLoop)

        # input is a Loop object
        self.assertEqual(id(runner.build_train_loop(loop)), id(loop))

        # test custom training loop
        cfg = dict(type='CustomTrainLoop', max_epochs=3)
        loop = runner.build_train_loop(cfg)
        self.assertIsInstance(loop, CustomTrainLoop)

    def test_build_val_loop(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        # input should be a Loop object or dict
        with self.assertRaisesRegex(TypeError, 'should be'):
            runner.build_test_loop('invalid-type')

        # input is a dict and contains type key
        cfg = dict(type='ValLoop', interval=1)
        loop = runner.build_test_loop(cfg)
        self.assertIsInstance(loop, ValLoop)

        # input is a dict but does not contain type key
        cfg = dict(interval=1)
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, ValLoop)

        # input is a Loop object
        self.assertEqual(id(runner.build_val_loop(loop)), id(loop))

        # test custom validation loop
        cfg = dict(type='CustomValLoop', interval=1)
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, CustomValLoop)

    def test_build_test_loop(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        # input should be a Loop object or dict
        with self.assertRaisesRegex(TypeError, 'should be'):
            runner.build_test_loop('invalid-type')

        # input is a dict and contains type key
        cfg = dict(type='TestLoop')
        loop = runner.build_test_loop(cfg)
        self.assertIsInstance(loop, TestLoop)

        # input is a dict but does not contain type key
        cfg = dict()
        loop = runner.build_test_loop(cfg)
        self.assertIsInstance(loop, TestLoop)

        # input is a Loop object
        self.assertEqual(id(runner.build_test_loop(loop)), id(loop))

        # test custom validation loop
        cfg = dict(type='CustomTestLoop')
        loop = runner.build_val_loop(cfg)
        self.assertIsInstance(loop, CustomTestLoop)

    def test_train(self):
        # 1. test `self.train_loop` is None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.pop('train_dataloader')
        cfg.pop('train_cfg')
        cfg.pop('optimizer')
        cfg.pop('param_scheduler')
        runner = Runner.build_from_cfg(cfg)
        with self.assertRaisesRegex(RuntimeError, 'should not be None'):
            runner.train()

        time.sleep(1)

        # 2. test iter and epoch counter of EpochBasedTrainLoop
        epoch_results = []
        epoch_targets = [i for i in range(3)]
        iter_results = []
        iter_targets = [i for i in range(4 * 3)]
        batch_idx_results = []
        batch_idx_targets = [i for i in range(4)] * 3  # train and val

        @HOOKS.register_module()
        class TestEpochHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner, batch_idx, data_batch=None):
                iter_results.append(runner.iter)
                batch_idx_results.append(batch_idx)

        self.epoch_based_cfg.custom_hooks = [
            dict(type='TestEpochHook', priority=50)
        ]
        runner = Runner.build_from_cfg(self.epoch_based_cfg)

        runner.train()

        assert isinstance(runner.train_loop, EpochBasedTrainLoop)

        for result, target, in zip(epoch_results, epoch_targets):
            self.assertEqual(result, target)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(batch_idx_results, batch_idx_targets):
            self.assertEqual(result, target)

        time.sleep(1)

        # 3. test iter and epoch counter of IterBasedTrainLoop
        epoch_results = []
        iter_results = []
        batch_idx_results = []
        iter_targets = [i for i in range(12)]
        batch_idx_targets = [i for i in range(12)]

        @HOOKS.register_module()
        class TestIterHook(Hook):

            def before_train_epoch(self, runner):
                epoch_results.append(runner.epoch)

            def before_train_iter(self, runner, batch_idx, data_batch=None):
                iter_results.append(runner.iter)
                batch_idx_results.append(batch_idx)

        self.iter_based_cfg.custom_hooks = [
            dict(type='TestIterHook', priority=50)
        ]
        self.iter_based_cfg.val_cfg = dict(interval=4)
        runner = Runner.build_from_cfg(self.iter_based_cfg)
        runner.train()

        assert isinstance(runner.train_loop, IterBasedTrainLoop)

        self.assertEqual(len(epoch_results), 1)
        self.assertEqual(epoch_results[0], 0)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(batch_idx_results, batch_idx_targets):
            self.assertEqual(result, target)

    def test_val(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.pop('val_dataloader')
        cfg.pop('val_cfg')
        cfg.pop('val_evaluator')
        runner = Runner.build_from_cfg(cfg)
        with self.assertRaisesRegex(RuntimeError, 'should not be None'):
            runner.val()

        time.sleep(1)
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.val()

    def test_test(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.pop('test_dataloader')
        cfg.pop('test_cfg')
        cfg.pop('test_evaluator')
        runner = Runner.build_from_cfg(cfg)
        with self.assertRaisesRegex(RuntimeError, 'should not be None'):
            runner.test()

        time.sleep(1)
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.test()

    def test_register_hook(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner._hooks = []

        # 1. test `hook` parameter
        # 1.1 `hook` should be either a Hook object or dict
        with self.assertRaisesRegex(
                TypeError, 'hook should be an instance of Hook or dict'):
            runner.register_hook(['string'])

        # 1.2 `hook` is a dict
        timer_cfg = dict(type='IterTimerHook')
        runner.register_hook(timer_cfg)
        self.assertEqual(len(runner._hooks), 1)
        self.assertTrue(isinstance(runner._hooks[0], IterTimerHook))
        # default priority of `IterTimerHook` is 'NORMAL'
        self.assertEqual(
            get_priority(runner._hooks[0].priority), get_priority('NORMAL'))

        runner._hooks = []
        # 1.2.1 `hook` is a dict and contains `priority` field
        # set the priority of `IterTimerHook` as 'BELOW_NORMAL'
        timer_cfg = dict(type='IterTimerHook', priority='BELOW_NORMAL')
        runner.register_hook(timer_cfg)
        self.assertEqual(len(runner._hooks), 1)
        self.assertTrue(isinstance(runner._hooks[0], IterTimerHook))
        self.assertEqual(
            get_priority(runner._hooks[0].priority),
            get_priority('BELOW_NORMAL'))

        # 1.3 `hook` is a hook object
        optimizer_hook = OptimizerHook()
        runner.register_hook(optimizer_hook)
        self.assertEqual(len(runner._hooks), 2)
        # The priority of `OptimizerHook` is `HIGH` which is greater than
        # `IterTimerHook`, so the first item of `_hooks` should be
        # `OptimizerHook`
        self.assertTrue(isinstance(runner._hooks[0], OptimizerHook))
        self.assertEqual(
            get_priority(runner._hooks[0].priority), get_priority('HIGH'))

        # 2. test `priority` parameter
        # `priority` argument is not None and it will be set as priority of
        # hook
        param_scheduler_cfg = dict(type='ParamSchedulerHook', priority='LOW')
        runner.register_hook(param_scheduler_cfg, priority='VERY_LOW')
        self.assertEqual(len(runner._hooks), 3)
        self.assertTrue(isinstance(runner._hooks[2], ParamSchedulerHook))
        self.assertEqual(
            get_priority(runner._hooks[2].priority), get_priority('VERY_LOW'))

        # `priority` is Priority
        logger_cfg = dict(type='LoggerHook', priority='BELOW_NORMAL')
        runner.register_hook(logger_cfg, priority=Priority.VERY_LOW)
        self.assertEqual(len(runner._hooks), 4)
        self.assertTrue(isinstance(runner._hooks[3], LoggerHook))
        self.assertEqual(
            get_priority(runner._hooks[3].priority), get_priority('VERY_LOW'))

    def test_default_hooks(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner._hooks = []

        # register five hooks by default
        runner.register_default_hooks()
        self.assertEqual(len(runner._hooks), 5)
        # the forth registered hook should be `ParamSchedulerHook`
        self.assertTrue(isinstance(runner._hooks[3], ParamSchedulerHook))

        runner._hooks = []
        # remove `ParamSchedulerHook` from default hooks
        runner.register_default_hooks(hooks=dict(timer=None))
        self.assertEqual(len(runner._hooks), 4)
        # `ParamSchedulerHook` was popped so the forth is `CheckpointHook`
        self.assertTrue(isinstance(runner._hooks[3], CheckpointHook))

        # add a new default hook
        runner._hooks = []
        runner.register_default_hooks(hooks=dict(ToyHook=dict(type='ToyHook')))
        self.assertEqual(len(runner._hooks), 6)
        self.assertTrue(isinstance(runner._hooks[5], ToyHook))

    def test_custom_hooks(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        self.assertEqual(len(runner._hooks), 5)
        custom_hooks = [dict(type='ToyHook')]
        runner.register_custom_hooks(custom_hooks)
        self.assertEqual(len(runner._hooks), 6)
        self.assertTrue(isinstance(runner._hooks[5], ToyHook))

    def test_register_hooks(self):
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner._hooks = []
        custom_hooks = [dict(type='ToyHook')]
        runner.register_hooks(custom_hooks=custom_hooks)
        # five default hooks + custom hook (ToyHook)
        self.assertEqual(len(runner._hooks), 6)
        self.assertTrue(isinstance(runner._hooks[5], ToyHook))

    def test_custom_loop(self):
        # test custom loop with additional hook
        @LOOPS.register_module()
        class CustomTrainLoop2(IterBasedTrainLoop):
            """Custom train loop with additional warmup stage."""

            def __init__(self, runner, dataloader, max_iters, warmup_loader,
                         max_warmup_iters):
                super().__init__(
                    runner=runner, dataloader=dataloader, max_iters=max_iters)
                self.warmup_loader = self.runner.build_dataloader(
                    warmup_loader)
                self.max_warmup_iters = max_warmup_iters

            def run(self):
                self.runner.call_hook('before_train')
                self.runner.cur_dataloader = self.warmup_loader
                for idx, data_batch in enumerate(self.warmup_loader, 1):
                    self.warmup_iter(data_batch)
                    if idx == self.max_warmup_iters:
                        break

                self.runner.cur_dataloader = self.warmup_loader
                self.runner.call_hook('before_train_epoch')
                while self.runner.iter < self._max_iters:
                    data_batch = next(self.dataloader)
                    self.run_iter(data_batch)
                self.runner.call_hook('after_train_epoch')

                self.runner.call_hook('after_train')

            def warmup_iter(self, data_batch):
                self.runner.call_hook(
                    'before_warmup_iter', data_batch=data_batch)
                self.runner.outputs = self.runner.model(
                    data_batch, return_loss=True)
                self.runner.call_hook(
                    'after_warmup_iter',
                    data_batch=data_batch,
                    outputs=self.runner.outputs)

        before_warmup_iter_results = []
        after_warmup_iter_results = []

        @HOOKS.register_module()
        class TestWarmupHook(Hook):
            """test custom train loop."""

            def before_warmup_iter(self, runner, data_batch=None):
                before_warmup_iter_results.append('before')

            def after_warmup_iter(self, runner, data_batch=None, outputs=None):
                after_warmup_iter_results.append('after')

        self.iter_based_cfg.train_cfg = dict(
            type='CustomTrainLoop2',
            max_iters=10,
            warmup_loader=dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='InfiniteSampler', shuffle=True),
                batch_size=1,
                num_workers=0),
            max_warmup_iters=5)
        self.iter_based_cfg.custom_hooks = [
            dict(type='TestWarmupHook', priority=50)
        ]
        runner = Runner.build_from_cfg(self.iter_based_cfg)
        runner.train()

        self.assertIsInstance(runner.train_loop, CustomTrainLoop2)

        # test custom hook triggered as expected
        self.assertEqual(len(before_warmup_iter_results), 5)
        self.assertEqual(len(after_warmup_iter_results), 5)
        for before, after in zip(before_warmup_iter_results,
                                 after_warmup_iter_results):
            self.assertEqual(before, 'before')
            self.assertEqual(after, 'after')

    def test_checkpoint(self):
        # 1. test epoch based
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.train()

        # 1.1 test `save_checkpoint` which called by `CheckpointHook`
        path = osp.join(self.temp_dir, 'epoch_3.pth')
        self.assertTrue(osp.exists(path))
        self.assertTrue(osp.exists(osp.join(self.temp_dir, 'latest.pth')))
        self.assertFalse(osp.exists(osp.join(self.temp_dir, 'epoch_4.pth')))

        ckpt = torch.load(path)
        self.assertEqual(ckpt['meta']['epoch'], 3)
        self.assertEqual(ckpt['meta']['iter'], 12)
        # self.assertEqual(ckpt['meta']['hook_msgs']['last_ckpt'], path)
        assert isinstance(ckpt['optimizer'], dict)
        assert isinstance(ckpt['param_schedulers'], list)

        time.sleep(1)
        # 1.2 test `load_checkpoint`
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.load_checkpoint(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 0)
        self.assertTrue(runner._has_loaded)

        time.sleep(1)
        # 1.3 test `resume`
        runner = Runner.build_from_cfg(self.epoch_based_cfg)
        runner.resume(path)
        self.assertEqual(runner.epoch, 3)
        self.assertEqual(runner.iter, 12)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)

        # 2. test iter based
        time.sleep(1)
        runner = Runner.build_from_cfg(self.iter_based_cfg)
        runner.train()

        # 2.1 test `save_checkpoint` which called by `CheckpointHook`
        path = osp.join(self.temp_dir, 'iter_12.pth')
        self.assertTrue(osp.exists(path))
        self.assertTrue(osp.exists(osp.join(self.temp_dir, 'latest.pth')))
        self.assertFalse(osp.exists(osp.join(self.temp_dir, 'epoch_13.pth')))

        ckpt = torch.load(path)
        self.assertEqual(ckpt['meta']['epoch'], 0)
        self.assertEqual(ckpt['meta']['iter'], 12)
        # self.assertEqual(ckpt['meta']['hook_msgs']['last_ckpt'], path)
        assert isinstance(ckpt['optimizer'], dict)
        assert isinstance(ckpt['param_schedulers'], list)

        # 2.2 test `load_checkpoint`
        time.sleep(1)
        runner = Runner.build_from_cfg(self.iter_based_cfg)
        runner.load_checkpoint(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 0)
        self.assertTrue(runner._has_loaded)

        time.sleep(1)
        # 2.3 test `resume`
        runner = Runner.build_from_cfg(self.iter_based_cfg)
        runner.resume(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 12)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)
