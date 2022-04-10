# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import shutil
import tempfile
from unittest import TestCase

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset

from mmengine.config import Config
from mmengine.data import DefaultSampler
from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.hooks import (DistSamplerSeedHook, Hook, IterTimerHook,
                            LoggerHook, OptimizerHook, ParamSchedulerHook)
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.logging import MessageHub, MMLogger
from mmengine.optim.scheduler import MultiStepLR, StepLR
from mmengine.registry import (DATASETS, HOOKS, LOOPS, METRICS, MODEL_WRAPPERS,
                               MODELS, PARAM_SCHEDULERS, Registry)
from mmengine.runner import (BaseLoop, EpochBasedTrainLoop, IterBasedTrainLoop,
                             Runner, TestLoop, ValLoop)
from mmengine.runner.priority import Priority, get_priority
from mmengine.utils import is_list_of
from mmengine.visualization import Visualizer


@MODELS.register_module()
class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, data_batch, return_loss=False):
        inputs, labels = zip(*map(lambda x:
                                  (x['inputs'], x['data_sample']), data_batch))
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
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


@METRICS.register_module()
class ToyMetric1(BaseMetric):

    def __init__(self, collect_device='cpu', dummy_metrics=None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'acc': 1}
        self.results.append(result)

    def compute_metrics(self, results):
        return dict(acc=1)


@METRICS.register_module()
class ToyMetric2(BaseMetric):

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
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
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
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
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
            val_evaluator=dict(type='ToyMetric1'),
            test_evaluator=dict(type='ToyMetric1'),
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
            logger=dict(type='LoggerHook', by_epoch=False),
            optimizer=dict(type='OptimizerHook', grad_clip=None),
            param_scheduler=dict(type='ParamSchedulerHook'))

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # 1. test arguments
        # 1.1 train_dataloader, train_cfg, optimizer and param_scheduler
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init1'
        cfg.pop('train_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        # all of training related configs are None and param_scheduler should
        # also be None
        cfg.experiment_name = 'test_init2'
        cfg.pop('train_dataloader')
        cfg.pop('optimizer')
        cfg.pop('param_scheduler')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # all of training related configs are not None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init3'
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # all of training related configs are not None and param_scheduler
        # can be None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init4'
        cfg.pop('param_scheduler')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # param_scheduler should be None when optimizer is None
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init5'
        cfg.pop('train_cfg')
        cfg.pop('train_dataloader')
        cfg.pop('optimizer')
        with self.assertRaisesRegex(ValueError, 'should be None'):
            runner = Runner(**cfg)

        # 1.2 val_dataloader, val_evaluator, val_cfg
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init6'
        cfg.pop('val_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            Runner(**cfg)

        cfg.experiment_name = 'test_init7'
        cfg.pop('val_dataloader')
        cfg.pop('val_evaluator')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init8'
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # 1.3 test_dataloader, test_evaluator and test_cfg
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init9'
        cfg.pop('test_cfg')
        with self.assertRaisesRegex(ValueError, 'either all None or not None'):
            runner = Runner(**cfg)

        cfg.experiment_name = 'test_init10'
        cfg.pop('test_dataloader')
        cfg.pop('test_evaluator')
        runner = Runner(**cfg)
        self.assertIsInstance(runner, Runner)

        # 1.4 test env params
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init11'
        runner = Runner(**cfg)
        self.assertFalse(runner.distributed)
        self.assertFalse(runner.deterministic)

        # 1.5 message_hub, logger and visualizer
        # they are all not specified
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init12'
        runner = Runner(**cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertIsInstance(runner.visualizer, Visualizer)

        # they are all specified
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_init13'
        cfg.log_level = 'INFO'
        cfg.visualizer = dict(name='test_visualizer')
        runner = Runner(**cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertIsInstance(runner.visualizer, Visualizer)

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
        self.assertIsInstance(runner.val_loop.evaluator, Evaluator)

        # After calling runner.test(), test_dataloader should be initialized
        self.assertIsInstance(runner.test_loop, dict)
        runner.test()
        self.assertIsInstance(runner.test_loop, BaseLoop)
        self.assertIsInstance(runner.test_loop.dataloader, DataLoader)
        self.assertIsInstance(runner.test_loop.evaluator, Evaluator)

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
            val_evaluator=ToyMetric1(),
            test_cfg=dict(),
            test_dataloader=test_dataloader,
            test_evaluator=ToyMetric1(),
            default_hooks=dict(param_scheduler=toy_hook),
            custom_hooks=[toy_hook2],
            experiment_name='test_init14')
        runner.train()
        runner.test()

        # 5. test `dump_config`
        # TODO

    def test_from_cfg(self):
        runner = Runner.from_cfg(cfg=self.epoch_based_cfg)
        self.assertIsInstance(runner, Runner)

    def test_setup_env(self):
        # TODO
        pass

    def test_build_logger(self):
        self.epoch_based_cfg.experiment_name = 'test_build_logger1'
        runner = Runner.from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.logger, MMLogger)
        self.assertEqual(runner.experiment_name, runner.logger.instance_name)

        # input is a dict
        logger = runner.build_logger(name='test_build_logger2')
        self.assertIsInstance(logger, MMLogger)
        self.assertEqual(logger.instance_name, 'test_build_logger2')

        # input is a dict but does not contain name key
        runner._experiment_name = 'test_build_logger3'
        logger = runner.build_logger()
        self.assertIsInstance(logger, MMLogger)
        self.assertEqual(logger.instance_name, 'test_build_logger3')

    def test_build_message_hub(self):
        self.epoch_based_cfg.experiment_name = 'test_build_message_hub1'
        runner = Runner.from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.message_hub, MessageHub)
        self.assertEqual(runner.message_hub.instance_name,
                         runner.experiment_name)

        # input is a dict
        message_hub_cfg = dict(name='test_build_message_hub2')
        message_hub = runner.build_message_hub(message_hub_cfg)
        self.assertIsInstance(message_hub, MessageHub)
        self.assertEqual(message_hub.instance_name, 'test_build_message_hub2')

        # input is a dict but does not contain name key
        runner._experiment_name = 'test_build_message_hub3'
        message_hub_cfg = dict()
        message_hub = runner.build_message_hub(message_hub_cfg)
        self.assertIsInstance(message_hub, MessageHub)
        self.assertEqual(message_hub.instance_name, 'test_build_message_hub3')

        # input is not a valid type
        with self.assertRaisesRegex(TypeError, 'message_hub should be'):
            runner.build_message_hub('invalid-type')

    def test_build_visualizer(self):
        self.epoch_based_cfg.experiment_name = 'test_build_visualizer1'
        runner = Runner.from_cfg(self.epoch_based_cfg)
        self.assertIsInstance(runner.visualizer, Visualizer)
        self.assertEqual(runner.experiment_name,
                         runner.visualizer.instance_name)

        # input is a Visualizer object
        self.assertEqual(
            id(runner.build_visualizer(runner.visualizer)),
            id(runner.visualizer))

        # input is a dict
        visualizer_cfg = dict(name='test_build_visualizer2')
        visualizer = runner.build_visualizer(visualizer_cfg)
        self.assertIsInstance(visualizer, Visualizer)
        self.assertEqual(visualizer.instance_name, 'test_build_visualizer2')

        # input is a dict but does not contain name key
        runner._experiment_name = 'test_build_visualizer3'
        visualizer_cfg = dict()
        visualizer = runner.build_visualizer(visualizer_cfg)
        self.assertIsInstance(visualizer, Visualizer)
        self.assertEqual(visualizer.instance_name, 'test_build_visualizer3')

        # input is not a valid type
        with self.assertRaisesRegex(TypeError, 'visualizer should be'):
            runner.build_visualizer('invalid-type')

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

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_default_scope'
        runner = Runner.from_cfg(cfg)
        runner.train()
        self.assertIsInstance(runner.param_schedulers[0], ToyScheduler)

    def test_build_model(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_model'
        runner = Runner.from_cfg(cfg)
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

        # test init weights
        @MODELS.register_module()
        class ToyModel2(ToyModel):

            def __init__(self):
                super().__init__()
                self.initiailzed = False

            def init_weights(self):
                self.initiailzed = True

        model = runner.build_model(dict(type='ToyModel2'))
        self.assertTrue(model.initiailzed)

        # test init weights with model object
        _model = ToyModel2()
        model = runner.build_model(_model)
        self.assertFalse(model.initiailzed)

    def test_wrap_model(self):
        # TODO: test on distributed environment
        # custom model wrapper
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_wrap_model'
        cfg.model_wrapper_cfg = dict(type='CustomModelWrapper')
        runner = Runner.from_cfg(cfg)
        self.assertIsInstance(runner.model, CustomModelWrapper)

    def test_build_optimizer(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_optimizer'
        runner = Runner.from_cfg(cfg)

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
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_param_scheduler'
        runner = Runner.from_cfg(cfg)

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
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_evaluator'
        runner = Runner.from_cfg(cfg)

        # input is a BaseEvaluator or ComposedEvaluator object
        evaluator = Evaluator(ToyMetric1())
        self.assertEqual(id(runner.build_evaluator(evaluator)), id(evaluator))

        evaluator = Evaluator([ToyMetric1(), ToyMetric2()])
        self.assertEqual(id(runner.build_evaluator(evaluator)), id(evaluator))

        # input is a dict
        evaluator = dict(type='ToyMetric1')
        self.assertIsInstance(runner.build_evaluator(evaluator), Evaluator)

        # input is a list of dict
        evaluator = [dict(type='ToyMetric1'), dict(type='ToyMetric2')]
        self.assertIsInstance(runner.build_evaluator(evaluator), Evaluator)

        # test collect device
        evaluator = [
            dict(type='ToyMetric1', collect_device='cpu'),
            dict(type='ToyMetric2', collect_device='gpu')
        ]
        _evaluator = runner.build_evaluator(evaluator)
        self.assertEqual(_evaluator.metrics[0].collect_device, 'cpu')
        self.assertEqual(_evaluator.metrics[1].collect_device, 'gpu')

    def test_build_dataloader(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_dataloader'
        runner = Runner.from_cfg(cfg)

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
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_train_loop'
        runner = Runner.from_cfg(cfg)

        # input should be a Loop object or dict
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
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_val_loop'
        runner = Runner.from_cfg(cfg)

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
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_build_test_loop'
        runner = Runner.from_cfg(cfg)

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
        cfg.experiment_name = 'test_train1'
        cfg.pop('train_dataloader')
        cfg.pop('train_cfg')
        cfg.pop('optimizer')
        cfg.pop('param_scheduler')
        runner = Runner.from_cfg(cfg)
        with self.assertRaisesRegex(RuntimeError, 'should not be None'):
            runner.train()

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

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_train2'
        cfg.custom_hooks = [dict(type='TestEpochHook', priority=50)]
        runner = Runner.from_cfg(cfg)

        runner.train()

        assert isinstance(runner.train_loop, EpochBasedTrainLoop)

        for result, target, in zip(epoch_results, epoch_targets):
            self.assertEqual(result, target)
        for result, target, in zip(iter_results, iter_targets):
            self.assertEqual(result, target)
        for result, target, in zip(batch_idx_results, batch_idx_targets):
            self.assertEqual(result, target)

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

        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_train3'
        cfg.custom_hooks = [dict(type='TestIterHook', priority=50)]
        cfg.val_cfg = dict(interval=4)
        runner = Runner.from_cfg(cfg)
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
        cfg.experiment_name = 'test_val1'
        cfg.pop('val_dataloader')
        cfg.pop('val_cfg')
        cfg.pop('val_evaluator')
        runner = Runner.from_cfg(cfg)
        with self.assertRaisesRegex(RuntimeError, 'should not be None'):
            runner.val()

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_val2'
        runner = Runner.from_cfg(cfg)
        runner.val()

    def test_test(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_test1'
        cfg.pop('test_dataloader')
        cfg.pop('test_cfg')
        cfg.pop('test_evaluator')
        runner = Runner.from_cfg(cfg)
        with self.assertRaisesRegex(RuntimeError, 'should not be None'):
            runner.test()

        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_test2'
        runner = Runner.from_cfg(cfg)
        runner.test()

    def test_register_hook(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_register_hook'
        runner = Runner.from_cfg(cfg)
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
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_default_hooks'
        runner = Runner.from_cfg(cfg)
        runner._hooks = []

        # register five hooks by default
        runner.register_default_hooks()
        self.assertEqual(len(runner._hooks), 6)
        # the third registered hook should be `DistSamplerSeedHook`
        self.assertTrue(isinstance(runner._hooks[2], DistSamplerSeedHook))
        # the fifth registered hook should be `ParamSchedulerHook`
        self.assertTrue(isinstance(runner._hooks[4], ParamSchedulerHook))

        runner._hooks = []
        # remove `ParamSchedulerHook` from default hooks
        runner.register_default_hooks(hooks=dict(timer=None))
        self.assertEqual(len(runner._hooks), 5)
        # `ParamSchedulerHook` was popped so the forth is `CheckpointHook`
        self.assertTrue(isinstance(runner._hooks[4], CheckpointHook))

        # add a new default hook
        runner._hooks = []
        runner.register_default_hooks(hooks=dict(ToyHook=dict(type='ToyHook')))
        self.assertEqual(len(runner._hooks), 7)
        self.assertTrue(isinstance(runner._hooks[6], ToyHook))

    def test_custom_hooks(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_custom_hooks'
        runner = Runner.from_cfg(cfg)

        self.assertEqual(len(runner._hooks), 6)
        custom_hooks = [dict(type='ToyHook')]
        runner.register_custom_hooks(custom_hooks)
        self.assertEqual(len(runner._hooks), 7)
        self.assertTrue(isinstance(runner._hooks[6], ToyHook))

    def test_register_hooks(self):
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_register_hooks'
        runner = Runner.from_cfg(cfg)

        runner._hooks = []
        custom_hooks = [dict(type='ToyHook')]
        runner.register_hooks(custom_hooks=custom_hooks)
        # six default hooks + custom hook (ToyHook)
        self.assertEqual(len(runner._hooks), 7)
        self.assertTrue(isinstance(runner._hooks[6], ToyHook))

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
        self.iter_based_cfg.experiment_name = 'test_custom_loop'
        runner = Runner.from_cfg(self.iter_based_cfg)
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
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint1'
        runner = Runner.from_cfg(cfg)
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

        # 1.2 test `load_checkpoint`
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint2'
        runner = Runner.from_cfg(cfg)
        runner.load_checkpoint(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 0)
        self.assertTrue(runner._has_loaded)

        # 1.3 test `resume`
        cfg = copy.deepcopy(self.epoch_based_cfg)
        cfg.experiment_name = 'test_checkpoint3'
        runner = Runner.from_cfg(cfg)
        runner.resume(path)
        self.assertEqual(runner.epoch, 3)
        self.assertEqual(runner.iter, 12)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)

        # 2. test iter based
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_checkpoint4'
        runner = Runner.from_cfg(cfg)
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
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_checkpoint5'
        runner = Runner.from_cfg(cfg)
        runner.load_checkpoint(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 0)
        self.assertTrue(runner._has_loaded)

        # 2.3 test `resume`
        cfg = copy.deepcopy(self.iter_based_cfg)
        cfg.experiment_name = 'test_checkpoint6'
        runner = Runner.from_cfg(cfg)
        runner.resume(path)
        self.assertEqual(runner.epoch, 0)
        self.assertEqual(runner.iter, 12)
        self.assertTrue(runner._has_loaded)
        self.assertIsInstance(runner.optimizer, SGD)
        self.assertIsInstance(runner.param_schedulers[0], MultiStepLR)
