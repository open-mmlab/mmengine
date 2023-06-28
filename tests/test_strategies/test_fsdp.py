# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from tempfile import TemporaryDirectory
from unittest import TestCase, skipIf

import torch
import torch.nn as nn

try:
    from torch.distributed.fsdp import (FullStateDictConfig,
                                        FullyShardedDataParallel,
                                        LocalStateDictConfig, StateDictType)
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullOptimStateDictConfig, LocalOptimStateDictConfig)

    from mmengine._strategy import FSDPStrategy
except:  # noqa: E722
    pass
from torch.multiprocessing.spawn import start_processes
from torch.optim import SGD

from mmengine.dist import (all_gather_object, broadcast_object_list,
                           is_main_process)
from mmengine.optim import LinearLR, OptimWrapper
from mmengine.testing.runner_test_case import ToyModel
from mmengine.utils import digit_version


def linear_wrap_policy(
    module,
    recurse,
    nonwrapped_numel,
) -> bool:
    if recurse:
        return True  # always recurse
    return isinstance(module, nn.Linear)


@skipIf(
    digit_version(torch.__version__) < digit_version('2.0.0')
    or not torch.cuda.is_available(),
    'Only test FSDP with CUDA and PyTorch >= 2.0.0')
class TestStrategy(TestCase):

    def setUp(self):
        self.world_size = 2
        self.temp_dir = TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_init(self):
        strategy = FSDPStrategy()
        self.assertFalse(strategy.skip_init_weights)
        strategy = FSDPStrategy(state_dict_cfg='local')
        self._assert_local(strategy)

        strategy = FSDPStrategy(state_dict_cfg='full')
        self._assert_full(strategy)

        strategy = FSDPStrategy(
            state_dict_cfg=dict(
                state_dict_type=StateDictType.LOCAL_STATE_DICT))
        self._assert_local(strategy)

        strategy = FSDPStrategy(
            state_dict_cfg=dict(
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
                optim_state_dict_config=FullOptimStateDictConfig(),
            ))
        self._assert_full(strategy)

        strategy = FSDPStrategy(
            state_dict_cfg=dict(
                state_dict_type='FULL_STATE_DICT',
                state_dict_config=dict(type='FullStateDictConfig'),
                optim_state_dict_config=dict(type='FullOptimStateDictConfig'),
            ))
        self._assert_full(strategy)

        strategy = FSDPStrategy(
            state_dict_cfg=dict(
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=dict(type=FullStateDictConfig),
                optim_state_dict_config=dict(type=FullOptimStateDictConfig),
            ))
        self._assert_full(strategy)

        with self.assertRaises(ValueError):
            strategy = FSDPStrategy(state_dict_cfg='error-str')

        # state_dict_cfg should be a str or a dict
        with self.assertRaises(TypeError):
            strategy = FSDPStrategy(state_dict_cfg=[])

        # state_dict_type must be a str or a enumerate of StateDictType
        with self.assertRaises(TypeError):
            strategy = FSDPStrategy(
                state_dict_cfg=dict(
                    state_dict_type=[],
                    state_dict_config=dict(type=FullStateDictConfig),
                    optim_state_dict_config=dict(
                        type=FullOptimStateDictConfig),
                ))

        # state_dict_config should be a dict or a subclass of StateDictConfig
        with self.assertRaises(TypeError):
            strategy = FSDPStrategy(
                state_dict_cfg=dict(
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=[],
                    optim_state_dict_config=dict(
                        type=FullOptimStateDictConfig),
                ))

        # optim_state_dict_config should be a dict or a subclass of
        # OptimStateDictConfig
        with self.assertRaises(TypeError):
            strategy = FSDPStrategy(
                state_dict_cfg=dict(
                    state_dict_type=StateDictType.FULL_STATE_DICT,
                    state_dict_config=dict(type=FullStateDictConfig),
                    optim_state_dict_config=[],
                ))

    def run_strategy(self):
        # Strategy can run with the built model, optimizer and schedulers.
        for skip_init_weights, state_dict_cfg in [(True, 'local'),
                                                  (False, 'full')]:
            strategy = FSDPStrategy(
                skip_init_weights=skip_init_weights,
                state_dict_cfg=state_dict_cfg,
                model_wrapper=dict(auto_wrap_policy=linear_wrap_policy))
            model = ToyModel()
            optim = OptimWrapper(SGD(model.parameters(), lr=0.1, momentum=0.9))
            lr_scheduler = LinearLR(optimizer=optim)
            model, optim, lr_scheduler = strategy.prepare(
                model=model, optim_wrapper=optim, param_scheduler=lr_scheduler)
            self.assertIsInstance(model, FullyShardedDataParallel)
            self.assertIsInstance(model.linear1, FullyShardedDataParallel)
            self.assertIsInstance(model.linear2, FullyShardedDataParallel)

            data = torch.ones(2, 2).cuda()
            data_samples = torch.zeros(2, 2).cuda()
            loss = model(data, data_samples=data_samples, mode='loss')['loss']
            loss.backward()
            optim.step()
            [scheduler.step() for scheduler in lr_scheduler]

            ckpt_path = osp.join(self.temp_dir.name,
                                 f'checkpoint_{state_dict_cfg}.pth')
            strategy.save_checkpoint(ckpt_path)

            if state_dict_cfg == 'full':
                if not is_main_process():
                    self.assertFalse(osp.exists(ckpt_path))
                ckpt_path = [ckpt_path]
                broadcast_object_list(ckpt_path)
                ckpt_path = ckpt_path[0]

            strategy.load_checkpoint(ckpt_path)
            loss = model(data, data_samples=data_samples, mode='loss')['loss']
            loss.backward()
            optim.step()
            [scheduler.step() for scheduler in lr_scheduler]

        # optimizer with multiple param_groups can be reconstructed.
        model = ToyModel()
        strategy = FSDPStrategy(
            model_wrapper=dict(auto_wrap_policy=linear_wrap_policy))
        param_groups = []
        for param in model.parameters():
            param_groups.append(dict(params=[param], lr=0.1))
        optim = SGD(param_groups, lr=0.1, momentum=0.9)
        lr_scheduler = LinearLR(optimizer=optim)
        model, optim, lr_scheduler = strategy.prepare(
            model=model, optim_wrapper=optim, param_scheduler=lr_scheduler)
        data = torch.ones(2, 2).cuda()
        data_samples = torch.zeros(2, 2).cuda()
        loss = model(data, data_samples=data_samples, mode='loss')['loss']
        loss.backward()
        optim.step()
        [scheduler.step() for scheduler in lr_scheduler]
        optim_state = optim.state_dict()['state']
        optim_state = all_gather_object(optim_state)

    @classmethod
    def _worker(cls, rank, func):
        # local mode
        self = cls()
        self.setUp()
        self.rank = rank

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(self.world_size)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = str(12123)
        torch.cuda.set_device(f'cuda:{rank}')

        getattr(self, func)()
        self.tearDown()

    def test_run_strategy(self):
        start_processes(
            TestStrategy._worker,
            args=('run_strategy', ),
            nprocs=self.world_size)

    def test_build_model(self):
        ...
        # TODO
        # strategy = FSDPStrategy()
        # model = ToyModel()
        # state_dict = dict()

    def _assert_local(self, strategy):
        self.assertEqual(strategy.state_dict_type,
                         StateDictType.LOCAL_STATE_DICT)
        self.assertIsInstance(strategy.state_dict_config, LocalStateDictConfig)
        self.assertIsInstance(strategy.optim_state_dict_config,
                              LocalOptimStateDictConfig)

    def _assert_full(self, strategy):
        self.assertEqual(strategy.state_dict_type,
                         StateDictType.FULL_STATE_DICT)
        self.assertIsInstance(strategy.state_dict_config, FullStateDictConfig)
        self.assertIsInstance(strategy.optim_state_dict_config,
                              FullOptimStateDictConfig)
