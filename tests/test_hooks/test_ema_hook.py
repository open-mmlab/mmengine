# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase
from unittest.mock import Mock
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.evaluator import Evaluator
from mmengine.hooks import EMAHook
from mmengine.model import BaseModel, ExponentialMovingAverage
from mmengine.optim import OptimWrapper
from mmengine.registry import DATASETS, MODEL_WRAPPERS
from mmengine.runner import Runner
from mmengine.testing import assert_allclose

class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, data_sample, mode='tensor'):
        labels = torch.stack(data_sample)
        inputs = torch.stack(inputs)
        outputs = self.linear(inputs)
        if mode == 'tensor':
            return outputs
        elif mode == 'loss':
            loss = (labels - outputs).sum()
            outputs = dict(loss=loss)
            return outputs
        else:
            return outputs


class ToyModel1(BaseModel, ToyModel):

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return super(BaseModel, self).forward(*args, **kwargs)


class ToyModel2(BaseModel, ToyModel):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 1)

    def forward(self, *args, **kwargs):
        return super(BaseModel, self).forward(*args, **kwargs)


@DATASETS.register_module()
class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class TestEMAHook(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        logging.shutdown()
        self.temp_dir.cleanup()

    def test_ema_hook(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = ToyModel1().to(device)
        evaluator = Evaluator([])
        evaluator.evaluate = Mock(return_value=dict(acc=0.5))
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=evaluator,
            work_dir=self.temp_dir.name,
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            val_cfg=dict(),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook', )],
            experiment_name='test1')
        runner.train()
        for hook in runner.hooks:
            if isinstance(hook, EMAHook):
                self.assertTrue(
                    isinstance(hook.ema_model, ExponentialMovingAverage))

        self.assertTrue(
            osp.exists(osp.join(self.temp_dir.name, 'epoch_2.pth')))
        checkpoint = torch.load(osp.join(self.temp_dir.name, 'epoch_2.pth'))
        self.assertTrue('ema_state_dict' in checkpoint)
        self.assertTrue(checkpoint['ema_state_dict']['steps'] == 8)

        # load and testing
        runner = Runner(
            model=model,
            test_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            test_evaluator=evaluator,
            test_cfg=dict(),
            work_dir=self.temp_dir.name,
            load_from=osp.join(self.temp_dir.name, 'epoch_2.pth'),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook')],
            experiment_name='test2')
        runner.test()

        @MODEL_WRAPPERS.register_module()
        class DummyWrapper(BaseModel):

            def __init__(self, model):
                super().__init__()
                self.module = model

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        # with model wrapper
        runner = Runner(
            model=DummyWrapper(ToyModel()),
            test_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            test_evaluator=evaluator,
            test_cfg=dict(),
            work_dir=self.temp_dir.name,
            load_from=osp.join(self.temp_dir.name, 'epoch_2.pth'),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook')],
            experiment_name='test3')
        runner.test()

        # Test load checkpoint without ema_state_dict
        ckpt = torch.load(osp.join(self.temp_dir.name, 'epoch_2.pth'))
        ckpt.pop('ema_state_dict')
        torch.save(ckpt,
                   osp.join(self.temp_dir.name, 'without_ema_state_dict.pth'))
        runner = Runner(
            model=DummyWrapper(ToyModel()),
            test_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            test_evaluator=evaluator,
            test_cfg=dict(),
            work_dir=self.temp_dir.name,
            load_from=osp.join(self.temp_dir.name,
                               'without_ema_state_dict.pth'),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook')],
            experiment_name='test4')
        runner.test()

        # Test does not load ckpt strict_loadly.
        # Test load checkpoint without ema_state_dict
        runner = Runner(
            model=ToyModel2(),
            test_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            test_evaluator=evaluator,
            test_cfg=dict(),
            work_dir=self.temp_dir.name,
            load_from=osp.join(self.temp_dir.name, 'epoch_2.pth'),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook', strict_load=False)],
            experiment_name='test5')
        runner.test()

        # Test enable ema at 5 epochs.
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=evaluator,
            work_dir=self.temp_dir.name,
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(by_epoch=True, max_epochs=10, val_interval=1),
            val_cfg=dict(),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook', begin_epoch=5)],
            experiment_name='test6')
        runner.train()
        state_dict = torch.load(
            osp.join(self.temp_dir.name, 'epoch_4.pth'), map_location='cpu')
        self.assertIn('ema_state_dict', state_dict)
        for k, v in state_dict['state_dict'].items():
            assert_allclose(v, state_dict['ema_state_dict']['module.' + k])
        state_dict = torch.load(
            osp.join(self.temp_dir.name, 'epoch_5.pth'), map_location='cpu')
        self.assertIn('ema_state_dict', state_dict)

        # Test enable ema at 5 iterations.
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            val_dataloader=dict(
                dataset=dict(type='DummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=False),
                batch_size=3,
                num_workers=0),
            val_evaluator=evaluator,
            work_dir=self.temp_dir.name,
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(ToyModel().parameters())),
            train_cfg=dict(by_epoch=False, max_iters=10, val_interval=1),
            val_cfg=dict(),
            default_hooks=dict(
                checkpoint=dict(
                    type='CheckpointHook', interval=1, by_epoch=False)),
            custom_hooks=[dict(type='EMAHook', begin_iter=5)],
            experiment_name='test7')
        runner.train()
        state_dict = torch.load(
            osp.join(self.temp_dir.name, 'iter_4.pth'), map_location='cpu')
        self.assertIn('ema_state_dict', state_dict)
        for k, v in state_dict['state_dict'].items():
            assert_allclose(v, state_dict['ema_state_dict']['module.' + k])
        state_dict = torch.load(
            osp.join(self.temp_dir.name, 'iter_5.pth'), map_location='cpu')
        self.assertIn('ema_state_dict', state_dict)
