# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.hooks import EMAHook
from mmengine.model import ExponentialMovingAverage
from mmengine.registry import DATASETS, MODEL_WRAPPERS
from mmengine.runner import Runner


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, data_batch, return_loss=False):
        inputs, labels = [], []
        for x in data_batch:
            inputs.append(x['inputs'])
            labels.append(x['data_sample'])

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


@DATASETS.register_module()
class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class TestEMAHook(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_ema_hook(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = ToyModel().to(device)
        evaluator = Mock()
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
            optimizer=torch.optim.Adam(ToyModel().parameters()),
            train_cfg=dict(by_epoch=True, max_epochs=2),
            val_cfg=dict(interval=1),
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
        class DummyWrapper(nn.Module):

            def __init__(self, model):
                super().__init__()
                self.module = model

            def forward(self, *args, **kwargs):
                return self.module(*args, **kwargs)

        # with model wrapper
        runner = Runner(
            model=DummyWrapper(model),
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
