# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile
from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.device import get_device
from mmengine.hooks import RewriteCheckPointHook
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper
from mmengine.registry import DATASETS
from mmengine.runner import Runner


class Model1(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear1(x) + self.linear2(x)


class Model2(nn.Module):

    def __init__(self):
        super().__init__()
        self.module1 = Model1()
        self.module2 = Model1()

    def forward(self, x):
        return self.module1(x) + self.module2(x)


class Model3(nn.Module):

    def __init__(self):
        super().__init__()
        self.module1 = Model2()
        self.module2 = Model2()

    def forward(self, x):
        return self.module1(x) + self.module2(x)


class Model(BaseModel):

    def __init__(self):
        super().__init__()
        self.module1 = Model1()
        self.module2 = Model2()
        self.module3 = Model3()

    def forward(self, batch_inputs, labels, mode='tensor'):
        if mode == 'loss':
            loss = self.module1(self.module2(self.module3(batch_inputs)))
            return dict(loss=loss)
        else:
            return []


class ModelOld(BaseModel):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
        self.module2 = Model2()
        self.module3 = Model3()

    def forward(self, batch_inputs, labels, mode='tensor'):
        if mode == 'loss':
            loss = self.module2(self.linear1(self.linear2(batch_inputs)))
            return dict(loss=loss)
        else:
            return []


@DATASETS.register_module()
class ToyDummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 1)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class TestModifyStateDictHook(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_init(self):
        # Test with default constructor
        hook = RewriteCheckPointHook()
        self.assertEqual(hook.applied_key, 'state_dict')
        self.assertEqual(hook.removed_prefix, [])
        self.assertEqual(hook.prefix_mapping, [])
        self.assertEqual(hook.merged_state_dicts, [])

        # Test with non-list arguments
        hook = RewriteCheckPointHook(
            applied_key='ema_state_dict',
            removed_prefix='module',
            prefix_mapping=dict(src='layer', dst='linear'),
            merged_state_dicts='path_to_ckpt',
        )
        self.assertEqual(hook.applied_key, 'ema_state_dict')
        self.assertEqual(hook.removed_prefix, ['module'])
        self.assertEqual(hook.prefix_mapping,
                         [dict(src='layer', dst='linear')])
        self.assertEqual(hook.merged_state_dicts, ['path_to_ckpt'])

        # Test with error format arguments.
        with self.assertRaisesRegex(AssertionError,
                                    'applied_key should be a string'):
            RewriteCheckPointHook(applied_key=dict())

        with self.assertRaisesRegex(AssertionError,
                                    'removed_prefix should be a list'):
            RewriteCheckPointHook(removed_prefix=dict())

        with self.assertRaisesRegex(AssertionError,
                                    'prefix_mapping should be a list'):
            RewriteCheckPointHook(prefix_mapping='unknown')

    def test_after_load_checkpoint(self):
        model = Model()
        state_dict = model.state_dict()
        # module1.linear1
        # module1.linear2
        # module2.module1.linear1
        # module2.module1.linear2
        # module2.module2.linear1
        # module3.module1.module2.linear2
        # module3.module1.module1.linear1
        # module3.module1.module1.linear2
        # module3.module1.module2.linear1
        # module3.module2.module2.linear2
        # module3.module2.module2.linear2
        # module3.module2.module1.linear1
        # module3.module2.module1.linear2
        # module3.module2.module2.linear1
        # module3.module2.module2.linear2

        # Test remove
        # 1.1 remove specific key
        self.assertIn('module3.module2.module2.linear1.weight', state_dict)
        self.assertIn('module3.module2.module2.linear2.weight', state_dict)
        ori_state_dict = copy.deepcopy(state_dict)
        hook = RewriteCheckPointHook(removed_prefix=[
            'module3.module2.module2.linear1.weight',
            'module3.module2.module2.linear2.weight'
        ])
        checkpoint = dict(state_dict=state_dict)
        hook.after_load_checkpoint(Mock(), checkpoint)

        self.assertNotIn('module3.module2.module2.linear1.weight',
                         checkpoint['state_dict'])
        self.assertNotIn('module3.module2.module2.linear2.weight',
                         checkpoint['state_dict'])
        # test state dict does not change.
        self.assertEqual(state_dict, ori_state_dict)

        # 1.2 Test remove keys with prefix
        checkpoint = dict(state_dict=state_dict)
        hook = RewriteCheckPointHook(removed_prefix=['module2', 'module3'])
        hook.after_load_checkpoint(Mock(), checkpoint)
        target_state_dict = dict()
        for key, value in state_dict.items():
            if key.startswith('module1'):
                target_state_dict[key] = value
        self.assertEqual(target_state_dict, checkpoint['state_dict'])

        # 1.3 Test overlapped removed keys
        with self.assertRaisesRegex(ValueError,
                                    'removed_prefix have a vague meaning'):
            checkpoint = dict(state_dict=state_dict)
            hook = RewriteCheckPointHook(
                removed_prefix=['module2.module1', 'module2'])
            hook.after_load_checkpoint(Mock(), checkpoint)

        # 2. Test merge state dict
        checkpoint = dict(state_dict=state_dict)
        _merged_dict = {'a': 1, 'b': 2, 'module1.linear1.weight': 3}
        merged_dict = dict(state_dict=_merged_dict)
        torch.save(merged_dict, osp.join(self.temp_dir.name, 'tmp_ckpt'))
        hook = RewriteCheckPointHook(
            merged_state_dicts=[osp.join(self.temp_dir.name, 'tmp_ckpt')])
        hook.after_load_checkpoint(Mock(), checkpoint)
        self.assertEqual(checkpoint['state_dict']['a'], 1)
        self.assertEqual(checkpoint['state_dict']['b'], 2)
        self.assertEqual(checkpoint['state_dict']['module1.linear1.weight'], 3)

        # Test remapping keys.
        # 3.1 Test modify single key
        checkpoint = dict(state_dict=state_dict)
        self.assertNotIn('module3.module2.module2.linear3.weight', state_dict)
        hook = RewriteCheckPointHook(prefix_mapping=[
            dict(
                src='module3.module2.module2.linear1.weight',
                dst='module3.module2.module2.linear3.weight')
        ])
        hook.after_load_checkpoint(Mock(), checkpoint)
        self.assertNotIn('module3.module2.module2.linear1.weight',
                         checkpoint['state_dict'])
        self.assertIn('module3.module2.module2.linear3.weight',
                      checkpoint['state_dict'])

        # 3.2 Test remapping prefix
        checkpoint = dict(state_dict=state_dict)
        self.assertNotIn('module3.linear2.weight', state_dict)
        hook = RewriteCheckPointHook(prefix_mapping=[
            dict(src='module3.module2.module2', dst='module3')
        ])
        hook.after_load_checkpoint(Mock(), checkpoint)
        self.assertIn('module3.linear2.weight', checkpoint['state_dict'])

        # 4 Test load state dict.
        checkpoint = dict(state_dict=state_dict)
        old_model = ModelOld()
        hook = RewriteCheckPointHook(
            prefix_mapping=[dict(src='module1.', dst='')])
        hook.after_load_checkpoint(Mock(), checkpoint)
        old_model.load_state_dict(checkpoint['state_dict'])

    def test_with_runner(self):
        device = get_device()
        model = Model().to(device)
        runner = Runner(
            model=model,
            train_dataloader=dict(
                dataset=dict(type='ToyDummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            work_dir=self.temp_dir.name,
            optim_wrapper=OptimWrapper(torch.optim.Adam(model.parameters())),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            default_hooks=dict(logger=None),
            custom_hooks=[dict(type='EMAHook')],
            experiment_name='test_rewrite_hook1')
        runner.train()

        old_model = ModelOld()
        runner = Runner(
            model=old_model,
            train_dataloader=dict(
                dataset=dict(type='ToyDummyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=3,
                num_workers=0),
            work_dir=self.temp_dir.name,
            optim_wrapper=OptimWrapper(
                torch.optim.Adam(old_model.parameters())),
            train_cfg=dict(by_epoch=True, max_epochs=2, val_interval=1),
            default_hooks=dict(logger=None),
            custom_hooks=[
                dict(type='EMAHook'),
                dict(
                    type='RewriteCheckPointHook',
                    prefix_mapping=[dict(src='module1.', dst='')]),
                dict(
                    type='RewriteCheckPointHook',
                    applied_key='ema_state_dict',
                    prefix_mapping=[
                        dict(src='module.module1.', dst='module.')
                    ])
            ],
            experiment_name='test_rewrite_hook2',
            load_from=osp.join(self.temp_dir.name, 'epoch_2.pth'))
        runner.train()
