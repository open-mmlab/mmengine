import copy
from unittest import TestCase
from unittest.mock import Mock
import tempfile
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from mmengine.hooks import ModifyStateDictHook
from mmengine.registry import DATASETS
from mmengine.runner import Runner


class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = Model1()
        self.module2 = Model1()


class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = Model2()
        self.module2 = Model2()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = Model1()
        self.module2 = Model2()
        self.module3 = Model3()


class ModelOld(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
        self.module2 = Model2()
        self.module3 = Model3()


@DATASETS.register_module()
class DummyToyDataset(Dataset):
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


class TestModifyStateDictHook(TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

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
        hook = ModifyStateDictHook(
            remove_keys=['module3.module2.module2.linear1.weight',
                         'module3.module2.module2.linear2.weight']
        )
        checkpoint = dict(state_dict=state_dict)
        hook.after_load_checkpoint(Mock(), checkpoint)

        self.assertNotIn('module3.module2.module2.linear1.weight', checkpoint[
            'state_dict'])
        self.assertNotIn('module3.module2.module2.linear2.weight', checkpoint[
            'state_dict'])
        # test state dict does not change.
        self.assertEqual(state_dict, ori_state_dict)

        # 1.2 Test remove keys with prefix
        checkpoint = dict(state_dict=state_dict)
        hook = ModifyStateDictHook(
            remove_keys=['module2', 'module3']
        )
        hook.after_load_checkpoint(Mock(), checkpoint)
        target_state_dict = dict()
        for key, value in state_dict.items():
            if key.startswith('module1'):
                target_state_dict[key] = value
        self.assertEqual(target_state_dict, checkpoint['state_dict'])

        # 1.3 Test overlapped removed keys
        with self.assertRaisesRegex(
                ValueError, 'removed_keys have a vague meaning'):
            checkpoint = dict(state_dict=state_dict)
            hook = ModifyStateDictHook(
                remove_keys=['module2.module1', 'module2']
            )
            hook.after_load_checkpoint(Mock(), checkpoint)

        # 2. Test merge state dict
        checkpoint = dict(state_dict=state_dict)
        merged_dict = {'a': 1, 'b': 2, 'module1.linear1.weight': 3}
        torch.save(merged_dict, osp.join(self.temp_dir.name, 'tmp_ckpt'))
        hook = ModifyStateDictHook(
            merged_state_dicts=[osp.join(self.temp_dir.name, 'tmp_ckpt')]
        )
        hook.after_load_checkpoint(Mock(), checkpoint)
        self.assertEqual(checkpoint['state_dict']['a'], 1)
        self.assertEqual(checkpoint['state_dict']['b'], 2)
        self.assertEqual(checkpoint['state_dict']['module1.linear1.weight'], 3)

        # Test remapping keys.
        # 3.1 Test modify single key
        checkpoint = dict(state_dict=state_dict)
        self.assertNotIn('module3.module2.module2.linear3.weight', state_dict)
        hook = ModifyStateDictHook(
            name_mappings=[dict(src='module3.module2.module2.linear1.weight',
                                dst='module3.module2.module2.linear3.weight')]
        )
        hook.after_load_checkpoint(Mock(), checkpoint)
        self.assertNotIn(
            'module3.module2.module2.linear1.weight', checkpoint['state_dict'])
        self.assertIn(
            'module3.module2.module2.linear3.weight', checkpoint['state_dict'])

        # 3.2 Test remapping prefix
        checkpoint = dict(state_dict=state_dict)
        self.assertNotIn('module3.linear2.weight', state_dict)
        hook = ModifyStateDictHook(
            name_mappings=[dict(src='module3.module2.module2',
                                dst='module3')]
        )
        hook.after_load_checkpoint(Mock(), checkpoint)
        self.assertIn('module3.linear2.weight', checkpoint['state_dict'])

        # 4 Test load state dict.
        checkpoint = dict(state_dict=state_dict)
        old_model = ModelOld()
        hook = ModifyStateDictHook(
            name_mappings=[dict(src='module1.',
                                dst='')]
        )
        hook.after_load_checkpoint(Mock(), checkpoint)
        old_model.load_state_dict(checkpoint['state_dict'])
