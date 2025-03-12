# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from mmengine.model import BaseDataPreprocessor, BaseModel
from mmengine.model.wrappers.pipeline_distributed import (
    _chunk_data, _convert_memory_map, _init_memory_map,
    _MMPipelineParallelFlag, _parameter_size)
from mmengine.structures import BaseDataElement
from mmengine.utils.version_utils import digit_version

if digit_version(torch.__version__) >= digit_version('2.0.0'):
    from mmengine.model.wrappers.pipeline_distributed import MMPipelineParallel


class ToyDataPreprocessor(BaseDataPreprocessor):

    def forward(self, data: dict, training: bool = False):
        self.called = True
        return super().forward(data, training)


class ToyVisionBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 200)
        self.linear2 = nn.Linear(200, 100)
        buffer1 = torch.randn(200)
        self.register_buffer('buffer', buffer1)

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = x + self.buffer
        x = self.linear2(x)
        return x


class ToyVisionModel(BaseModel):

    def __init__(self):
        super().__init__(data_preprocessor=ToyDataPreprocessor())
        self.backbone = ToyVisionBackbone()

    def forward(self, inputs, data_samples=None, mode='predict'):
        x = self.backbone(inputs)
        if mode == 'predict':
            return [BaseDataElement(x=x)]
        else:
            raise NotImplementedError


class ToyLanguageBackbone(nn.Module):

    def __init__(self):
        super().__init__()
        self.token = nn.Parameter(torch.randn(100))
        self.head = nn.Linear(100, 100)
        self.linear1 = nn.Linear(100, 200)
        self.linear2 = nn.Linear(200, 100)
        self.output = nn.Linear(100, 100)
        # tie weights
        self.output.weight = self.head.weight

    def forward(self, inputs):
        x = inputs + self.token
        x = self.head(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output(x)
        return x

    def generate(self, inputs):
        x = inputs + self.token
        for _ in range(5):
            x = self.head(x)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.output(x)
        return x


class ToyLanguageModel(BaseModel):

    def __init__(self):
        super().__init__(data_preprocessor=ToyDataPreprocessor())
        self.backbone = ToyLanguageBackbone()

    def forward(self, inputs, data_samples=None, mode='predict'):
        x = self.backbone(inputs)
        if mode == 'predict':
            return [BaseDataElement(x=x)]
        else:
            raise NotImplementedError

    def generate(self, inputs, data_samples=None, mode='predict'):
        x = self.backbone.generate(inputs)
        if mode == 'predict':
            return [BaseDataElement(x=x)]
        else:
            raise NotImplementedError

    def test_step(self, data):
        data = self.data_preprocessor(data, training=False)
        result = self.generate(**data)
        return result


class ToyMultiModalModel(BaseModel):

    def __init__(self):
        super().__init__(data_preprocessor=ToyDataPreprocessor())
        self.vision = ToyVisionBackbone()
        self.neck = nn.Linear(100, 100)
        self.language = ToyLanguageBackbone()

    def forward(self, inputs, data_samples=None, mode='predict'):
        x = self.vision(inputs)
        x = self.neck(x)
        x = self.language.generate(x)
        if mode == 'predict':
            return [BaseDataElement(x=x)]
        else:
            raise NotImplementedError


@unittest.skipIf(
    digit_version(torch.__version__) < digit_version('2.0.0'),
    'MMPipelineParallel is only available in PyTorch >= 2.0.0')
class TestPipelineParallel(unittest.TestCase):

    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is not available')
    def test_init_memory_map(self):
        # case 1: memory map is right
        toy_memory_map = {'cpu': '10MB', 'cuda:0': '1MB'}
        toy_threshold = 0.5
        memory_map = _init_memory_map(toy_memory_map, toy_threshold)
        self.assertEqual(memory_map['cpu'], 10 * 1000 * 1000 * 0.5)
        self.assertEqual(memory_map['cuda:0'], 1 * 1000 * 1000 * 0.5)
        # case 2: invalid device
        toy_memory_map = {'cpu': '10MB', 'toy': '1MB'}
        with self.assertRaises(ValueError):
            _init_memory_map(toy_memory_map, toy_threshold)
        # case 3: invalid memory size
        toy_memory_map = {'cpu': '1000GIB'}
        with self.assertRaises(ValueError):
            _init_memory_map(toy_memory_map, toy_threshold)

    def test_convert_memory_map(self):
        # case 1: memory map is right
        toy_memory_map = {
            '1': '1GIB',
            '2': '2MIB',
            '3': '3KIB',
            '4': '4GB',
            '5': '5MB',
            '6': '6KB',
            '7': '7Gb',
            '8': '8Mb',
            '9': '9Kb',
        }
        converted_memory_map = _convert_memory_map(toy_memory_map)
        self.assertEqual(converted_memory_map['1'], 1024 * 1024 * 1024)
        self.assertEqual(converted_memory_map['2'], 2 * 1024 * 1024)
        self.assertEqual(converted_memory_map['3'], 3 * 1024)
        self.assertEqual(converted_memory_map['4'], 4 * 1000 * 1000 * 1000)
        self.assertEqual(converted_memory_map['5'], 5 * 1000 * 1000)
        self.assertEqual(converted_memory_map['6'], 6 * 1000)
        self.assertEqual(converted_memory_map['7'],
                         7 * 1000 * 1000 * 1000 // 8)
        self.assertEqual(converted_memory_map['8'], 8 * 1000 * 1000 // 8)
        self.assertEqual(converted_memory_map['9'], 9 * 1000 // 8)
        # case 2: invalid memory format
        toy_memory_map = {'1': '1024'}
        with self.assertRaises(ValueError):
            _convert_memory_map(toy_memory_map)

    def test_parameter_size(self):
        model = ToyVisionModel()
        size = _parameter_size(model)
        target = (100 * 200 + 200 + 200 * 100 + 100) * 4
        # (weight + bias) * 4 bytes
        self.assertEqual(size, target)

    def test_chunk_data(self):
        toy_data = {}
        toy_data['tensor'] = torch.randn(5, 10)
        toy_data['list'] = [i for i in range(5)]
        toy_data['other'] = 'Test'
        chunked_data = _chunk_data(toy_data, 2)
        self.assertEqual(len(chunked_data), 2)
        self.assertEqual(chunked_data[0]['tensor'].shape, (3, 10))
        self.assertEqual(chunked_data[0]['list'], [0, 1, 2])
        self.assertEqual(chunked_data[0]['other'], 'Test')
        self.assertEqual(chunked_data[1]['tensor'].shape, (2, 10))
        self.assertEqual(chunked_data[1]['list'], [3, 4])
        self.assertEqual(chunked_data[1]['other'], 'Test')
        # smaller than chunk size
        toy_data_other = {'tensor': torch.randn(5, 10)}
        chunked_data_other = _chunk_data(toy_data_other, 7)
        self.assertEqual(len(chunked_data_other), 5)

    @unittest.skipIf(torch.cuda.device_count() < 2, 'CUDA is not available')
    def test_model_tree(self):
        # case 1 vision model
        model = ToyVisionModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        self.assertEqual(ppmodel.module_tree.module.__class__.__name__,
                         'ToyVisionModel')
        self.assertEqual(ppmodel.module_tree.parameters, None)
        self.assertEqual(ppmodel.module_tree.buffers, None)
        self.assertEqual(len(ppmodel.module_tree.submodules), 2)
        self.assertEqual(
            list(ppmodel.module_tree.submodules.keys()),
            ['data_preprocessor', 'backbone'])
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].module.__class__.
            __name__, 'ToyVisionBackbone')
        self.assertEqual(ppmodel.module_tree.submodules['backbone'].parameters,
                         None)
        self.assertEqual(
            len(ppmodel.module_tree.submodules['backbone'].buffers), 1)
        self.assertEqual(
            list(ppmodel.module_tree.submodules['backbone'].buffers.keys()),
            ['backbone.buffer'])
        self.assertEqual(
            len(ppmodel.module_tree.submodules['backbone'].submodules), 2)
        self.assertEqual(
            list(ppmodel.module_tree.submodules['backbone'].submodules.keys()),
            ['backbone.linear1', 'backbone.linear2'])
        # case 2 language model
        model = ToyLanguageModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        self.assertEqual(ppmodel.module_tree.module.__class__.__name__,
                         'ToyLanguageModel')
        self.assertEqual(ppmodel.module_tree.parameters, None)
        self.assertEqual(ppmodel.module_tree.buffers, None)
        self.assertEqual(len(ppmodel.module_tree.submodules), 2)
        self.assertEqual(
            list(ppmodel.module_tree.submodules.keys()),
            ['data_preprocessor', 'backbone'])
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].module.__class__.
            __name__, 'ToyLanguageBackbone')
        self.assertEqual(
            len(ppmodel.module_tree.submodules['backbone'].parameters), 1)
        self.assertEqual(
            list(ppmodel.module_tree.submodules['backbone'].parameters.keys()),
            ['backbone.token'])
        self.assertEqual(ppmodel.module_tree.submodules['backbone'].buffers,
                         None)
        self.assertEqual(
            len(ppmodel.module_tree.submodules['backbone'].submodules), 4)
        self.assertEqual(
            list(ppmodel.module_tree.submodules['backbone'].submodules.keys()),
            [
                'backbone.head', 'backbone.linear1', 'backbone.linear2',
                'backbone.output'
            ])
        # case 3 multimodal model
        model = ToyMultiModalModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        self.assertEqual(ppmodel.module_tree.module.__class__.__name__,
                         'ToyMultiModalModel')
        self.assertEqual(ppmodel.module_tree.parameters, None)
        self.assertEqual(ppmodel.module_tree.buffers, None)
        self.assertEqual(len(ppmodel.module_tree.submodules), 4)
        self.assertEqual(
            list(ppmodel.module_tree.submodules.keys()),
            ['data_preprocessor', 'vision', 'neck', 'language'])
        self.assertEqual(
            ppmodel.module_tree.submodules['neck'].module.__class__.__name__,
            'Linear')
        self.assertEqual(
            len(ppmodel.module_tree.submodules['neck'].parameters), 2)
        self.assertEqual(
            list(ppmodel.module_tree.submodules['neck'].parameters.keys()),
            ['neck.weight', 'neck.bias'])
        self.assertEqual(ppmodel.module_tree.submodules['neck'].buffers, None)
        self.assertEqual(ppmodel.module_tree.submodules['neck'].submodules,
                         None)

    @unittest.skipIf(torch.cuda.device_count() < 2, 'CUDA is not available')
    def test_iter_tree(self):
        model = ToyMultiModalModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        # case 1 empty
        toy_tree = ppmodel._iter_tree('')
        self.assertIs(toy_tree, ppmodel.module_tree)
        # case 2 parameters
        toy_tree = ppmodel._iter_tree('vision.linear1.weight')
        self.assertIs(
            toy_tree, ppmodel.module_tree.submodules['vision'].
            submodules['vision.linear1'].parameters['vision.linear1.weight'])
        # case 3 buffers
        toy_tree = ppmodel._iter_tree('vision.buffer')
        self.assertIs(
            toy_tree,
            ppmodel.module_tree.submodules['vision'].buffers['vision.buffer'])
        # case 4 submodules
        toy_tree = ppmodel._iter_tree('vision.linear1')
        self.assertIs(
            toy_tree, ppmodel.module_tree.submodules['vision'].
            submodules['vision.linear1'])
        # case 5
        model = ToyMultiModalModel()
        ppmodel = MMPipelineParallel(
            model,
            num_pipelines=2,
            no_split_module_classes='ToyVisionBackbone')
        toy_tree = ppmodel._iter_tree('vision.linear1')
        self.assertEqual(toy_tree, None)

    def test_flops_and_exec_order(self):
        # case 1
        model = ToyVisionModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': [BaseDataElement() for _ in range(5)]
        }
        ppmodel._get_flops_and_exec_order(toy_data)
        vision_flop = 100 * 200 + 200 * 100
        self.assertEqual(ppmodel.module_tree.flops, 5 * vision_flop)
        self.assertEqual(ppmodel.module_tree.exec_order, 0)
        self.assertEqual(ppmodel.module_tree.max_exec_order, 4)
        self.assertEqual(
            ppmodel.module_tree.submodules['data_preprocessor'].exec_order, -1)
        self.assertEqual(ppmodel.module_tree.submodules['backbone'].flops,
                         5 * vision_flop)
        self.assertEqual(ppmodel.module_tree.submodules['backbone'].exec_order,
                         1)
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.linear1'].flops, 5 * (100 * 200))
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.linear1'].exec_order, 2)
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.linear2'].flops, 5 * (200 * 100))
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.linear2'].exec_order, 3)
        # case 2
        model = ToyLanguageModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        ppmodel._get_flops_and_exec_order(toy_data)
        language_flop = 100 * 100 + 100 * 200 + 200 * 100 + 100 * 100
        self.assertEqual(ppmodel.module_tree.flops, 5 * language_flop)
        self.assertEqual(ppmodel.module_tree.exec_order, 0)
        self.assertEqual(ppmodel.module_tree.max_exec_order, 6)
        self.assertEqual(
            ppmodel.module_tree.submodules['data_preprocessor'].exec_order, -1)
        self.assertEqual(ppmodel.module_tree.submodules['backbone'].flops,
                         5 * language_flop)
        self.assertEqual(ppmodel.module_tree.submodules['backbone'].exec_order,
                         1)
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.head'].flops, 5 * (100 * 100))
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.head'].exec_order, 2)
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.linear1'].flops, 5 * (100 * 200))
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.linear1'].exec_order, 3)
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.linear2'].flops, 5 * (200 * 100))
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.linear2'].exec_order, 4)
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.output'].flops, 5 * (100 * 100))
        self.assertEqual(
            ppmodel.module_tree.submodules['backbone'].
            submodules['backbone.output'].exec_order, 5)
        # case 3
        model = ToyMultiModalModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        ppmodel._get_flops_and_exec_order(toy_data)
        multimodal_flop = vision_flop + 100 * 100 + 5 * language_flop
        self.assertEqual(ppmodel.module_tree.flops, 5 * multimodal_flop)
        self.assertEqual(ppmodel.module_tree.exec_order, 0)
        self.assertEqual(ppmodel.module_tree.max_exec_order, 9)
        self.assertEqual(
            ppmodel.module_tree.submodules['data_preprocessor'].exec_order, -1)
        self.assertEqual(ppmodel.module_tree.submodules['vision'].flops,
                         5 * vision_flop)
        self.assertEqual(ppmodel.module_tree.submodules['vision'].exec_order,
                         1)
        self.assertEqual(
            ppmodel.module_tree.submodules['vision'].
            submodules['vision.linear1'].flops, 5 * (100 * 200))
        self.assertEqual(
            ppmodel.module_tree.submodules['vision'].
            submodules['vision.linear1'].exec_order, 2)
        self.assertEqual(
            ppmodel.module_tree.submodules['vision'].
            submodules['vision.linear2'].flops, 5 * (200 * 100))
        self.assertEqual(
            ppmodel.module_tree.submodules['vision'].
            submodules['vision.linear2'].exec_order, 3)
        self.assertEqual(ppmodel.module_tree.submodules['neck'].flops,
                         5 * (100 * 100))
        self.assertEqual(ppmodel.module_tree.submodules['neck'].exec_order, 4)
        self.assertEqual(ppmodel.module_tree.submodules['language'].flops,
                         5 * 5 * language_flop)
        # because we do not call the forward of language model
        self.assertEqual(ppmodel.module_tree.submodules['language'].exec_order,
                         None)
        self.assertEqual(
            ppmodel.module_tree.submodules['language'].
            submodules['language.head'].flops, 5 * 5 * (100 * 100))
        self.assertEqual(
            ppmodel.module_tree.submodules['language'].
            submodules['language.head'].exec_order, 5)
        self.assertEqual(
            ppmodel.module_tree.submodules['language'].
            submodules['language.linear1'].flops, 5 * 5 * (100 * 200))
        self.assertEqual(
            ppmodel.module_tree.submodules['language'].
            submodules['language.linear1'].exec_order, 6)
        self.assertEqual(
            ppmodel.module_tree.submodules['language'].
            submodules['language.linear2'].flops, 5 * 5 * (200 * 100))
        self.assertEqual(
            ppmodel.module_tree.submodules['language'].
            submodules['language.linear2'].exec_order, 7)
        self.assertEqual(
            ppmodel.module_tree.submodules['language'].
            submodules['language.output'].flops, 5 * 5 * (100 * 100))
        self.assertEqual(
            ppmodel.module_tree.submodules['language'].
            submodules['language.output'].exec_order, 8)

    @unittest.skipIf(torch.cuda.device_count() < 2, 'CUDA is not available')
    def test_device_map(self):
        # case 1
        model = ToyVisionModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        ppmodel._prepare_forward(toy_data)
        device_map = {
            'data_preprocessor': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.buffer': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear1': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear2': {
                'part_id': 1,
                'init_device': 'cuda:1',
                'exec_device': 'cuda:1',
            },
        }
        self.assertEqual(ppmodel.device_map, device_map)
        # case 2
        model = ToyLanguageModel()
        ppmodel = MMPipelineParallel(
            model,
            num_pipelines=2,
            language_module_class='ToyLanguageBackbone')
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        ppmodel._prepare_forward(toy_data)
        device_map = {
            'data_preprocessor': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.token': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.head': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear1': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear2': {
                'part_id': 1,
                'init_device': 'cuda:1',
                'exec_device': 'cuda:1',
            },
            'backbone.output': {
                'part_id': 1,
                'init_device': 'cuda:1',
                'exec_device': 'cuda:1',
            }
        }
        self.assertEqual(ppmodel.device_map, device_map)
        # case 3
        model = ToyMultiModalModel()
        ppmodel = MMPipelineParallel(
            model,
            num_pipelines=2,
            no_split_module_classes=[
                'ToyVisionBackbone', 'ToyLanguageBackbone'
            ])
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        ppmodel._prepare_forward(toy_data)
        device_map = {
            'data_preprocessor': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'vision': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'neck': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'language': {
                'part_id': 1,
                'init_device': 'cuda:1',
                'exec_device': 'cuda:1',
            },
        }
        self.assertEqual(ppmodel.device_map, device_map)

    @unittest.skipIf(torch.cuda.device_count() < 2, 'CUDA is not available')
    def test_load_and_dispatch(self):
        # case 1
        model = ToyVisionModel()
        torch.save(model.state_dict(), 'load_cv_ckpt.pth')
        ppmodel = MMPipelineParallel(
            deepcopy(model), weights='load_cv_ckpt.pth', num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        ppmodel._prepare_forward(toy_data)
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.linear1.weight.cpu(),
                        model.backbone.linear1.weight.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.linear1.bias.cpu(),
                        model.backbone.linear1.bias.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.linear2.weight.cpu(),
                        model.backbone.linear2.weight.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.linear2.bias.cpu(),
                        model.backbone.linear2.bias.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.buffer.cpu(),
                        model.backbone.buffer.cpu()))
        # case 2
        model = ToyLanguageModel()
        torch.save(model.state_dict(), 'load_lm_ckpt.pth')
        ppmodel = MMPipelineParallel(
            deepcopy(model),
            weights='load_lm_ckpt.pth',
            num_pipelines=2,
            language_module_class='ToyLanguageBackbone')
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        ppmodel._prepare_forward(toy_data)
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.token.cpu(),
                        model.backbone.token.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.head.weight.cpu(),
                        model.backbone.head.weight.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.head.bias.cpu(),
                        model.backbone.head.bias.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.linear1.weight.cpu(),
                        model.backbone.linear1.weight.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.linear1.bias.cpu(),
                        model.backbone.linear1.bias.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.linear2.weight.cpu(),
                        model.backbone.linear2.weight.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.linear2.bias.cpu(),
                        model.backbone.linear2.bias.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.output.weight.cpu(),
                        model.backbone.output.weight.cpu()))
        self.assertTrue(
            torch.equal(ppmodel.module.backbone.output.bias.cpu(),
                        model.backbone.output.bias.cpu()))

    @unittest.skipIf(torch.cuda.device_count() < 2, 'CUDA is not available')
    def test_forward(self):
        # case 1
        model = ToyVisionModel()
        torch.save(model.state_dict(), 'fwd_cv_ckpt.pth')
        ppmodel = MMPipelineParallel(
            deepcopy(model), weights='fwd_cv_ckpt.pth', num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        with torch.no_grad():
            model.eval()
            model.cuda()
            toy_result = model.test_step(toy_data)[0].x.cpu()
        pp_result = ppmodel.test_step(toy_data)
        pp_result = torch.cat(
            [pp_result[i].x.cpu() for i in range(len(pp_result))], dim=0)
        self.assertTrue(torch.allclose(toy_result, pp_result, atol=1e-6))
        # case 2
        model = ToyLanguageModel()
        torch.save(model.state_dict(), 'fwd_lm_ckpt.pth')
        ppmodel = MMPipelineParallel(
            deepcopy(model),
            weights='fwd_lm_ckpt.pth',
            num_pipelines=2,
            language_module_class='ToyLanguageBackbone')
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        with torch.no_grad():
            model.eval()
            model.cuda()
            toy_result = model.test_step(toy_data)[0].x.cpu()
        pp_result = ppmodel.test_step(toy_data)
        pp_result = torch.cat(
            [pp_result[i].x.cpu() for i in range(len(pp_result))], dim=0)
        self.assertTrue(torch.allclose(toy_result, pp_result, atol=1e-6))
        # case 3
        model = ToyMultiModalModel()
        torch.save(model.state_dict(), 'fwd_mm_ckpt.pth')
        ppmodel = MMPipelineParallel(
            deepcopy(model), weights='fwd_mm_ckpt.pth', num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        with torch.no_grad():
            model.eval()
            model.cuda()
            toy_result = model.test_step(toy_data)[0].x.cpu()
        pp_result = ppmodel.test_step(toy_data)
        pp_result = torch.cat(
            [pp_result[i].x.cpu() for i in range(len(pp_result))], dim=0)
        self.assertTrue(torch.allclose(toy_result, pp_result, atol=1e-6))

    def test_disk_offload(self):
        model = ToyVisionModel()
        model.backbone.linear1.to(torch.bfloat16)
        device_map = {
            'data_preprocessor': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.buffer': {
                'part_id': 0,
                'init_device': 'cpu',
                'exec_device': 'cuda:0',
            },
            'backbone.linear1': {
                'part_id': 0,
                'init_device': 'disk',
                'exec_device': 'cuda:0',
            },
            'backbone.linear2': {
                'part_id': 1,
                'init_device': 'disk',
                'exec_device': 'cuda:1',
            },
        }
        ppmodel = MMPipelineParallel(
            model,
            num_pipelines=2,
            device_map=device_map,
            offload_directory='offloaded')
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        ppmodel._prepare_forward(toy_data)
        self.assertEqual(ppmodel.offload_map, {0: 0, 1: 0})
        offloaded_files = os.listdir(ppmodel.offload_directory)
        self.assertEqual(len(offloaded_files), 4)
        self.assertIn('backbone.linear1.weight.npy', offloaded_files)
        self.assertIn('backbone.linear1.bias.npy', offloaded_files)
        self.assertIn('backbone.linear2.weight.npy', offloaded_files)
        self.assertIn('backbone.linear2.bias.npy', offloaded_files)
        self.assertEqual(ppmodel.offloaded_weights['backbone.linear1.weight'],
                         {
                             'dtype': 'bfloat16',
                             'shape': [200, 100]
                         })
        self.assertEqual(ppmodel.offloaded_weights['backbone.linear1.bias'], {
            'dtype': 'bfloat16',
            'shape': [200]
        })
        self.assertEqual(ppmodel.offloaded_weights['backbone.linear2.weight'],
                         {
                             'dtype': np.dtype('float32'),
                             'shape': [100, 200]
                         })
        self.assertEqual(ppmodel.offloaded_weights['backbone.linear2.bias'], {
            'dtype': np.dtype('float32'),
            'shape': [100]
        })
        self.assertEqual(ppmodel.module.data_preprocessor.device,
                         torch.device('cuda:0'))
        self.assertEqual(ppmodel.module.backbone.buffer.device,
                         torch.device('cpu'))

    @unittest.skipIf(torch.cuda.device_count() < 2, 'CUDA is not available')
    def test_offload_forward(self):
        # case 1
        model = ToyVisionModel()
        torch.save(model.state_dict(), 'offload_cv_ckpt.pth')
        device_map = {
            'data_preprocessor': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.buffer': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear1': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear2': {
                'part_id': 1,
                'init_device': 'cpu',
                'exec_device': 'cuda:0',
            },
        }
        ppmodel = MMPipelineParallel(
            deepcopy(model),
            weights='offload_cv_ckpt.pth',
            num_pipelines=2,
            device_map=device_map)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        with torch.no_grad():
            model.eval()
            model.cuda()
            toy_result = model.test_step(toy_data)[0].x.cpu()
        pp_result = ppmodel.test_step(toy_data)
        pp_result = torch.cat(
            [pp_result[i].x.cpu() for i in range(len(pp_result))], dim=0)
        self.assertTrue(torch.allclose(toy_result, pp_result, atol=1e-6))
        # case 2
        model = ToyVisionModel()
        torch.save(model.state_dict(), 'offload_cv_ckpt.pth')
        device_map = {
            'data_preprocessor': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.buffer': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear1': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear2': {
                'part_id': 1,
                'init_device': 'disk',
                'exec_device': 'cuda:0',
            },
        }
        ppmodel = MMPipelineParallel(
            deepcopy(model),
            weights='offload_cv_ckpt.pth',
            num_pipelines=2,
            device_map=device_map)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        with torch.no_grad():
            model.eval()
            model.cuda()
            toy_result = model.test_step(toy_data)[0].x.cpu()
        pp_result = ppmodel.test_step(toy_data)
        pp_result = torch.cat(
            [pp_result[i].x.cpu() for i in range(len(pp_result))], dim=0)
        self.assertTrue(torch.allclose(toy_result, pp_result, atol=1e-6))
        # case 3
        model = ToyVisionModel().to(torch.bfloat16)
        torch.save(model.state_dict(), 'offload_cv_ckpt.pth')
        device_map = {
            'data_preprocessor': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.buffer': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear1': {
                'part_id': 0,
                'init_device': 'cuda:0',
                'exec_device': 'cuda:0',
            },
            'backbone.linear2': {
                'part_id': 1,
                'init_device': 'disk',
                'exec_device': 'cuda:0',
            },
        }
        ppmodel = MMPipelineParallel(
            deepcopy(model),
            weights='offload_cv_ckpt.pth',
            num_pipelines=2,
            device_map=device_map)
        toy_data = {
            'inputs': torch.randn(5, 100).to(torch.bfloat16),
            'data_samples': BaseDataElement()
        }
        pp_result = ppmodel.test_step(toy_data)
        pp_result = torch.cat(
            [pp_result[i].x.cpu() for i in range(len(pp_result))], dim=0)
        self.assertEqual(pp_result.dtype, torch.bfloat16)
        self.assertEqual(pp_result.shape, (5, 100))

    @unittest.skipIf(torch.cuda.device_count() < 2, 'CUDA is not available')
    def test_other_case(self):
        # case 1: `test_step` is not implemented
        model = ToyVisionBackbone()
        with self.assertRaises(NotImplementedError):
            MMPipelineParallel(model, num_pipelines=2)
        # case 2: num_pipelines
        model = ToyVisionModel()
        with self.assertWarns(Warning):
            MMPipelineParallel(model, num_pipelines=16)
        model = ToyVisionModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=None)
        self.assertEqual(ppmodel.num_pipelines, torch.cuda.device_count())
        # case 3: num_chunks
        model = ToyVisionModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2, num_chunks=2)
        self.assertEqual(ppmodel.num_chunks, 2)
        # case 4: device_map_policy
        model = ToyVisionModel()
        ppmodel = MMPipelineParallel(
            model, num_pipelines=2, device_map='invalid')
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        with self.assertRaises(ValueError):
            ppmodel._prepare_forward(toy_data)
        # case 5: too many num_pipelines
        model = ToyVisionModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=16)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        with self.assertRaises(RuntimeError):
            ppmodel._prepare_forward(toy_data)
        # case 6: Flag
        flag = _MMPipelineParallelFlag(1)
        self.assertEqual(str(flag), 'Part 1')
        self.assertEqual(repr(flag), 'Part 1')
        # case 7: train_step and val_step
        model = ToyVisionModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        with self.assertRaises(NotImplementedError):
            ppmodel.train_step(toy_data)
        val_result = ppmodel.val_step(toy_data)
        val_result = torch.cat(
            [val_result[i].x.cpu() for i in range(len(val_result))], dim=0)
        test_result = ppmodel.test_step(toy_data)
        test_result = torch.cat(
            [test_result[i].x.cpu() for i in range(len(test_result))], dim=0)
        self.assertTrue(torch.equal(val_result, test_result))
        # case 8: state_dict
        model = ToyVisionModel()
        torch.save(dict(state_dict=model.state_dict()), 'other_ckpt.pth')
        ppmodel = MMPipelineParallel(
            deepcopy(model), weights='other_ckpt.pth', num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        with torch.no_grad():
            model.eval()
            model.cuda()
            toy_result = model.test_step(toy_data)[0].x.cpu()
        pp_result = ppmodel.test_step(toy_data)
        pp_result = torch.cat(
            [pp_result[i].x.cpu() for i in range(len(pp_result))], dim=0)
        self.assertTrue(torch.allclose(toy_result, pp_result, atol=1e-6))
        # case 9: worker
        model = ToyVisionModel()
        ppmodel = MMPipelineParallel(model, num_pipelines=2)
        toy_data = {
            'inputs': torch.randn(5, 100),
            'data_samples': BaseDataElement()
        }
        ppmodel._prepare_forward(toy_data)
        ppmodel.is_inited = True
        toy_data = {
            'inputs': torch.randn(5, 10),
            'data_samples': BaseDataElement()
        }
        with self.assertRaises(RuntimeError):
            ppmodel.test_step(toy_data)
