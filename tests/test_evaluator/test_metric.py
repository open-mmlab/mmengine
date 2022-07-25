# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import unittest
from unittest import TestCase

import torch
from torch import Tensor

from mmengine.evaluator import DumpResults
from mmengine.evaluator.metric import _align_gpu_tensor_device
from mmengine.fileio import load


class TestDumpResults(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(ValueError,
                                    'The output file must be a pkl file.'):
            DumpResults(out_file_path='./results.json')

    def test_process(self):
        metric = DumpResults(out_file_path='./results.pkl')
        predictions = [dict(data=(Tensor([1, 2, 3]), Tensor([4, 5, 6])))]
        metric.process(None, predictions)
        self.assertEqual(len(metric.results), 1)
        self.assertEqual(metric.results[0]['data'][0].device,
                         torch.device('cpu'))

    def test_compute_metrics(self):
        temp_dir = tempfile.TemporaryDirectory()
        path = osp.join(temp_dir.name, 'results.pkl')
        metric = DumpResults(out_file_path=path)
        predictions = [dict(data=(Tensor([1, 2, 3]), Tensor([4, 5, 6])))]
        metric.process(None, predictions)
        metric.compute_metrics(metric.results)
        self.assertTrue(osp.isfile(path))

        results = load(path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['data'][0].device, torch.device('cpu'))

        temp_dir.cleanup()


class TestAlignGpuTensorDevice(TestCase):

    @unittest.skipUnless(torch.cuda.is_available(), 'must run with gpu')
    def test_align_gpu_tensor_device(self):
        data = [{
            'input': (torch.zeros(
                (1, 2), device='cuda'), torch.zeros((2, 1), device='cpu'))
        } for _ in range(5)]
        _align_gpu_tensor_device(data, device_id=0)
        for d in data:
            tenosr_gpu, tensor_cpu = d['input']
            self.assertEqual(tenosr_gpu.device,
                             torch.device(type='cuda', index=0))
            self.assertEqual(tensor_cpu.device, torch.device('cpu'))
