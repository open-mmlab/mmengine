# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
from unittest import TestCase

import torch
from torch import Tensor

from mmengine.evaluator import DumpResults
from mmengine.fileio import load


class TestDumpResults(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(ValueError,
                                    'The output file must be a pkl file.'):
            DumpResults(out_file_path='./results.json')

        # collect_dir could only be configured when collect_device='cpu'
        with self.assertRaises(ValueError):
            DumpResults(
                out_file_path='./results.json',
                collect_device='gpu',
                collect_dir='./tmp')

    def test_process(self):
        metric = DumpResults(out_file_path='./results.pkl')
        data_samples = [dict(data=(Tensor([1, 2, 3]), Tensor([4, 5, 6])))]
        metric.process(None, data_samples)
        self.assertEqual(len(metric.results), 1)
        self.assertEqual(metric.results[0]['data'][0].device,
                         torch.device('cpu'))

    def test_compute_metrics(self):
        temp_dir = tempfile.TemporaryDirectory()
        path = osp.join(temp_dir.name, 'results.pkl')
        metric = DumpResults(out_file_path=path)
        data_samples = [dict(data=(Tensor([1, 2, 3]), Tensor([4, 5, 6])))]
        metric.process(None, data_samples)
        metric.compute_metrics(metric.results)
        self.assertTrue(osp.isfile(path))

        results = load(path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['data'][0].device, torch.device('cpu'))

        temp_dir.cleanup()
