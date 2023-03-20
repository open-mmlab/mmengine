# Copyright (c) OpenMMLab. All rights reserved.
import math
import unittest
from typing import Dict, List, Optional, Sequence
from unittest import TestCase

import numpy as np
import torch

from mmengine.evaluator import BaseMetric, Evaluator, get_metric_value
from mmengine.logging import MMLogger
from mmengine.registry import METRICS
from mmengine.structures import BaseDataElement


@METRICS.register_module()
class ToyMetric(BaseMetric):
    """Evaluaotr that calculates the metric `accuracy` from predictions and
    labels. Alternatively, this evaluator can return arbitrary dummy metrics
    set in the config.

    Default prefix: Toy

    Metrics:
        - accuracy (float): The classification accuracy. Only when
            `dummy_metrics` is None.
        - size (int): The number of test samples. Only when `dummy_metrics`
            is None.

        If `dummy_metrics` is set as a dict in the config, it will be
        returned as the metrics and override `accuracy` and `size`.
    """

    default_prefix = 'Toy'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 dummy_metrics: Optional[Dict] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.dummy_metrics = dummy_metrics

    def process(self, data_batch, predictions):
        results = [{
            'pred': prediction['label'],
            'label': prediction['label']
        } for prediction in predictions]
        self.results.extend(results)

    def compute_metrics(self, results: List):
        if self.dummy_metrics is not None:
            assert isinstance(self.dummy_metrics, dict)
            return self.dummy_metrics.copy()

        pred = np.array([result['pred'] for result in results])
        label = np.array([result['label'] for result in results])
        acc = (pred == label).sum() / pred.size

        metrics = {
            'accuracy': acc,
            'size': pred.size,  # To check the number of testing samples
        }

        return metrics


@METRICS.register_module()
class NonPrefixedMetric(BaseMetric):
    """Evaluator with unassigned `default_prefix` to test the warning
    information."""

    def process(self, data_batch, predictions: Sequence[dict]) -> None:
        pass

    def compute_metrics(self, results: list) -> dict:
        return dict(dummy=0.0)


def generate_test_results(size, batch_size, pred, label):
    num_batch = math.ceil(size / batch_size)
    bs_residual = size % batch_size
    for i in range(num_batch):
        bs = bs_residual if i == num_batch - 1 else batch_size
        data_batch = {
            'inputs': [np.zeros((3, 10, 10)) for _ in range(bs)],
            'data_sample': [BaseDataElement(label=label) for _ in range(bs)]
        }
        predictions = [
            BaseDataElement(pred=pred, label=label) for _ in range(bs)
        ]
        yield (data_batch, predictions)


class TestEvaluator(TestCase):

    def test_single_metric(self):
        cfg = dict(type='ToyMetric')
        evaluator = Evaluator(cfg)

        size = 10
        batch_size = 4

        for data_samples, outputs in generate_test_results(
                size, batch_size, pred=1, label=1):
            evaluator.process(data_samples=outputs, data_batch=data_samples)

        metrics = evaluator.evaluate(size=size)
        self.assertAlmostEqual(metrics['Toy/accuracy'], 1.0)
        self.assertEqual(metrics['Toy/size'], size)

        # Test empty results
        cfg = dict(type='ToyMetric', dummy_metrics=dict(accuracy=1.0))
        evaluator = Evaluator(cfg)
        # Warning should be raised if the results are empty
        with self.assertLogs(MMLogger.get_current_instance(), level='WARNING'):
            evaluator.evaluate(0)

    def test_composed_metrics(self):
        cfg = [
            dict(type='ToyMetric'),
            dict(type='ToyMetric', dummy_metrics=dict(mAP=0.0))
        ]

        evaluator = Evaluator(cfg)

        size = 10
        batch_size = 4

        for data_samples, outputs in generate_test_results(
                size, batch_size, pred=1, label=1):
            evaluator.process(data_samples=outputs, data_batch=data_samples)

        metrics = evaluator.evaluate(size=size)

        self.assertAlmostEqual(metrics['Toy/accuracy'], 1.0)
        self.assertAlmostEqual(metrics['Toy/mAP'], 0.0)
        self.assertEqual(metrics['Toy/size'], size)

    def test_ambiguous_metric(self):
        cfg = [
            dict(type='ToyMetric', dummy_metrics=dict(mAP=0.0)),
            dict(type='ToyMetric', dummy_metrics=dict(mAP=0.0))
        ]

        evaluator = Evaluator(cfg)

        size = 10
        batch_size = 4

        for data_samples, outputs in generate_test_results(
                size, batch_size, pred=1, label=1):
            evaluator.process(data_samples=outputs, data_batch=data_samples)

        with self.assertRaisesRegex(
                ValueError,
                'There are multiple evaluation results with the same metric '
                'name'):
            _ = evaluator.evaluate(size=size)

    def test_dataset_meta(self):
        dataset_meta = dict(classes=('cat', 'dog'))

        cfg = [
            dict(type='ToyMetric'),
            dict(type='ToyMetric', dummy_metrics=dict(mAP=0.0))
        ]

        evaluator = Evaluator(cfg)
        evaluator.dataset_meta = dataset_meta

        self.assertDictEqual(evaluator.dataset_meta, dataset_meta)
        for _evaluator in evaluator.metrics:
            self.assertDictEqual(_evaluator.dataset_meta, dataset_meta)

    def test_collect_device(self):
        cfg = [
            dict(type='ToyMetric', collect_device='cpu'),
            dict(
                type='ToyMetric',
                collect_device='gpu',
                dummy_metrics=dict(mAP=0.0))
        ]

        evaluator = Evaluator(cfg)
        self.assertEqual(evaluator.metrics[0].collect_device, 'cpu')
        self.assertEqual(evaluator.metrics[1].collect_device, 'gpu')

    def test_prefix(self):
        cfg = dict(type='NonPrefixedMetric')
        logger = MMLogger.get_current_instance()
        # Warning should be raised if prefix is not set.
        with self.assertLogs(logger, 'WARNING'):
            Evaluator(cfg)

    def test_get_metric_value(self):

        metrics = {
            'prefix_0/metric_0': 0,
            'prefix_1/metric_0': 1,
            'prefix_1/metric_1': 2,
            'nonprefixed': 3,
        }

        # Test indicator with prefix
        indicator = 'prefix_0/metric_0'  # correct indicator
        self.assertEqual(get_metric_value(indicator, metrics), 0)

        indicator = 'prefix_1/metric_0'  # correct indicator
        self.assertEqual(get_metric_value(indicator, metrics), 1)

        indicator = 'prefix_0/metric_1'  # unmatched indicator (wrong metric)
        with self.assertRaisesRegex(ValueError, 'can not match any metric'):
            _ = get_metric_value(indicator, metrics)

        indicator = 'prefix_2/metric'  # unmatched indicator (wrong prefix)
        with self.assertRaisesRegex(ValueError, 'can not match any metric'):
            _ = get_metric_value(indicator, metrics)

        # Test indicator without prefix
        indicator = 'metric_1'  # correct indicator (prefixed metric)
        self.assertEqual(get_metric_value(indicator, metrics), 2)

        indicator = 'nonprefixed'  # correct indicator (non-prefixed metric)
        self.assertEqual(get_metric_value(indicator, metrics), 3)

        indicator = 'metric_0'  # ambiguous indicator
        with self.assertRaisesRegex(ValueError, 'matches multiple metrics'):
            _ = get_metric_value(indicator, metrics)

        indicator = 'metric_2'  # unmatched indicator
        with self.assertRaisesRegex(ValueError, 'can not match any metric'):
            _ = get_metric_value(indicator, metrics)

    def test_offline_evaluate(self):
        cfg = dict(type='ToyMetric')
        evaluator = Evaluator(cfg)

        size = 10

        all_data = [dict() for _ in range(10)]
        all_predictions = [
            BaseDataElement(pred=0, label=1) for _ in range(size)
        ]
        evaluator.offline_evaluate(all_predictions, all_data)

        # Test with None data
        all_data = None
        evaluator.offline_evaluate(all_predictions, all_data)

        # Different length of data and predictions will raise an error.
        all_data = [dict() for _ in range(9)]
        with self.assertRaisesRegex(
                AssertionError,
                'data_samples and data should have the same length'):
            evaluator.offline_evaluate(all_predictions, all_data)

    @unittest.skipUnless(torch.cuda.is_available(), 'can only run with gpu')
    def test_evaluate_cast_cpu(self):
        cfg = dict(type='ToyMetric')
        evaluator = Evaluator(cfg)

        size = 10

        all_data = [
            dict(
                inputs=torch.zeros((3, 10, 10), device='cuda'),
                data_sample=BaseDataElement(
                    label=torch.ones((1, ), device='cuda')))
            for _ in range(size)
        ]
        all_predictions = [
            BaseDataElement(
                pred=torch.zeros((1, ), device='cuda'),
                label=torch.ones((1, ), device='cuda')) for _ in range(size)
        ]
        for data, pred in zip(all_data, all_predictions):
            evaluator.process([pred], [data])

        def test_results_device(results: List):
            for result in results:
                self.assertEqual(result['pred'].device, torch.device('cpu'))
                self.assertEqual(result['label'].device, torch.device('cpu'))
            return {}

        # replace the `compute_metrics` to the test function
        evaluator.metrics[0].compute_metrics = test_results_device
        evaluator.evaluate(size)
