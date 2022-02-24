# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional
from unittest import TestCase

import numpy as np

from mmengine.evaluator import BaseEvaluator, ComposedEvaluator
from mmengine.registry import EVALUATORS


@EVALUATORS.register_module()
class ToyEvaluator(BaseEvaluator):

    def __init__(self,
                 collect_device: str = 'cpu',
                 dummy_metrics: Optional[Dict] = None):
        super().__init__(collect_device=collect_device)
        self.dummy_metrics = dummy_metrics

    def process(self, data_samples, predictions):
        result = {'pred': predictions['pred'], 'label': data_samples['label']}
        self.results.append(result)

    def compute_metrics(self, results: List):
        if self.dummy_metrics is not None:
            assert isinstance(self.dummy_metrics, dict)
            return self.dummy_metrics.copy()

        pred = np.concatenate([result['pred'] for result in results])
        label = np.concatenate([result['label'] for result in results])
        acc = (pred == label).sum() / pred.size

        metrics = {
            'accuracy': acc,
            'size': pred.size,  # To check the number of testing samples
        }

        return metrics


def generate_test_results(size, batch_size, pred, label):
    num_batch = np.ceil(size / batch_size, dtype=np.int64)
    bs_residual = size - (num_batch - 1) * batch_size
    for i in range(size):
        bs = bs_residual if i == size - 1 else batch_size
        data_samples = {'label': np.full(bs, label)}
        predictions = {'pred': np.full(bs, pred)}
        yield (data_samples, predictions)


class TestBaseEvaluator(TestCase):

    def build_evaluator(self, cfg):
        if isinstance(cfg, (list, dict)):
            evaluators = [EVALUATORS.build(_cfg) for _cfg in cfg]
            return ComposedEvaluator(evaluators=evaluators)
        else:
            return EVALUATORS.build(cfg)

    def test_single_evaluator(self):
        cfg = dict(type='ToyEvaluator')
        evaluator = self.build_evaluator(cfg)

        size = 10
        batch_size = 4

        for data_samples, predictions in generate_test_results(
                size, batch_size, pred=1, label=1):
            evaluator.process(data_samples, predictions)

        metrics = evaluator.evaluate(size=size)
        self.assertAlmostEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['size'], size)

    def test_composed_evaluator(self):
        cfg = [
            dict(type='ToyEvaluator'),
            dict(type='ToyEvaluator', dummy_metrics=dict(mAP=0.0))
        ]

        evaluator = self.build_evaluator(cfg)

        size = 10
        batch_size = 4

        for data_samples, predictions in generate_test_results(
                size, batch_size, pred=1, label=1):
            evaluator.process(data_samples, predictions)

        metrics = evaluator.evaluate(size=size)

        self.assertAlmostEqual(metrics['accuracy'], 1.0)
        self.assertAlmostEqual(metrics['mAP'], 0.0)
        self.assertEqual(metrics['size'], size)

    def test_ambiguate_metric(self):

        cfg = [
            dict(type='ToyEvaluator', dummy_metrics=dict(mAP=0.0)),
            dict(type='ToyEvaluator', dummy_metrics=dict(mAP=0.0))
        ]

        evaluator = self.build_evaluator(cfg)

        size = 10
        batch_size = 4

        for data_samples, predictions in generate_test_results(
                size, batch_size, pred=1, label=1):
            evaluator.process(data_samples, predictions)

        with self.assertRaisesRegex(
                ValueError,
                'There are multiple evaluators with the same metric name'):
            _ = evaluator.evaluate(size=size)

    def test_dataset_meta(self):
        dataset_meta = dict(classes=('cat', 'dog'))

        cfg = [
            dict(type='ToyEvaluator'),
            dict(type='ToyEvaluator', dummy_metrics=dict(mAP=0.0))
        ]

        evaluator = self.build_evaluator(cfg)
        evaluator.dataset_meta = dataset_meta

        for _evaluator in evaluator.evaluators():
            self.assertDictEqual(_evaluator.dataset_meta, dataset_meta)
