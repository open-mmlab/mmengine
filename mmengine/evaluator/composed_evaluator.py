# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from .base import BaseEvaluator


class ComposedEvaluator(BaseEvaluator):
    """Wrapper class to compose multiple :class:`DatasetEvaluator` instances.

    Args:
        evaluators (list[BaseEvaluator]): The evaluators to compose.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def __init__(self, evaluators: List[BaseEvaluator], collect_device='cpu'):
        super().__init__(collect_device)
        self.evaluators = evaluators.copy()

    @property
    def dataset_meta(self) -> dict:
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        self._dataset_meta = dataset_meta
        for evaluator in self.evaluators:
            evaluator.dataset_meta = dataset_meta

    def process(self, data_samples: dict, predictions: dict):
        """Invoke process method of each wrapped evaluator.

        Args:
            data_samples (dict): The data samples from the dataset.
            predictions (dict): The output of the model.
        """

        for evalutor in self.evaluators:
            evalutor.process(data_samples, predictions)

    def evaluate(self, size: int) -> dict:
        """Invoke evaluate method of each wrapped evaluator and collect the
        metrics dict.

        Args:
            size (int): Length of the entire val dataset. When batch size > 1,
                the dataloader may pad some samples to make sure all ranks
                have the same length of dataset slice. The ``collect_results``
                function will drop the padded data base on this size.

        Returns:
            metrics (dict): Evaluation metrics of all wrapped evaluators. The
                keys are the names of the metrics, and the values are
                corresponding results.
        """
        metrics = {}
        for evaluator in self.evaluators:
            _metric = evaluator.evaluate(size)

            # Check metric name conflicts
            for name in _metric.keys():
                if name in _metric:
                    raise ValueError(
                        'There are multiple evaluators with the same metric '
                        f'name {name}')

            metrics.update(_metric)
        return metrics
