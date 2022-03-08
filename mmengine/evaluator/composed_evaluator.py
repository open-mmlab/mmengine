# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

from mmengine.data import BaseDataSample
from .base import BaseEvaluator


class ComposedEvaluator:
    """Wrapper class to compose multiple :class:`DatasetEvaluator` instances.

    Args:
        evaluators (Sequence[BaseEvaluator]): The evaluators to compose.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def __init__(self,
                 evaluators: Sequence[BaseEvaluator],
                 collect_device='cpu'):
        self._dataset_meta: Union[None, dict] = None
        self.collect_device = collect_device
        self.evaluators = evaluators

    @property
    def dataset_meta(self) -> Optional[dict]:
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        self._dataset_meta = dataset_meta
        for evaluator in self.evaluators:
            evaluator.dataset_meta = dataset_meta

    def process(self, data_samples: BaseDataSample, predictions: dict):
        """Invoke process method of each wrapped evaluator.

        Args:
            data_samples (BaseDataSample): The data samples from the dataset.
            predictions (dict): The output of the model.
        """

        for evalutor in self.evaluators:
            evalutor.process(data_samples, predictions)

    def evaluate(self, size: int) -> dict:
        """Invoke evaluate method of each wrapped evaluator and collect the
        metrics dict.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data base on
                this size.

        Returns:
            dict: Evaluation metrics of all wrapped evaluators. The keys are
            the names of the metrics, and the values are corresponding results.
        """
        metrics = {}
        for evaluator in self.evaluators:
            _metrics = evaluator.evaluate(size)

            # Check metric name conflicts
            for name in _metrics.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluators with the same metric '
                        f'name {name}')

            metrics.update(_metrics)
        return metrics
