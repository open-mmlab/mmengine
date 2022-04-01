# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterator, Optional, Sequence, Tuple, Union

from mmengine.data import BaseDataElement
from ..registry.root import METRICS
from .metric import BaseMetric


class Evaluator:
    """Wrapper class to compose multiple :class:`BaseEvaluator` instances.

    Args:
        metrics (dict, BaseMetric, BaseMetric): The config of metrics.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def __init__(self,
                 metrics: Union[dict, BaseMetric, BaseMetric],
                 collect_device='cpu'):
        self._dataset_meta: Union[None, dict] = None
        self.collect_device = collect_device
        if not isinstance(metrics, Sequence):
            metrics = [metrics]  # type: ignore
        self.metrics = []  # type: ignore
        for metric in metrics:  # type: ignore
            if isinstance(metric, BaseMetric):
                self.metrics.append(metric)
            elif isinstance(metric, dict):
                self.metrics.append(METRICS.build(metric))
            else:
                raise TypeError('metric should be a dict or a BaseMetric, '
                                f'but got {metric}.')

    @property
    def dataset_meta(self) -> Optional[dict]:
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        self._dataset_meta = dataset_meta
        for evaluator in self.metrics:
            evaluator.dataset_meta = dataset_meta

    def process(self, data_batch: Sequence[Tuple[Any, BaseDataElement]],
                predictions: Sequence[BaseDataElement]):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            data_batch (Sequence[Tuple[Any, BaseDataElement]]): A batch of data
                from the dataloader.
            predictions (Sequence[BaseDataElement]): A batch of outputs from
                the model.
        """
        data_batch = [
            (input, self._datasample2dict(data))  # type: ignore
            for input, data in data_batch
        ]
        predictions = [
            self._datasample2dict(prediction)  # type: ignore
            for prediction in predictions
        ]

        for metric in self.metrics:
            metric.process(data_batch, predictions)  # type: ignore

    def evaluate(self, size: int) -> dict:
        """Invoke evaluate method of each metric and collect the metrics dict.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation results of all metrics. The keys are the names
            of the metrics, and the values are corresponding results.
        """
        metrics = {}
        for metric in self.metrics:
            _results = metric.evaluate(size)

            # Check metric name conflicts
            for name in _results.keys():
                if name in metrics:
                    raise ValueError(
                        'There are multiple evaluate results with the same '
                        f'metric name {name}')

            metrics.update(_results)
        return metrics

    def offline_evaluate(self,
                         data: Sequence,
                         predictions: Sequence,
                         chunk_size: int = 1):
        """Offline evaluate the dumped predictions on the given data .

        Args:
            data (Sequence): All data of the validation set.
            predictions (Sequence): All predictions of the model on the
                validation set.
            chunk_size (int): The number of data samples and predictions to be
                processed in a batch.
        """

        # support chunking iterable objects
        def get_chunks(seq: Iterator, chunk_size=1):
            stop = False
            while not stop:
                chunk = []
                for _ in range(chunk_size):
                    try:
                        chunk.append(next(seq))
                    except StopIteration:
                        stop = True
                        break
                if chunk:
                    yield chunk

        size = 0
        for data_chunk, pred_chunk in zip(
                get_chunks(iter(data), chunk_size),
                get_chunks(iter(predictions), chunk_size)):
            size += len(data_chunk)
            self.process(data_chunk, pred_chunk)
        return self.evaluate(size)

    def _datasample2dict(self, datasample: Union[dict,
                                                 BaseDataElement]) -> dict:
        """Convert ``BaseDataElement`` to dict."""
        if isinstance(datasample, BaseDataElement):
            return {k: datasample.get(k) for k in datasample.keys()}
        else:
            return datasample
