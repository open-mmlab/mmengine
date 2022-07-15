# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterator, List, Optional, Sequence, Union

from mmengine.data import BaseDataElement, pseudo_collate
from ..registry.root import METRICS
from .metric import BaseMetric


class Evaluator:
    """Wrapper class to compose multiple :class:`BaseMetric` instances.

    Args:
        metrics (dict or BaseMetric or Sequence): The config of metrics.
    """

    def __init__(self, metrics: Union[dict, BaseMetric, Sequence]):
        self._dataset_meta: Optional[dict] = None
        if not isinstance(metrics, Sequence):
            metrics = [metrics]
        self.metrics: List[BaseMetric] = []
        for metric in metrics:
            if isinstance(metric, BaseMetric):
                self.metrics.append(metric)
            elif isinstance(metric, dict):
                self.metrics.append(METRICS.build(metric))
            else:
                raise TypeError('metric should be a dict or a BaseMetric, '
                                f'but got {metric}.')

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the evaluator and it's metrics."""
        self._dataset_meta = dataset_meta
        for metric in self.metrics:
            metric.dataset_meta = dataset_meta

    def process(self,
                predictions: Sequence[BaseDataElement],
                data_batch: Optional[Any] = None):
        """Convert ``BaseDataSample`` to dict and invoke process method of each
        metric.

        Args:
            predictions (Sequence[BaseDataElement]): A batch of outputs from
                the model.
            data_batch (Any, optional): A batch of data from the dataloader.
        """
        _predictions = []
        for pred in predictions:
            if isinstance(pred, BaseDataElement):
                _predictions.append(pred.to_dict())
            else:
                _predictions.append(pred)

        for metric in self.metrics:
            metric.process(_predictions, data_batch)

    def evaluate(self, size: int) -> dict:
        """Invoke ``evaluate`` method of each metric and collect the metrics
        dictionary.

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
                        'There are multiple evaluation results with the same '
                        f'metric name {name}. Please make sure all metrics '
                        'have different prefixes.')

            metrics.update(_results)
        return metrics

    def offline_evaluate(self,
                         predictions: Sequence,
                         data: Optional[Sequence] = None,
                         chunk_size: int = 1):
        """Offline evaluate the dumped predictions on the given data .

        Args:
            predictions (Sequence): All predictions of the model on the
                validation set.
            data (Sequence, optional): All data of the validation set.
            chunk_size (int): The number of data samples and predictions to be
                processed in a batch.
        """

        # support chunking iterable objects
        if data is not None:
            assert len(predictions) == len(data), (
                'predictions and data should have the same length, but got '
                f'predictions length: {len(predictions)} '
                f'data length: {len(data)}')

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
        for pred_chunk in get_chunks(iter(predictions), chunk_size):
            if data:
                data_chunk = pseudo_collate(data[size:size + chunk_size])
            else:
                data_chunk = None
            size += len(pred_chunk)
            self.process(pred_chunk, data_chunk)
        return self.evaluate(size)
