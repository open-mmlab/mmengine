# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import warnings
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist

from mmengine.data import BaseDataSample
from mmengine.utils import mkdir_or_exist


class BaseEvaluator(metaclass=ABCMeta):
    """Base class for an evaluator.

    The evaluator first processes each batch of data_samples and
    predictions, and appends the processed results in to the results list.
    Then it collects all results together from all ranks if distributed
    training is used. Finally, it computes the metrics of the entire dataset.

    Args:
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def __init__(self, collect_device: str = 'cpu') -> None:
        self._dataset_meta: Union[None, dict] = None
        self.collect_device = collect_device
        self.results: List[Any] = []

        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

    @property
    def dataset_meta(self) -> Optional[dict]:
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        self._dataset_meta = dataset_meta

    @abstractmethod
    def process(self, data_samples: BaseDataSample, predictions: dict) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_samples (BaseDataSample): The data samples from the dataset.
            predictions (dict): The output of the model.
        """

    @abstractmethod
    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data base on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self._results`. Please '
                'ensure that the processed results are properly added into '
                '`self._results` in `process` method.')

        if self.world_size == 1:
            # non-distributed
            results = self.results
        else:
            results = collect_results(self.results, size, self.collect_device)

        if self.rank == 0:
            # TODO: replace with mmengine.dist.master_only
            metrics = [self.compute_metrics(results)]
        else:
            metrics = [None]  # type: ignore
        # TODO: replace with mmengine.dist.broadcast
        if self.world_size > 1:
            metrics = dist.broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        return metrics[0]


# TODO: replace with mmengine.dist.get_dist_info
def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


# TODO: replace with mmengine.dist.collect_results
def collect_results(results, size, device='cpu'):
    """Collected results in distributed environments."""
    # TODO: replace with mmengine.dist.collect_results
    if device == 'gpu':
        return collect_results_gpu(results, size)
    elif device == 'cpu':
        return collect_results_cpu(results, size)
    else:
        NotImplementedError(f"device must be 'cpu' or 'gpu', but got {device}")


# TODO: replace with mmengine.dist.collect_results
def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:
        pickle.dump(result_part, f, protocol=2)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            with open(osp.join(tmpdir, f'part_{i}.pkl'), 'wb') as f:
                part_list.append(pickle.load(f))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


# TODO: replace with mmengine.dist.collect_results
def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
