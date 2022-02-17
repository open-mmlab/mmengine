# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DefaultSampler(Sampler[int]):
    """The default data sampler for both dist and non-dist environment.

    Modified from the ``DistributedSampler`` in PyTorch with some differences:

    1. This sampler supports non-dist environment.

    2. The round up behaviors are a little different. If ``round_up=True``,
       the behavior of this sampler is the same as the ``DistributedSampler``
       with ``drop_last=False``. But if ``round_up=False``, this sampler won't
       remove or add any samples while the ``DistributedSampler`` with
       ``drop_last=True`` will remove tail samples.

    Args:
        dataset (Sized): The dataset.
        num_replicas (int, optional): Number of processes participating in
            distributed training. Defaults to None.
        rank (int, optional): Rank of current process. Default: None.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, use 0. Defaults to 0.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the ``num_replicas``. Defaults to True.
    """

    def __init__(self,
                 dataset: Sized,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: Optional[int] = 0,
                 round_up: bool = True) -> None:
        _rank, _num_replicas = get_dist_info()
        rank = _rank if rank is None else rank
        num_replicas = _num_replicas if num_replicas is None else num_replicas
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f'Invalid rank {rank}, rank should be in the interval'
                ' [0, {num_replicas - 1}]')

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed if seed is not None else 0
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / num_replicas)
            self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


@DATA_SAMPLERS.register_module()
class InfiniteSampler(Sampler[int]):
    """It's designed for iteration-based runner and yields a mini-batch indices
    each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (Sized): The dataset.
        num_replicas (int, optional): Number of processes participating in
            distributed training. Defaults to None.
        rank (int, optional): Rank of current process. Defaults to None.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, use 0. Defaults to 0.
    """  # noqa: W605

    def __init__(self,
                 dataset: Sized,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: Optional[int] = 0) -> None:
        _rank, _num_replicas = get_dist_info()
        rank = _rank if rank is None else rank
        num_replicas = _num_replicas if num_replicas is None else num_replicas
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f'Invalid rank {rank}, rank should be in the interval'
                ' [0, {num_replicas - 1}]')

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed if seed is not None else 0
        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self) -> Iterator[int]:
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self) -> Iterator[int]:
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.num_replicas)

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        for idx in self.indices:
            yield idx

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Not supported in iteration-based runner."""
        raise NotImplementedError(
            'The `InfiniteSampler` is only used in iteration-based runner, '
            "and doesn't need `set_epoch`")
