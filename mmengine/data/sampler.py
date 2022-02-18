# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

from mmengine.dist import get_dist_info, sync_random_seed
from mmengine.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DefaultSampler(Sampler[int]):
    """The default data sampler for both distributed and non-distributed
    environment.

    It has several differences from the PyTorch ``DistributedSampler`` as
    below:

    1. This sampler supports non-distributed environment.

    2. The round up behaviors are a little different.

       - If ``round_up=True``, this sampler will add extra samples to make the
         number of samples is evenly divisible by the world size. And
         this behavior is the same as the ``DistributedSampler`` with
         ``drop_last=False``.
       - If ``round_up=False``, this sampler won't remove or add any samples
         while the ``DistributedSampler`` with ``drop_last=True`` will remove
         tail samples.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        rank, num_replicas = get_dist_info()
        self.rank = rank
        self.num_replicas = num_replicas

        self.dataset = dataset
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
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
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed. If None, set a random seed.
            Defaults to None.
    """  # noqa: W605

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None) -> None:
        rank, num_replicas = get_dist_info()
        self.rank = rank
        self.num_replicas = num_replicas

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
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
