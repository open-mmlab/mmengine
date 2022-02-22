# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import copy
import math
import warnings
from collections import defaultdict
from typing import List, Sequence, Tuple

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset

from .base_dataset import BaseDataset, force_full_init


class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as ``torch.utils.data.dataset.ConcatDataset`` and support lazy_init.

    Args:
        datasets (Sequence[BaseDataset]): A list of datasets which will be
            concatenated.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
    """

    def __init__(self,
                 datasets: Sequence[BaseDataset],
                 lazy_init: bool = False):
        # Only use meta of first dataset.
        self._meta = datasets[0].meta
        self.datasets = datasets  # type: ignore
        for i, dataset in enumerate(datasets, 1):
            if self._meta != dataset.meta:
                warnings.warn(
                    f'The meta information of the {i}-th dataset does not '
                    'match meta information of the first dataset')

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def meta(self) -> dict:
        """Get the meta information of the first dataset in ``self.datasets``.

        Returns:
            dict: Meta information of first dataset.
        """
        # Prevent `self._meta` from being modified by outside.
        return copy.deepcopy(self._meta)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return
        for d in self.datasets:
            d.full_init()
        # Get the cumulative sizes of `self.datasets`. For example, the length
        # of `self.datasets` is [2, 3, 4], the cumulative sizes is [2, 5, 9]
        super().__init__(self.datasets)
        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> Tuple[int, int]:
        """Convert global idx to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            Tuple[int, int]: The index of ``self.datasets`` and the local
            index of data.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    f'absolute value of index({idx}) should not exceed dataset'
                    f'length({len(self)}).')
            idx = len(self) + idx
        # Get the inner index of single dataset
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return dataset_idx, sample_idx

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_data_info(sample_idx)

    @force_full_init
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if not self._fully_initialized:
            warnings.warn('Please call `full_init` method manually to '
                          'accelerate the speed.')
            self.full_init()
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx][sample_idx]


class RepeatDataset:
    """A wrapper of repeated dataset.

    The length of repeated dataset will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatDataset can reduce the data loading time between
    epochs.

    Args:
        dataset (BaseDataset): The dataset to be repeated.
        times (int): Repeat times.
        lazy_init (bool, optional): Whether to load annotation during
            instantiation. Defaults to False.
    """

    def __init__(self,
                 dataset: BaseDataset,
                 times: int,
                 lazy_init: bool = False):
        self.dataset = dataset
        self.times = times
        self._meta = dataset.meta

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def meta(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._meta)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._ori_len = len(self.dataset)
        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx: Global index of ``RepeatDataset``.

        Returns:
            idx (int): Local index of data.
        """
        return idx % self._ori_len

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the datasets.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx):
        if not self._fully_initialized:
            warnings.warn('Please call `full_init` method manually to '
                          'accelerate the speed.')
            self.full_init()

        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset[sample_idx]

    @force_full_init
    def __len__(self):
        return self.times * self._ori_len


class ClassBalancedDataset:
    """A wrapper of class balanced dataset.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :meth:`get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Args:
        dataset (BaseDataset): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False
    """

    def __init__(self,
                 dataset: BaseDataset,
                 oversample_thr: float,
                 lazy_init: bool = False):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self._meta = dataset.meta

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def meta(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._meta)

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()

        repeat_factors = self._get_repeat_factors(self.dataset,
                                                  self.oversample_thr)
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        self._fully_initialized = True

    def _get_repeat_factors(self, dataset: BaseDataset,
                            repeat_thr: float) -> List[float]:
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (BaseDataset): The dataset.
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            List[float]: The repeat factors for each images in the dataset.
        """
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq: defaultdict = defaultdict(float)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            assert v > 0, f'caterogy {k} does not contain any images'
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I and its labels L(I), compute the image-level
        # repeat factor:
        #    r(I) = max_{c in L(I)} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            repeat_factor = max(
                {category_repeat[cat_id]
                 for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return repeat_factors

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            int: Local index of data.
        """
        return self.repeat_indices[idx]

    @force_full_init
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids of class balanced dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_cat_ids(sample_idx)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx):
        warnings.warn('Please call `full_init` method manually to '
                      'accelerate the speed.')
        if not self._fully_initialized:
            self.full_init()

        ori_index = self._get_ori_dataset_idx(idx)
        return self.dataset[ori_index]

    @force_full_init
    def __len__(self):
        return len(self.repeat_indices)
