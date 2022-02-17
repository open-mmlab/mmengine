import warnings
import copy
import bisect
from typing import List

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from .base_dataset import full_init_before_called, BaseDataset


class ConcatDataset(_ConcatDataset):

    def __init__(self,
                 datasets: List[BaseDataset],
                 lazy_init: bool = False):
        """A wrapper of concatenated dataset.

        Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
        concat the group flag for image aspect ratio.

        Args:
            datasets (list[mmengine.BaseDataset]): A list of datasets.
            lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False.
        """
        # Only use meta of first dataset.
        self._meta = datasets[0].meta
        self.datasets = datasets
        for i, dataset in enumerate(datasets):
            if self._meta != dataset.meta:
                warnings.warn(
                    f"The meta data of {i + 1}-th dataset is not same as meta "
                    f"data of 1-st dataset.")

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    @property
    def meta(self) -> dict:
        """Get the meta information of the first dataset in ConcatDataset.
        Returns:
            dict: meta.
        """

        return copy.deepcopy(self._meta)

    def full_init(self):
        """Loop to full_init each dataset"""
        if self._fully_initialized:
            return
        for d in self.datasets:
            d.full_init()
        # explain the reason.
        super(ConcatDataset, self).__init__(self.datasets)
        self._fully_initialized = True

    @full_init_before_called
    def _get_ori_dataset_idx(self, idx: int):
        """Get dataset index and innder index of dataset.
        Args:
            idx: total index

        Returns:

        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    'absolute value of index should not exceed dataset length')
            idx = len(self) + idx
        # Get the inner index of single dataset
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

    @full_init_before_called
    def get_data_info(self, idx):
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_data_info(sample_idx)

    @full_init_before_called
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        if not self._fully_initialized:
            warnings.warn(
                'Please call self.full_init() manually to accrelate '
                'the speed.')
            self.full_init()

        return super().__getitem__(idx)
