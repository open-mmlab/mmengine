# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Optional, Union

import torch

from mmengine import MODELS
from mmengine.optim import OptimWrapper
from mmengine.structures import BaseDataElement
from ..base_model import BaseDataPreprocessor, BaseModel

EnhancedInputs = List[Union[torch.Tensor, List[torch.Tensor]]]
EnhancedDataSamples = List[List[BaseDataElement]]
MergedDataSamples = List[BaseDataElement]


@MODELS.register_module()
class BaseTestTimeAugModel(BaseModel):
    """Base model for test time augmentation.

    ``BaseTestTimeAugModel`` is a wrapper for specific algorithm
    model. It implements the :meth:`forward` for multi-batch
    data inference. `multi-batch` data means different enhanced results for
    the same batch.

    All subclasses should implement :meth:`merge_results` for results fusion.

    During test time augmentation, the data processed by
    :obj:`mmcv.transforms.TestTimeAug`, and then collated by
    `pseudo_collate` will have the following format:

    .. code-block::

        result = dict(
            inputs=[
                [image1_aug1, image2_aug1],
                [image1_aug2, image2_aug2]
            ],
            data_samples=[
                [data_sample1_aug1, data_sample2_aug1],
                [data_sample1_aug2, data_sample2_aug2],
            ]
        )

    ``image{1}_aug{1}`` means the 1st image of the batch, which is
    augmented by the 1st augmentation.

    ``BaseTestTimeAugModel`` will collate the data to:

     .. code-block::

        data1 = dict(
            inputs=[image1_aug1, image2_aug1],
            data_samples=[data_sample1_aug1, data_sample2_aug1]
        )

        data2 = dict(
            inputs=[image1_aug2, image2_aug2],
            data_samples=[data_sample1_aug2, data_sample2_aug2]
        )

    ``data1`` and ``data2`` will be passed to model, and the results will be
    merged by :meth:`merge_results`

    Note:
        :meth:`merge_results` is an abstract method, all subclasses should
        implement it.

    Args:
        model (BaseModel): Tested model.
        data_preprocessor (BaseDataPreprocessor or dict, optional): The
            pre-process config For :class:`BaseDataPreprocessor`.
    """

    def __init__(
            self,
            model: BaseModel,
            data_preprocessor: Optional[Union[dict,
                                              BaseDataPreprocessor]] = None):

        super().__init__(data_preprocessor)
        self.module = model

    @abstractmethod
    def merge_results(self, data_samples_list: EnhancedDataSamples) \
            -> List[BaseDataElement]:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (EnhancedDataSamples): List of predictions of
                all enhanced data.

        Returns:
            List[BaseDataElement]: Merged prediction.
        """

    def forward(self, inputs, data_samples, mode):
        """

        Args:
            inputs (EnhancedInputs): List of enhanced batch data from single
                batch data.
            data_samples (EnhancedDataSamples): List of enhanced data
                samples from single batch data sample.
            mode (str): Current mode of model, see more information in
                :meth:`mmengine.model.BaseModel.forward`.

        Returns:
            MergedDataSamples: Merged prediction.
        """
        data_samples_list = []
        for batch_inputs, batch_data_samples in zip(inputs, data_samples):
            data = self.module.data_preprocessor(
                dict(inputs=batch_inputs, data_samples=batch_data_samples))
            data_samples_list.append(self.module(**data, mode='predict'))
        return self.merge_results(data_samples_list)

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """``BaseTestTimeAugModel`` is only for testing for validation,
        ``train_step`` should not be called."""
        raise NotImplementedError('train_step ')
