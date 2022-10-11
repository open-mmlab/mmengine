# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict, List, Union

import torch
import torch.nn as nn

from mmengine import MODELS
from mmengine.structures import BaseDataElement

# multi-batch inputs processed by different augmentations from the same batch.
EnhancedBatchInputs = List[Union[torch.Tensor, List[torch.Tensor]]]
# multi-batch data samples processed by different augmentations from the same
# batch. The inner list stands for different augmentations and the outer list
# stands for batch.
EnhancedBatchDataSamples = List[List[BaseDataElement]]
DATA_BATCH = Union[Dict[str, Union[EnhancedBatchInputs,
                                   EnhancedBatchDataSamples]], tuple, dict]
MergedDataSamples = List[BaseDataElement]


@MODELS.register_module()
class BaseTTAModel(nn.Module):
    """Base model for inference with test-time augmentation.

    ``BaseTTAModel`` is a wrapper for inference given multi-batch data.
    It implements the :meth:`test_step` for multi-batch data inference.
    ``multi-batch`` data means data processed by different augmentation
    from the same batch.

    During test time augmentation, the data processed by
    :obj:`mmcv.transforms.TestTimeAug`, and then collated by
    ``pseudo_collate`` will have the following format:

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

    ``image{i}_aug{j}`` means the i-th image of the batch, which is
    augmented by the j-th augmentation.

    ``BaseTTAModel`` will collate the data to:

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
    merged by :meth:`merge_preds`.

    Note:
        :meth:`merge_preds` is an abstract method, all subclasses should
        implement it.

    Args:
        module (dict or nn.Module): Tested model.
    """

    def __init__(self, module: Union[dict, nn.Module]):
        super().__init__()
        if isinstance(module, nn.Module):
            self.module = module
        elif isinstance(module, dict):
            self.module = MODELS.build(module)
        else:
            raise TypeError('The type of module should be a `nn.Module` '
                            f'instance or a dict, but got {module}')
        self.data_preprocessor = lambda data: data
        assert hasattr(self.module, 'test_step'), (
            'Model wrapped by BaseTTAModel must implement `test_step`!')

    @abstractmethod
    def merge_preds(self, data_samples_list: EnhancedBatchDataSamples) \
            -> MergedDataSamples:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (EnhancedBatchDataSamples): List of predictions
                of all enhanced data.

        Returns:
            List[BaseDataElement]: Merged prediction.
        """

    def forward(self, inputs, datasamples) -> MergedDataSamples:
        """Get predictions of each enhanced data, a multiple predictions.

        Args:
            data (DataBatch): Enhanced data batch sampled from dataloader.

        Returns:
            MergedDataSamples: Merged prediction.
        """
        predictions = []
        for input, datasample in zip(datasamples, inputs):  # type: ignore
            predictions.append(
                self.module.test_step(
                    dict(inputs=input, datasamples=datasample)))
        super
        return self.merge_preds(list(zip(*predictions)))  # type: ignore
