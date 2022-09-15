# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from mmengine import MODELS
from ..base_model import BaseDataPreprocessor, BaseModel


@MODELS.register_module()
class TestTimeAugDataPreprocessor(BaseDataPreprocessor):
    def forward(self,
                data: dict,
                training: bool = False) -> Union[dict, list]:
        data = self.cast_data(data)
        first_inputs = data[0]['inputs']
        num_augs = len(first_inputs)
        num_imgs = len(data)
        output = []
        for aug_idx in range(num_augs):
            _single_batch = []
            for batch_idx in range(num_imgs):
                _data = dict()
                for key in data[0].keys():
                    _data[key] = data[batch_idx][key][aug_idx]
                _single_batch.append(_data)
            output.append(_single_batch)
        for data in data:
            data['data_samples'] = data['data_samples'][0]
        return output


@MODELS.register_module()
class BaseTestTimeAugModelWrapper(BaseModel):

    def __init__(self,
                 model: BaseModel,
                 data_preprocessor: Optional[Union[dict, nn.Module]]=None):
        if data_preprocessor is None:
            data_preprocessor = dict(type='TestTimeAugDataPreprocessor')
        super(BaseTestTimeAugModelWrapper, self).__init__(data_preprocessor)
        self.model = model

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
