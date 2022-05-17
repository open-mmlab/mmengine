# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union, Dict

import torch
import torch.nn as nn

from mmengine.optim import OptimizerWrapper
from mmengine.data import InstanceData
from ..base_model import BaseModel


class ModelWrapper(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, module: BaseModel):
        super().__init__()
        self.module = module.cuda()

    def forward(self,
                data: List[dict],
                mode: str,
                optimizer: OptimizerWrapper = None,
                return_val_loss: bool = False
                ) -> Union[Dict[str, torch.Tensor], List[InstanceData]]:
        assert mode in ('train', 'val', 'test')
        if isinstance(self.module, BaseModel):
            if mode == 'train':
                return self.module.train_step(data, optimizer)
            elif mode == 'val':
                return self.module.val_step(data, return_loss=return_val_loss)
            elif mode == 'test':
                return self.module.test_step(data)
