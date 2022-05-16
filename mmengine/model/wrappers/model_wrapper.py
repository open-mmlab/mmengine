# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

import mmengine.optim.optimizer.optimizer_wrapper as optim_wrapper
from ..base_model import BaseModel


class _ModelWrapper(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, module: BaseModel):
        super().__init__()
        self.module = module.cuda()

    def forward(self,
                data: dict,
                mode: str,
                optimizer: optim_wrapper._OptimizerWrapper = None,
                return_val_loss=False):
        assert mode in ('train', 'val', 'test')
        if isinstance(self.module, BaseModel):
            if mode == 'train':
                return self.module.train_step(data, optimizer)
            elif mode == 'val':
                return self.module.val_step(data, return_loss=return_val_loss)
            elif mode == 'test':
                return self.module.test_step(data)
