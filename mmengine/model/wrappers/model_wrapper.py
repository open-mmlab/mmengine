# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from mmengine.data import InstanceData
from mmengine.optim import OptimizerWrapper


class ModelWrapper(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module.cuda()

    def forward(
        self,
        data: List[dict],
        mode: str,
        optimizer_wrapper: Optional[OptimizerWrapper] = None,
        return_val_loss: bool = False
    ) -> Union[Dict[str, torch.Tensor], List[InstanceData]]:
        assert mode in ('train', 'val', 'test')
        if mode == 'train':
            with optimizer_wrapper.\
                    gradient_accumulative_context():  # type: ignore
                return self.module.train_step(data, optimizer_wrapper)
        elif mode == 'val':
            return self.module.val_step(data, return_loss=return_val_loss)
        else:
            return self.module.test_step(data)
