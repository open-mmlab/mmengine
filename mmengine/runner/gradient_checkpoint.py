# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn


class CheckpointWrapper(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint_sequential(
            self.module, len(self.module), x)


def set_gredient_checkpoint(model):
    for name, child in model.named_children():
        if isinstance(child, nn.Sequential):
            new_child = CheckpointWrapper(child)
            setattr(model, name, new_child)
    return model
