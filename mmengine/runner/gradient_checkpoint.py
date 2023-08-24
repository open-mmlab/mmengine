# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


class CheckpointWrapper(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        if isinstance(self.module, nn.Module):
            return checkpoint(self.module, x)
        elif isinstance(self.module, nn.Sequential):
            return checkpoint_sequential(self.module, len(self.module), x)
        else:
            return self.module(x)


def turn_on_gredient_checkpoint(model):
    for name, child in model.named_children():
        new_child = CheckpointWrapper(child)
        setattr(model, name, new_child)
