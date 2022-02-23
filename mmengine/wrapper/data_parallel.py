# Copyright (c) OpenMMLab. All rights reserved.

from torch.nn.parallel import DataParallel, DistributedDataParallel


class MMDataParallel(DataParallel):
    """There is no difference between MMDataParallel and pytorch's
    DataParallel, "train_step" and "val_step" are added just to avoid bc
    breaking.

    Warning:
        MMDataParallel only supports single GPU training, if you
        need to  train with multiple GPUs, please use MMDistributedDataParallel
        instead. If you have multiple GPUs and you just want to use
        MMDataParallel, you can set the environment variable
        ``CUDA_VISIBLE_DEVICES=0`` or instantiate ``MMDataParallel`` with
        ``device_ids=[0]``.
    """

    def train_step(self, *inputs, **kwargs):
        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')
        return self.forward(*inputs, **kwargs)

    def val_step(self, *inputs, **kwargs):
        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')
        return self.forward(*inputs, **kwargs)


class MMDistributedDataParallel(DistributedDataParallel):
    """There is no difference between MMDistributedDataParallel and pytorch's
    DistributedDataParallel, "train_step" and "val_step" are added just to
    avoid bc breaking."""

    def train_step(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def val_step(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)
