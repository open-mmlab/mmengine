# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import HOOKS
from .hook import Hook


@HOOKS.register_module()
class DistSamplerSeedHook(Hook):
    """Data-loading sampler for distributed training.

    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """

    priority = 'NORMAL'

    def before_train_epoch(self, runner, mode: str = 'train') -> None:
        """Set the seed for sampler and batch_sampler.

        Args:
            runner (Runner): The runner of the training process.
        """
        if hasattr(runner.train_loop.dataloader.sampler, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.train_loop.dataloader.sampler.set_epoch(runner.epoch)
        elif hasattr(runner.train_loop.dataloader.batch_sampler.sampler,
                     'set_epoch'):
            # batch sampler in pytorch warps the sampler as its attributes.
            runner.train_loop.dataloader.batch_sampler.sampler.set_epoch(
                runner.epoch)
