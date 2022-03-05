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

    def before_epoch(self, runner: object) -> None:
        """Set the seed for sampler and batch_sampler.

        Args:
            runner (object): The runner of the training process.
        """
        if hasattr(runner.data_loader.sampler, 'set_epoch'):  # type: ignore
            # in case the data loader uses `SequentialSampler` in Pytorch
            runner.data_loader.sampler.set_epoch(runner.epoch)  # type: ignore
        elif hasattr(
                runner.data_loader.batch_sampler.sampler,  # type: ignore
                'set_epoch'):
            # batch sampler in pytorch warps the sampler as its attributes.
            runner.data_loader.batch_sampler.sampler.set_epoch(  # type: ignore
                runner.epoch)  # type: ignore
