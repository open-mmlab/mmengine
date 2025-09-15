# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch.distributed.rpc import is_available

from mmengine.dist import is_main_process
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

# Handle PyTorch 2.6+ compatibility issues with distributed optimizers


def _safe_import_zero_optimizer():
    """Safely import ZeroRedundancyOptimizer to avoid JIT compilation issues.

    Starting from PyTorch 2.6.0, JIT compilation issues can occur when
    importing torch.distributed.optim. This function provides a safe import
    mechanism with fallback options.
    """
    try:
        # PyTorch 2.6+ introduced changes that can cause JIT compilation issues
        # when importing torch.distributed.optim. Apply safe import for 2.6+
        if digit_version(TORCH_VERSION) >= digit_version('2.6.0'):
            import os
            import torch

            # Strategy: Use dynamic import with JIT disabled
            # Save original state
            old_jit_enabled = os.environ.get('PYTORCH_JIT', '1')
            old_jit_disable = os.environ.get('PYTORCH_JIT_DISABLE', '0')

            # Disable JIT compilation
            os.environ['PYTORCH_JIT'] = '0'
            os.environ['PYTORCH_JIT_DISABLE'] = '1'

            try:
                # Try to disable JIT via torch.jit if available
                if hasattr(torch.jit, 'set_enabled'):
                    old_jit_torch_enabled = torch.jit.is_enabled()
                    torch.jit.set_enabled(False)
                else:
                    old_jit_torch_enabled = None

                try:
                    # Import with JIT disabled
                    from torch.distributed.optim import \
                        ZeroRedundancyOptimizer as _ZeroRedundancyOptimizer
                    return _ZeroRedundancyOptimizer
                finally:
                    # Restore torch.jit state
                    if (old_jit_torch_enabled is not None and
                            hasattr(torch.jit, 'set_enabled')):
                        torch.jit.set_enabled(old_jit_torch_enabled)
            finally:
                # Restore environment variables
                os.environ['PYTORCH_JIT'] = old_jit_enabled
                os.environ['PYTORCH_JIT_DISABLE'] = old_jit_disable
        else:
            from torch.distributed.optim import \
                ZeroRedundancyOptimizer as _ZeroRedundancyOptimizer
            return _ZeroRedundancyOptimizer
    except (ImportError, RuntimeError, AttributeError) as e:
        # If import fails due to JIT compilation or other issues, return object
        import warnings
        warnings.warn(
            f"Failed to import ZeroRedundancyOptimizer from "
            f"torch.distributed.optim. This is likely due to PyTorch "
            f"version compatibility issues. ZeroRedundancyOptimizer will "
            f"not be available. Error: {e}",
            UserWarning
        )
        return object


_ZeroRedundancyOptimizer = _safe_import_zero_optimizer()

from .builder import OPTIMIZERS  # noqa: E402


@OPTIMIZERS.register_module()
class ZeroRedundancyOptimizer(_ZeroRedundancyOptimizer):
    """A wrapper class of :class:`ZeroRedundancyOptimizer` that gets a
    optimizer type as string.

    This class wraps an arbitrary :class:`torch.optim.Optimizer` and shards its
    states across ranks in the group as described by ZeRO_. The local optimizer
    instance in each rank is only responsible for updating approximately
    ``1 / world_size`` parameters and hence only needs to keep
    ``1 / world_size`` optimizer states. After parameters are updated locally,
    each rank will broadcast its parameters to all other peers to keep all
    model replicas in the same state. ``ZeroRedundancyOptimizer`` can be used
    in conjunction with :class:`torch.nn.parallel.DistributedDataParallel` to
    reduce per-rank peak memory consumption.

    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.

    Warnings:
        ``ZeroRedundancyOptimizer`` requires PyTorch >= 1.8.

    Warnings:
        ``ZeroRedundancyOptimizer`` requires PyTorch >= 1.12 to enable param
        groups.

    Args:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_type (str): the string of the local optimizer class.

    .. _ZeRO: https://arxiv.org/abs/1910.02054
    """

    def __init__(self, params, optimizer_type: str, **kwargs):
        assert digit_version(TORCH_VERSION) >= digit_version('1.8.0'), (
            '`torch.distributed.optim.ZeroReundancyOptimizer` is only '
            'available when pytorch version >= 1.8.')
        assert is_available(), 'torch.distributed.rpc is not available.'

        # Check if ZeroRedundancyOptimizer is actually available
        if _ZeroRedundancyOptimizer is object:
            raise ImportError(
                'ZeroRedundancyOptimizer is not available. This might be '
                'due to PyTorch version compatibility issues. Please check '
                'if your PyTorch version is compatible with MMEngine.'
            )
        # Avoid the generator becoming empty after the following check
        params = list(params)
        assert (
            all(isinstance(p, torch.Tensor) for p in params)
            or digit_version(TORCH_VERSION) >= digit_version('1.12.0')), (
                'PyTorch ZeroRedundancyOptimizer started to support param '
                'groups since 1.12.0. Please update your pytorch version to '
                'enable this feature, or disable param groups by deleting '
                '`paramwise_cfg` filed in config file.')
        optimizer_class = getattr(torch.optim, optimizer_type)
        # TODO: Register a DDP communication hook for `overlap_with_ddp=True`.
        # Currently only `overlap_with_ddp=False` is supported. For more
        # details, please refer to the pytorch's official documentation.
        super().__init__(params, optimizer_class, **kwargs)

    def state_dict(self):
        """Consolidate `state_dict`s from ranks to save the `state_dict`."""
        self.consolidate_state_dict()
        state_dict = super().state_dict() if is_main_process() else dict()
        return state_dict
