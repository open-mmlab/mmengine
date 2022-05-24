# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch


def get_max_cuda_memory(device: Optional[torch.device] = None) -> int:
    """Returns the maximum GPU memory occupied by tensors in megabytes (MB) for
    a given device. By default, this returns the peak allocated memory since
    the beginning of this program.

    Args:
        device (torch.device, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~torch.cuda.current_device`, if ``device`` is None.
            Defaults to None.

    Returns:
        int: The maximum GPU memory occupied by tensors in megabytes
        for a given device.
    """
    mem = torch.cuda.max_memory_allocated(device=device)
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)],
                          dtype=torch.int,
                          device=device)
    torch.cuda.reset_peak_memory_stats()
    return int(mem_mb.item())
