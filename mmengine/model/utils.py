# Copyright (c) OpenMMLab. All rights reserved.
import logging
import warnings
from typing import List, Union

import torch
import torch.nn.functional as F


def stack_batch(tensor_list: List[torch.Tensor],
                pad_size_divisor: int = 1,
                pad_value: Union[int, float] = 0) -> torch.Tensor:
    """Stack multiple tensors to form a batch and pad the tensor to the max
    shape use the right bottom padding mode in these images. If
    ``pad_size_divisor > 0``, add padding to ensure the shape of each dim is
    divisible by ``pad_size_divisor``.

    Args:
        tensor_list (List[Tensor]): A list of tensors with the same dim.
        pad_size_divisor (int): If ``pad_size_divisor > 0``, add padding
            to ensure the shape of each dim is divisible by
            ``pad_size_divisor``. This depends on the model, and many
            models need to be divisible by 32. Defaults to 1
        pad_value (int, float): The padding value. Defaults to 0.

    Returns:
       Tensor: The n dim tensor.
    """
    assert isinstance(
        tensor_list,
        list), (f'Expected input type to be list, but got {type(tensor_list)}')
    assert tensor_list, '`tensor_list` could not be an empty list'
    assert len({
        tensor.ndim
        for tensor in tensor_list
    }) == 1, (f'Expected the dimensions of all tensors must be the same, '
              f'but got {[tensor.ndim for tensor in tensor_list]}')

    dim = tensor_list[0].dim()
    num_img = len(tensor_list)
    all_sizes: torch.Tensor = torch.Tensor(
        [tensor.shape for tensor in tensor_list])
    max_sizes = torch.ceil(
        torch.max(all_sizes, dim=0)[0] / pad_size_divisor) * pad_size_divisor
    padded_sizes = max_sizes - all_sizes
    # The first dim normally means channel,  which should not be padded.
    padded_sizes[:, 0] = 0
    if padded_sizes.sum() == 0:
        return torch.stack(tensor_list)
    # `pad` is the second arguments of `F.pad`. If pad is (1, 2, 3, 4),
    # it means that padding the last dim with 1(left) 2(right), padding the
    # penultimate dim to 3(top) 4(bottom). The order of `pad` is opposite of
    # the `padded_sizes`. Therefore, the `padded_sizes` needs to be reversed,
    # and only odd index of pad should be assigned to keep padding "right" and
    # "bottom".
    pad = torch.zeros(num_img, 2 * dim, dtype=torch.int)
    pad[:, 1::2] = padded_sizes[:, range(dim - 1, -1, -1)]
    batch_tensor = []
    for idx, tensor in enumerate(tensor_list):
        batch_tensor.append(
            F.pad(tensor, tuple(pad[idx].tolist()), value=pad_value))
    return torch.stack(batch_tensor)


def detect_anomalous_params(loss: torch.Tensor, model) -> None:
    parameters_in_graph = set()
    visited = set()

    def traverse(grad_fn):
        if grad_fn is None:
            return
        if grad_fn not in visited:
            visited.add(grad_fn)
            if hasattr(grad_fn, 'variable'):
                parameters_in_graph.add(grad_fn.variable)
            parents = grad_fn.next_functions
            if parents is not None:
                for parent in parents:
                    grad_fn = parent[0]
                    traverse(grad_fn)

    traverse(loss.grad_fn)
    from mmengine.logging import MMLogger
    logger = MMLogger.get_current_instance()
    for n, p in model.named_parameters():
        if p not in parameters_in_graph and p.requires_grad:
            logger.log(
                level=logging.ERROR,
                msg=f'{n} with shape {p.size()} is not '
                f'in the computational graph \n')


def merge_dict(*args):
    """Merge all dictionaries into one dictionary.

    If pytorch version >= 1.8, ``merge_dict`` will be wrapped
    by ``torch.fx.wrap``,  which will make ``torch.fx.symbolic_trace`` skip
    trace ``merge_dict``.

    Note:
        If a function needs to be traced by ``torch.fx.symbolic_trace``,
        but inevitably needs to use ``update`` method of ``dict``(``update``
        is not traceable). It should use ``merge_dict`` to replace
        ``xxx.update``.

    Args:
        *args: dictionary needs to be merged.

    Returns:
        dict: Merged dict from args
    """
    output = dict()
    for item in args:
        assert isinstance(
            item,
            dict), (f'all arguments of merge_dict should be a dict, but got '
                    f'{type(item)}')
        output.update(item)
    return output


# torch.fx is only available when pytorch version >= 1.8.
# If the subclass of `BaseModel` has multiple submodules, and each module
# will return a loss dict during training process, i.e., `TwoStageDetector`
# in mmdet. It should use `merge_dict` to get the total loss, rather than
# `loss.update` to keep model traceable.
try:
    import torch.fx

    # make torch.fx skip trace `merge_dict`.
    merge_dict = torch.fx.wrap(merge_dict)

except ImportError:
    warnings.warn('Cannot import torch.fx, `merge_dict` is a simple function '
                  'to merge multiple dicts')
