# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional, Tuple, Dict
import shutil
import pickle
import numpy as np
import tempfile
import torch
import os.path as osp
from torch import Tensor
from torch import distributed as dist

import mmengine
from .utils import (get_world_size, get_rank, get_backend, get_dist_info,
                    get_default_group)
from mmengine.utils import digit_version, TORCH_VERSION


def _get_reduce_op(name: str) -> dist.ReduceOp:
    op_mappings = {
        'sum': dist.ReduceOp.SUM,
        'product': dist.ReduceOp.PRODUCT,
        'min': dist.ReduceOp.MIN,
        'max': dist.ReduceOp.MAX,
        'band': dist.ReduceOp.BAND,
        'bor': dist.ReduceOp.BOR,
        'bxor': dist.ReduceOp.BXOR,
    }

    if name.lower() not in op_mappings:
        raise ValueError(
            f'reduce op should be one of {op_mappings.keys()}, bug got {name}')

    return op_mappings[name.lower()]


def all_reduce(data: Tensor,
               op: str = 'sum',
               group: Optional[dist.ProcessGroup] = None) -> None:
    """Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``data`` is going to be bitwise identical in all
    processes.

    Note:
        Calling ``all_reduce`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Input and output of the collective. The function
            operates in-place.
        op (str): Operation to reduce data. Defaults to 'sum'. Optional values
            are 'sum', 'mean' and 'produce', 'min', 'max', 'band', 'bor' and
            'bxor'.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> dist.all_reduce(data)
        >>> data
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.all_reduce(data, op=torch.dist.ReduceOp.SUM)
        >>> data
        tensor([4, 6]) # Rank 0
        tensor([4, 6]) # Rank 1
    """
    world_size = get_world_size(group)
    if world_size > 1:
        if group is None:
            group = get_default_group()

        # pytorch does not support 'mean' operation so we fall back to support
        # it with 'sum' operation.
        if op.lower() == 'mean':
            dist.all_reduce(data, _get_reduce_op('sum'), group)
            data.div_(world_size)
        else:
            dist.all_reduce(data, _get_reduce_op(op), group)


def all_gather(data: Tensor,
               group: Optional[dist.ProcessGroup] = None) -> List[Tensor]:
    """Gather data from the whole group in a list.

    Note:
        Calling ``all_gather`` in non-distributed environment does nothing
        and just returns a list containing :attr:`data` itself.

    Note:
        Unlike PyTorch ``torch.distributed.all_gather``, :meth:`all_gather` in
        MMEngine does not pass in an empty list ``gather_list`` and returns
        the ``gather_list`` directly, which is more convenient. The difference
        between their interfaces is as below:

        - MMEngine: all_gather(data, group) -> gather_list
        - PyTorch: all_gather(gather_list, data, group) -> None

    Args:
        data (Tensor): Tensor to be gathered.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: Return a list containing data from the whole group if
        in distributed environment, otherwise a list only containing
        :attr:`data` itself.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> output = dist.all_gather(data)
        >>> output
        [tensor([0, 1])]

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2])  # Rank 0
        tensor([3, 4])  # Rank 1
        >>> output = dist.all_gather(data)
        >>> output
        [tensor([1, 2]), tensor([3, 4])]  # Rank 0
        [tensor([1, 2]), tensor([3, 4])]  # Rank 1
    """
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    gather_list = [torch.empty_like(data) for _ in range(world_size)]
    dist.all_gather(gather_list, data, group)
    return gather_list


def gather(
        data: Tensor,
        dst: int = 0,
        group: Optional[dist.ProcessGroup] = None) -> List[Optional[Tensor]]:
    """Gather data from the whole group to ``dst`` process.

    Note:
        Calling ``gather`` in non-distributed environment dose nothing
        and just returns a list containing :attr:`data` itself.

    Note:
        ``NCCL`` backend does not support ``gather``.

    Note:
        Unlike PyTorch ``torch.distributed.gather``, :meth:`gather` in
        MMEngine does not pass in an empty list ``gather_list`` and returns
        the ``gather_list`` directly, which is more convenient. The difference
        between their interfaces is as below:

        - MMEngine: gather(data, dst, group) -> gather_list
        - PyTorch: gather(data, gather_list, dst, group) -> None

    Args:
        data (Tensor): Tensor to be gathered. CUDA tensor is not supported.
        dst (int): Destination rank. Defaults to 0.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: ``dst`` process will get a list of tensor gathering from
        the whole group. Other process will get a empty list. If in
        non-distributed environment, just return a list containing
        :attr:`data` itself.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> output = dist.gather(data)
        >>> output
        [tensor([0, 1])]

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> output = dist.gather(data)
        >>> output
        [tensor([1, 2]), tensor([3, 4])]  # Rank 0
        []  # Rank 1
    """
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    if get_rank(group) == dst:
        gather_list = [torch.empty_like(data) for _ in range(world_size)]
    else:
        gather_list = []

    dist.gather(data, gather_list, dst, group)
    return gather_list


def broadcast(data: Tensor,
              src: int = 0,
              group: Optional[dist.ProcessGroup] = None) -> None:
    """Broadcast the data from ``src`` process to the whole group.

    ``data`` must have the same number of elements in all processes
    participating in the collective.

    Note:
        Calling ``broadcast`` in non-distributed environment does nothing.

    Args:
        data (Tensor): Data to be sent if ``src`` is the rank of current
            process, and data to be used to save received data otherwise.
        src (int): Source rank. Defaults to 0.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = torch.arange(2, dtype=torch.int64)
        >>> data
        tensor([0, 1])
        >>> dist.broadcast(data)
        >>> data
        tensor([0, 1])

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([3, 4]) # Rank 1
        >>> dist.broadcast(data)
        >>> data
        tensor([1, 2]) # Rank 0
        tensor([1, 2]) # Rank 1
    """
    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        dist.broadcast(data, src, group)


def sync_random_seed(group: Optional[dist.ProcessGroup] = None) -> int:
    """Synchronize a random seed to all processes.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Random seed.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> seed = dist.sync_random_seed()
        >>> seed  # which a random number
        587791752

        >>> distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> seed = dist.sync_random_seed()
        >>> seed
        587791752  # Rank 0
        587791752  # Rank 1
    """
    seed = np.random.randint(2**31)
    if get_world_size(group) == 1:
        return seed

    if group is None:
        group = get_default_group()

    group_backend = get_backend(group)
    is_nccl_backend = group_backend == dist.Backend.NCCL
    current_device = torch.device('cpu')
    if is_nccl_backend:
        current_device = torch.device('cuda', torch.cuda.current_device())

    if get_rank(group) == 0:
        random_num = torch.tensor(seed, dtype=torch.int32).to(current_device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32).to(current_device)

    dist.broadcast(random_num, src=0, group=group)

    return random_num.item()


def _object_to_tensor(obj: Any) -> Tuple[Tensor, Tensor]:
    """Serialize picklable python object to tensor."""
    byte_storage = torch.ByteStorage.from_buffer(pickle.dumps(obj))
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor
    # and specifying dtype. Otherwise, it will cause 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage)
    local_size = torch.LongTensor([byte_tensor.numel()])
    return byte_tensor, local_size


def _tensor_to_object(tensor: Tensor, tensor_size: int) -> Any:
    """Deserialize tensor to picklable python object."""
    buf = tensor.cpu().numpy().tobytes()[:tensor_size]
    return pickle.loads(buf)


def _broadcast_object_list(object_list: List[Any],
                           src: int = 0,
                           group: Optional[dist.ProcessGroup] = None) -> None:
    """Broadcast picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in. Note
    that all objects in ``object_list`` must be picklable in order to be
    broadcasted.
    """
    if dist.distributed_c10d._rank_not_in_group(group):
        return

    my_rank = get_rank()
    # Serialize object_list elements to tensors on src rank.
    if my_rank == src:
        tensor_list, size_list = zip(
            *[_object_to_tensor(obj) for obj in object_list])
        object_sizes_tensor = torch.cat(size_list)
    else:
        object_sizes_tensor = torch.empty(len(object_list), dtype=torch.long)

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is ``None`` by default.
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In
    # the case it is not ``None`` we move the size and object tensors to be
    # broadcasted to this device.
    group_backend = get_backend(group)
    is_nccl_backend = group_backend == dist.Backend.NCCL
    current_device = torch.device('cpu')
    if is_nccl_backend:
        # See note about using torch.cuda.current_device() here in
        # docstring. We cannot simply use my_rank since rank == device is
        # not necessarily true.
        current_device = torch.device('cuda', torch.cuda.current_device())
        object_sizes_tensor = object_sizes_tensor.to(current_device)

    # Broadcast object sizes
    dist.broadcast(object_sizes_tensor, src=src, group=group)

    # Concatenate and broadcast serialized object tensors
    if my_rank == src:
        object_tensor = torch.cat(tensor_list)
    else:
        object_tensor = torch.empty(
            torch.sum(object_sizes_tensor).int().item(),
            dtype=torch.uint8,
        )

    if is_nccl_backend:
        object_tensor = object_tensor.to(current_device)
    dist.broadcast(object_tensor, src=src, group=group)
    # Deserialize objects using their stored sizes.
    offset = 0
    if my_rank != src:
        for i, obj_size in enumerate(object_sizes_tensor):
            obj_view = object_tensor[offset:offset + obj_size]
            obj_view = obj_view.type(torch.uint8)
            if obj_view.device != torch.device('cpu'):
                obj_view = obj_view.cpu()
            offset += obj_size
            object_list[i] = _tensor_to_object(obj_view, obj_size)


def broadcast_object_list(data: List[Any],
                          src: int = 0,
                          group: Optional[dist.ProcessGroup] = None) -> None:
    """Broadcasts picklable objects in ``object_list`` to the whole group.
    Similar to :func:`broadcast`, but Python objects can be passed in. Note
    that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Note:
        Calling ``broadcast_object_list`` in non-distributed environment does
        nothing.

    Args:
        data (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank
            will be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.

    Note:
        For NCCL-based process groups, internal tensor representations of
        objects must be moved to the GPU device before communication starts.
        In this case, the used device is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is correctly set so that each rank has an individual
        GPU, via ``torch.cuda.set_device()``.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]
        >>> dist.broadcast_object_list(data)
        >>> data
        ['foo', 12, {1: 2}]

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     data = ["foo", 12, {1: 2}]  # any picklable object
        >>> else:
        >>>     data = [None, None, None]
        >>> dist.broadcast_object_list(data)
        >>> data
        ["foo", 12, {1: 2}]  # Rank 0
        ["foo", 12, {1: 2}]  # Rank 1
    """
    assert isinstance(data, list)

    if get_world_size(group) > 1:
        if group is None:
            group = get_default_group()

        if digit_version(TORCH_VERSION) >= digit_version('1.8.0'):
            dist.broadcast_object_list(data, src, group)
        else:
            _broadcast_object_list(data, src, group)


def all_reduce_dict(data: Dict[str, Tensor],
                    op: str = 'sum',
                    group: Optional[dist.ProcessGroup] = None) -> None:
    """Reduces the dict across all machines in such a way that all get the
    final result.

    The code is modified from https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/utils/allreduce_norm.py.

    Args:
        data (dict[str, Tensor]): Data to be reduced.
        op (str): Operation to reduce data. Defaults to 'sum'. Optional values
            are 'sum', 'mean' and 'produce', 'min', 'max', 'band', 'bor' and
            'bxor'.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = {
                'key1': torch.arange(2, dtype=torch.int64),
                'key2': torch.arange(3, dtype=torch.int64)
            }
        >>> dist.all_reduce_dict(data)
        >>> data
            {'key1': tensor([0, 1]), 'key2': tensor([0, 1, 2])}

        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> data = {
                'key1': torch.arange(2, dtype=torch.int64),
                'key2': torch.arange(3, dtype=torch.int64)
            }
        >>> dist.all_reduce_dict(data)
        >>> data
        {'key1': tensor([0, 2]), 'key2': tensor([0, 2, 4])}  # Rank 0
        {'key1': tensor([0, 2]), 'key2': tensor([0, 2, 4])}  # Rank 1
    """
    assert isinstance(data, dict)

    world_size = get_world_size(group)
    if world_size > 1:

        if group is None:
            group = get_default_group()

        # ensure keys are consistent across processes
        keys = sorted(data.keys())
        tensor_shapes = [data[k].shape for k in keys]
        tensor_sizes = [data[k].numel() for k in keys]

        if digit_version(TORCH_VERSION) == digit_version('1.5.0'):
            # `torch.cat` in torch1.5 can not concatenate different types so
            # we fallback to convert them all to float type.
            flatten_tensor = torch.cat(
                [data[k].flatten().float() for k in keys])
        else:
            flatten_tensor = torch.cat([data[k].flatten() for k in keys])

        all_reduce(flatten_tensor, op=op, group=group)

        split_tensors = [
            x.reshape(shape) for x, shape in zip(
                torch.split(flatten_tensor, tensor_sizes), tensor_shapes)
        ]

        for k, v in zip(keys, split_tensors):
            data[k] = v


def _all_gather_object(object_list: List[Any],
                       obj: Any,
                       group: Optional[dist.ProcessGroup] = None) -> None:
    """Gather picklable objects from the whole group into a list.

    Similar to :func:`all_gather`, but Python objects can be passed in.
    Note that the object must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as
            the size of the group for this collective and will contain the
            output.
        object (Any): Pickable Python object to be broadcast from current
            process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list``
        will be unmodified.
    """
    if dist.distributed_c10d._rank_not_in_group(group):
        return

    input_tensor, local_size = _object_to_tensor(obj)
    group_backend = get_backend(group)
    current_device = torch.device('cpu')
    is_nccl_backend = group_backend == dist.Backend.NCCL
    if is_nccl_backend:
        # See note about using torch.cuda.current_device() here in docstring.
        # We cannot simply use my_rank since rank == device is not necessarily
        # true.
        current_device = torch.device('cuda', torch.cuda.current_device())
        input_tensor = input_tensor.to(current_device)
        local_size = local_size.to(current_device)
    # Gather all local sizes. This is so that we can find the max size, and
    # index until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device)
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes
    dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    coalesced_output_tensor = torch.empty(
        max_object_size * group_size, dtype=torch.uint8, device=current_device)
    # Output tensors are nonoverlapping views of coalesced_output_tensor
    output_tensors = [
        coalesced_output_tensor[max_object_size * i:max_object_size * (i + 1)]
        for i in range(group_size)
    ]
    dist.all_gather(output_tensors, input_tensor, group=group)
    # Deserialize outputs back to object.
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        tensor_size = object_size_list[i]
        object_list[i] = _tensor_to_object(tensor, tensor_size)


def all_gather_object(data: Any,
                      group: Optional[dist.ProcessGroup] = None) -> List[Any]:
    """Gather picklable objects from the whole group into a list. Similar to
    :func:`all_gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Note:
        Calling ``all_gather_object`` in non-distributed environment does
        nothing and just returns a list containing :attr:`data` itself.

    Note:
        Unlike PyTorch ``torch.distributed.all_gather_object``,
        :meth:`all_gather_object` in MMEngine does not pass in an empty list
        ``gather_list`` and returns the ``gather_list`` directly, which is
        more convenient. The difference between their interfaces is as below:

        - MMEngine: all_gather_object(data, group) -> gather_list
        - PyTorch: all_gather_object(gather_list, data, group) -> None

    Args:
        data (Any): Pickable Python object to be broadcast from current
            process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Tensor]: Return a list containing data from the whole group if
        in distributed environment, otherwise a list only containing
        :attr:`data` itself.

    Note:
        For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication starts.
        In this case, the used device is given by
        ``torch.cuda.current_device()`` and it is the user's responsibility to
        ensure that this is correctly set so that each rank has an individual
        GPU, via ``torch.cuda.set_device()``.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]  # any picklable object
        >>> gather_objects = dist.all_gather_object(data[dist.get_rank()])
        >>> output
        ['foo']

        >>> # distributed environment
        >>> # We have 3 process groups, 3 ranks.
        >>> output = dist.all_gather_object(data[dist.get_rank()])
        >>> output
        ['foo', 12, {1: 2}]  # Rank 0
        ['foo', 12, {1: 2}]  # Rank 1
        ['foo', 12, {1: 2}]  # Rank 2
    """
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    gather_list = [None] * world_size

    if digit_version(TORCH_VERSION) >= digit_version('1.8.0'):
        dist.all_gather_object(gather_list, data, group)
    else:
        _all_gather_object(gather_list, data, group)

    return gather_list


def _validate_output_list_for_rank(my_rank: int, dst: int,
                                   gather_list: Optional[list]) -> None:
    """Validate whether ``gather_list`` is None in non-dst ranks."""
    if dst == my_rank:
        if not gather_list:
            raise ValueError(
                'Argument ``gather_list`` must be specified on destination '
                'rank.')
    elif gather_list:
        raise ValueError('Argument ``gather_list`` must NOT be specified '
                         'on non-destination ranks.')


def _gather_object(obj: Any,
                   object_gather_list=None,
                   dst: int = 0,
                   group: Optional[dist.ProcessGroup] = None) -> None:
    """Gathers picklable objects from the whole group in a single process.

    Similar to :func:`gather`, but Python objects can be passed in. Note that
    the object must be picklable in order to be gathered.

    Args:
        obj (Any): Input object. Must be picklable.
        object_gather_list (list[Any], optional): Output list. On the ``dst``
            rank, it should be correctly sized as the size of the group for
            this collective and will contain the output. Must be ``None`` on
            non-dst ranks. Defaults to None.
        dst (int): Destination rank. Defaults to 0.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    """
    if dist.distributed_c10d._rank_not_in_group(group):
        return

    # Ensure object_gather_list is specified appopriately.
    my_rank = get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    input_tensor, local_size = _object_to_tensor(obj)
    group_backend = get_backend(group)
    current_device = torch.device('cpu')
    is_nccl_backend = group_backend == dist.Backend.NCCL
    if is_nccl_backend:
        current_device = torch.device('cuda', torch.cuda.current_device())
        input_tensor = input_tensor.to(current_device)
        local_size = local_size.to(current_device)
    # Gather all local sizes. This is so that we can find the max size, and
    # index until the correct size when deserializing the tensors.
    group_size = get_world_size(group=group)
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=current_device)
    object_size_list = [
        object_sizes_tensor[i].unsqueeze(dim=0) for i in range(group_size)
    ]
    # Allgather tensor sizes. An all-gather is needed here despite this being a
    # gather, since each rank needs to broadcast a tensor of the same (maximal)
    # size.
    dist.all_gather(object_size_list, local_size, group=group)
    max_object_size = int(max(object_size_list).item())
    # Resize tensor to max size across all ranks.
    input_tensor.resize_(max_object_size)
    # Avoid populating output tensors if the result won't be gathered on this
    # rank.
    if my_rank == dst:
        coalesced_output_tensor = torch.empty(
            max_object_size * group_size,
            dtype=torch.uint8,
            device=current_device)
        # Output tensors are nonoverlapping views of coalesced_output_tensor
        output_tensors = [
            coalesced_output_tensor[max_object_size * i:max_object_size *
                                    (i + 1)] for i in range(group_size)
        ]
    # All ranks call gather with equal-sized tensors.
    dist.gather(
        input_tensor,
        gather_list=output_tensors if my_rank == dst else None,
        dst=dst,
        group=group,
    )
    if my_rank != dst:
        return
    for i, tensor in enumerate(output_tensors):
        tensor = tensor.type(torch.uint8)
        tensor_size = object_size_list[i]
        object_gather_list[i] = _tensor_to_object(tensor, tensor_size)


def gather_object(
        data: Any,
        dst: int = 0,
        group: Optional[dist.ProcessGroup] = None) -> Optional[List[Any]]:
    """Gathers picklable objects from the whole group in a single process.
    Similar to :func:`gather`, but Python objects can be passed in. Note that
    the object must be picklable in order to be gathered.

    Note:
        ``NCCL backend`` dost not support ``gather_object``.

    Note:
        Unlike PyTorch ``torch.distributed.gather_object``,
        :meth:`gather_object` in MMEngine does not pass in an empty list
        ``gather_list`` and returns the ``gather_list`` directly, which is
        more convenient. The difference between their interfaces is as below:

        - MMEngine: gather_object(data, dst, group) -> gather_list
        - PyTorch: gather_object(data, gather_list, data, group) -> None

    Args:
        obj (Any): Input object. Must be picklable.
        dst (int): Destination rank. Defaults to 0.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        list[Any]. On the ``dst`` rank, return ``gather_list`` which contains
        the output of the collective.

    Examples:
        >>> import torch
        >>> import mmengine.dist as dist

        >>> # non-distributed environment
        >>> data = ['foo', 12, {1: 2}]  # any picklable object
        >>> gather_objects = dist.gather_object(data[dist.get_rank()])
        >>> output
        ['foo']

        >>> # distributed environment
        >>> # We have 3 process groups, 3 ranks.
        >>> dist.gather_object(gather_objects[dist.get_rank()], dst=0)
        >>> output
        ['foo', 12, {1: 2}]  # Rank 0
        None  # Rank 1
        None  # Rank 2
    """
    world_size = get_world_size(group)
    if world_size == 1:
        return [data]

    if group is None:
        group = get_default_group()

    gather_list = [None] * world_size if get_rank(group) == dst else None

    if digit_version(TORCH_VERSION) >= digit_version('1.8.0'):
        dist.gather_object(data, gather_list, dst, group)
    else:
        _gather_object(data, gather_list, dst, group)

    return gather_list


def collect_results(results: list,
                    size: int,
                    device: str = 'cpu',
                    tmpdir: Optional[str] = None) -> Optional[list]:
    """Collected results in distributed environments.

    Args:
        results (list[object]): Result list containing result parts to be
            collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        device (str): Device name. Optional values are 'cpu' and 'gpu'.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a temporal directory for it.
            ``tmpdir`` should be None when device is 'gpu'. Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results(data, size, device='cpu')
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    if device not in ['gpu', 'cpu']:
        raise NotImplementedError(
            f"device must be 'cpu' or 'gpu', but got {device}")

    if device == 'gpu':
        assert tmpdir is None, 'tmpdir should be None when device is "gpu"'
        return collect_results_gpu(results, size)
    else:
        return collect_results_cpu(results, size, tmpdir)


def collect_results_cpu(result_part: list,
                        size: int,
                        tmpdir: Optional[str] = None) -> Optional[list]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it. Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_cpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmengine.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:  # type: ignore
        pickle.dump(result_part, f, protocol=2)
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            path = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
            with open(path, 'rb') as f:
                part_list.append(pickle.load(f))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)  # type: ignore
        return ordered_results


def collect_results_gpu(result_part: list, size: int) -> Optional[list]:
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list[object]): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_gpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part. Note that NCCL does not support gather so use
    # all_gather
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    else:
        return None
