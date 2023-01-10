# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import functools
import os
import subprocess
from collections.abc import Iterable, Mapping
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import Tensor
from torch import distributed as torch_dist
from torch.distributed import ProcessGroup

from mmengine.device import is_mlu_available, is_npu_available

_LOCAL_PROCESS_GROUP = None


def is_distributed() -> bool:
    """Return True if distributed environment has been initialized."""
    return torch_dist.is_available() and torch_dist.is_initialized()


def get_local_group() -> Optional[ProcessGroup]:
    """Return local process group."""
    if not is_distributed():
        return None

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError('Local process group is not created, please use '
                           '`init_local_group` to setup local process group.')

    return _LOCAL_PROCESS_GROUP


def get_default_group() -> Optional[ProcessGroup]:
    """Return default process group."""

    return torch_dist.distributed_c10d._get_default_group()


def init_dist(launcher, backend='nccl', **kwargs) -> None:
    """Initialize distributed environment.

    Args:
        launcher (str): Way to launcher multi processes. Supported launchers
            are 'pytorch', 'mpi' and 'slurm'.
        backend (str): Communication Backends. Supported backends are 'nccl',
            'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    timeout = kwargs.get('timeout', None)
    if timeout is not None:
        # If a timeout (in seconds) is specified, it must be converted
        # to a timedelta object before forwarding the call to
        # the respective backend, because they expect a timedelta object.
        try:
            kwargs['timeout'] = datetime.timedelta(seconds=timeout)
        except TypeError as exception:
            raise TypeError(
                f'Timeout for distributed training must be provided as '
                f"timeout in seconds, but we've received the type "
                f'{type(timeout)}. Please specify the timeout like this: '
                f"dist_cfg=dict(backend='nccl', timeout=1800)") from exception
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs) -> None:
    """Initialize distributed environment with PyTorch launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    if is_mlu_available():
        import torch_mlu  # noqa: F401
        torch.mlu.set_device(rank)
        torch_dist.init_process_group(
            backend='cncl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    elif is_npu_available():
        import torch_npu  # noqa: F401
        torch.npu.set_device(rank)
        torch_dist.init_process_group(
            backend='hccl',
            rank=rank,
            world_size=int(os.environ['WORLD_SIZE']),
            **kwargs)
    else:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
        torch_dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs) -> None:
    """Initialize distributed environment with MPI launcher.

    Args:
        backend (str): Backend of torch.distributed. Supported backends are
            'nccl', 'gloo' and 'mpi'. Defaults to 'nccl'.
        **kwargs: keyword arguments are passed to ``init_process_group``.
    """
    if backend == 'smddp':
        try:
            import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                'Please use an Amazon SageMaker DLC to access smdistributed: '
                'https://github.com/aws/deep-learning-containers/blob/master'
                '/available_images.md#sagemaker-framework-containers'
                '-sm-support-only') from e
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    torch_dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None) -> None:
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    torch_dist.init_process_group(backend=backend)


def init_local_group(node_rank: int, num_proc_per_node: list):
    """Setup the local process group.

    Setup a process group which only includes processes that on the same
    machine as the current process.

    The code is modified from
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py

    Args:
        node_rank (int): Rank of machines used for training.
        num_gpus_per_node (int): Number of gpus used for training in a single
            machine.
    """  # noqa: W501
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None

    num_machines = len(num_proc_per_node)
    for i in range(num_machines):
        start = sum(num_proc_per_node[:i]) if i != 0 else 0
        end = sum(num_proc_per_node[:i + 1])
        ranks_on_i = list(range(start, end))
        pg = torch_dist.new_group(ranks_on_i)
        if i == node_rank:
            _LOCAL_PROCESS_GROUP = pg


def get_backend(group: Optional[ProcessGroup] = None) -> Optional[str]:
    """Return the backend of the given process group.

    Note:
        Calling ``get_backend`` in non-distributed environment will return
        None.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific
            group is specified, the calling process must be part of
            :attr:`group`. Defaults to None.

    Returns:
        str or None: Return the backend of the given process group as a lower
        case string if in distributed environment, otherwise None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_backend(group)
    else:
        return None


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """Return the number of the given process group.

    Note:
        Calling ``get_world_size`` in non-distributed environment will return
        1.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the number of processes of the given process group if in
        distributed environment, otherwise 1.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_world_size(group)
    else:
        return 1


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """Return the rank of the given process group.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Note:
        Calling ``get_rank`` in non-distributed environment will return 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        int: Return the rank of the process group if in distributed
        environment, otherwise 0.
    """

    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        return torch_dist.get_rank(group)
    else:
        return 0


def get_local_size() -> int:
    """Return the number of the current node.

    Returns:
        int: Return the number of processes in the current node if in
        distributed environment, otherwise 1.
    """
    if not is_distributed():
        return 1

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError('Local process group is not created, please use '
                           '`init_local_group` to setup local process group.')

    return torch_dist.get_world_size(_LOCAL_PROCESS_GROUP)


def get_local_rank() -> int:
    """Return the rank of current process in the current node.

    Returns:
        int: Return the rank of current process in the current node if in
        distributed environment, otherwise 0
    """
    if not is_distributed():
        return 0

    if _LOCAL_PROCESS_GROUP is None:
        raise RuntimeError('Local process group is not created, please use '
                           '`init_local_group` to setup local process group.')

    return torch_dist.get_rank(_LOCAL_PROCESS_GROUP)


def get_dist_info(group: Optional[ProcessGroup] = None) -> Tuple[int, int]:
    """Get distributed information of the given process group.

    Note:
        Calling ``get_dist_info`` in non-distributed environment will return
        (0, 1).

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        tuple[int, int]: Return a tuple containing the ``rank`` and
        ``world_size``.
    """
    world_size = get_world_size(group)
    rank = get_rank(group)
    return rank, world_size


def is_main_process(group: Optional[ProcessGroup] = None) -> bool:
    """Whether the current rank of the given process group is equal to 0.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.

    Returns:
        bool: Return True if the current rank of the given process group is
        equal to 0, otherwise False.
    """
    return get_rank(group) == 0


def master_only(func: Callable) -> Callable:
    """Decorate those methods which should be executed in master process.

    Args:
        func (callable): Function to be decorated.

    Returns:
        callable: Return decorated function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper


def barrier(group: Optional[ProcessGroup] = None) -> None:
    """Synchronize all processes from the given process group.

    This collective blocks processes until the whole group enters this
    function.

    Note:
        Calling ``barrier`` in non-distributed environment will do nothing.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Defaults to None.
    """
    if is_distributed():
        # handle low versions of torch like 1.5.0 which does not support
        # passing in None for group argument
        if group is None:
            group = get_default_group()
        torch_dist.barrier(group)


def get_data_device(data: Union[Tensor, Mapping, Iterable]) -> torch.device:
    """Return the device of ``data``.

    If ``data`` is a sequence of Tensor, all items in ``data`` should have a
    same device type.

    If ``data`` is a dict whose values are Tensor, all values should have a
    same device type.

    Args:
        data (Tensor or Sequence or dict): Inputs to be inferred the device.

    Returns:
        torch.device: The device of ``data``.

    Examples:
        >>> import torch
        >>> from mmengine.dist import cast_data_device
        >>> # data is a Tensor
        >>> data = torch.tensor([0, 1])
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a list of Tensor
        >>> data = [torch.tensor([0, 1]), torch.tensor([2, 3])]
        >>> get_data_device(data)
        device(type='cpu')
        >>> # data is a dict
        >>> data = {'key1': torch.tensor([0, 1]), 'key2': torch.tensor([0, 1])}
        >>> get_data_device(data)
        device(type='cpu')
    """
    if isinstance(data, Tensor):
        return data.device
    elif isinstance(data, Mapping):
        pre = None
        for v in data.values():
            cur = get_data_device(v)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    elif isinstance(data, Iterable) and not isinstance(data, str):
        pre = None
        for item in data:
            cur = get_data_device(item)
            if pre is None:
                pre = cur
            else:
                if cur != pre:
                    raise ValueError(
                        'device type in data should be consistent, but got '
                        f'{cur} and {pre}')
        if pre is None:
            raise ValueError('data should not be empty.')
        return pre
    else:
        raise TypeError('data should be a Tensor, sequence of tensor or dict, '
                        f'but got {data}')


def get_comm_device(group: Optional[ProcessGroup] = None) -> torch.device:
    """Return the device for communication among groups.

    Args:
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        torch.device: The device of backend.
    """
    backend = get_backend(group)
    if backend == 'hccl':
        import torch_npu  # noqa: F401
        return torch.device('npu', torch.npu.current_device())
    elif backend == torch_dist.Backend.NCCL:
        return torch.device('cuda', torch.cuda.current_device())
    elif backend == 'cncl':
        import torch_mlu  # noqa: F401
        return torch.device('mlu', torch.mlu.current_device())
    elif backend == 'smddp':
        return torch.device('cuda', torch.cuda.current_device())
    else:
        # GLOO and MPI backends use cpu device by default
        return torch.device('cpu')


def cast_data_device(
    data: Union[Tensor, Mapping, Iterable],
    device: torch.device,
    out: Optional[Union[Tensor, Mapping, Iterable]] = None
) -> Union[Tensor, Mapping, Iterable]:
    """Recursively convert Tensor in ``data`` to ``device``.

    If ``data`` has already on the ``device``, it will not be casted again.

    Args:
        data (Tensor or list or dict): Inputs to be casted.
        device (torch.device): Destination device type.
        out (Tensor or list or dict, optional): If ``out`` is specified, its
            value will be equal to ``data``. Defaults to None.

    Returns:
        Tensor or list or dict: ``data`` was casted to ``device``.
    """
    if out is not None:
        if type(data) != type(out):
            raise TypeError(
                'out should be the same type with data, but got data is '
                f'{type(data)} and out is {type(data)}')

        if isinstance(out, set):
            raise TypeError('out should not be a set')

    if isinstance(data, Tensor):
        if get_data_device(data) == device:
            data_on_device = data
        else:
            data_on_device = data.to(device)

        if out is not None:
            # modify the value of out inplace
            out.copy_(data_on_device)  # type: ignore

        return data_on_device
    elif isinstance(data, Mapping):
        data_on_device = {}
        if out is not None:
            data_len = len(data)
            out_len = len(out)  # type: ignore
            if data_len != out_len:
                raise ValueError('length of data and out should be same, '
                                 f'but got {data_len} and {out_len}')

            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device,
                                                     out[k])  # type: ignore
        else:
            for k, v in data.items():
                data_on_device[k] = cast_data_device(v, device)

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        # To ensure the type of output as same as input, we use `type(data)`
        # to wrap the output
        return type(data)(data_on_device)  # type: ignore
    elif isinstance(data, Iterable) and not isinstance(
            data, str) and not isinstance(data, np.ndarray):
        data_on_device = []
        if out is not None:
            for v1, v2 in zip(data, out):
                data_on_device.append(cast_data_device(v1, device, v2))
        else:
            for v in data:
                data_on_device.append(cast_data_device(v, device))

        if len(data_on_device) == 0:
            raise ValueError('data should not be empty')

        return type(data)(data_on_device)  # type: ignore
    else:
        raise TypeError('data should be a Tensor, list of tensor or dict, '
                        f'but got {data}')


def launch(
        main_func: Callable,
        num_proc_per_machine: Union[list, int] = 1,
        num_machines: int = 1,
        machine_rank: int = 0,
        master_addr: str = '127.0.0.1',
        master_port: str = 'auto',
        args: tuple = (),
) -> None:
    """Launch distributed task with single or multiple machines/GPU.

    Args:
        main_func (Callable): Function to be executed in multiple process.
        num_proc_per_machine (list or int): number of valid gpu for machines.
            For example, The task will be ran on 2 machine A and machine B.
            A has 2 valid GPUs, and B has 4 valid GPUs. Then
            ``num_proc_per_machine`` should be [2, 4]
        num_machines (int, optional): Number of used machines. Defaults to 1.
        machine_rank (int, optional): The rank of current machine.
            Defaults to 0.
        master_addr (str, optional): The FQDN of the host that is running
            worker with rank 0; used to initialize the Torch Distributed
            backend. Defaults to '127.0.0.1'.
        master_port (str, optional): The port on the ``master_addr`` that can
            be used to host the C10d TCP store. Defaults to 'auto'.
        args (tuple, optional): Arguments passde to main_func. Defaults to ().
    """
    if not isinstance(num_proc_per_machine, list):
        num_proc_per_machine = [num_proc_per_machine] * num_machines
    torch_dist.init_process_group
    world_size = sum(num_proc_per_machine)
    if master_port == 'auto':
        master_port = str(_find_free_port())

    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391

        mp.start_processes(
            _distributed_worker,
            nprocs=num_proc_per_machine[machine_rank],
            args=(
                main_func,
                world_size,
                num_proc_per_machine,
                machine_rank,
                master_addr,
                master_port,
                args,
            ),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_proc_per_machine: list,
    machine_rank: int,
    master_addr: str,
    master_port: str,
    args,
) -> None:
    """Run the task after initializing the environment.

    This function will be launched by ``mp.start_processes`` and ``local_rank``
    will be filled automatically.

    Part of environment variable defined in `pytorch official docs <https://pytorch.org/docs/stable/elastic/run.html#environment-variables>`_
    will be initialized.

    Warning:
        :func:`init_dist` must be called in the ``main_func``

    Args:
        local_rank (int): Local rank of the current machine.
        main_func (Callable): Function to be executed.
        world_size (int): The number of all processes launched on all machines.
        num_proc_per_machine (list): Number of valid gpu for machines.
        machine_rank (int): The rank of current machine.
        master_addr (str): The FQDN of the host that is running
            worker with rank 0; used to initialize the Torch Distributed
            backend.
        master_port (str): The port on the ``master_addr`` that can
            be used to host the C10d TCP store.
        args (tuple, optional): Arguments passde to main_func. Defaults to ().
    """  # noqa: E501
    has_gpu = torch.cuda.is_available()
    if has_gpu:
        assert num_proc_per_machine[machine_rank] <= torch.cuda.device_count()
    os.environ['RANK'] = \
        str(sum(num_proc_per_machine[:machine_rank]) + local_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    num_proc_per_machine = [str(num) for num in num_proc_per_machine]
    os.environ['NUM_PROC_PER_NODE'] = ' '.join(num_proc_per_machine)

    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group.
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    barrier()

    main_func(*args)


def _find_free_port() -> str:
    """Find free port.

    Returns:
        str: Free port.
    """
    import socket

    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
