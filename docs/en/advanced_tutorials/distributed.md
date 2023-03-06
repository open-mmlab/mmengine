# Distribution Communication

In distributed training, different processes sometimes need to apply different logics depending on their ranks, local_ranks, etc.
They also need to communicate with each other and do synchronizations on data.
These demands rely on distributed communication.
PyTorch provides a set of basic distributed communication primitives.
Based on these primitives, MMEngine provides some higher level APIs to meet more diverse demands.
Using these APIs provided by MMEngine, modules can:

- ignore the differences between distributed/non-distributed environment
- deliver data in various types apart from Tensor
- ignore the frameworks or backends used for communication

These APIs are roughly categorized into 3 types:

- Initialization: `init_dist` for setting up distributed environment for the runner
- Query & control: functions including `get_world_size` for querying `world_size`, `rank` and other distributed information
- Collective communication: collective communication functions such as `all_reduce`

We will detail on these APIs in the following chapters.

## Initialization

- [init_dist](mmengine.dist.init_dist): Launch function of distributed training. Currently it supports 3 launchers including pytorch, slurm and MPI. It also setup the given communication backends, defaults to NCCL.

  If you need to change the runtime timeout (default=30 minutes) for distributed operations that take very long, you can specify a different timeout in your `env_cfg` configuration passing in [Runner](mmengine.runner.Runner) like this:

  ```python
  env_cfg = dict(
      cudnn_benchmark=True,
      mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
      dist_cfg=dict(backend='nccl', timeout=10800), # Sets the timeout to 3h (10800 seconds)
  )
  runner = Runner(xxx, env_cfg=env_cfg)
  ```

## Query and control

The query and control functions are all argument free.
They can be used in both distributed and non-distributed environment.
Their functionalities are listed below:

- [get_world_size](mmengine.dist.get_world_size): Returns the number of processes in current process group. Returns 1 when non-distributed
- [get_rank](mmengine.dist.get_rank): Returns the global rank of current process in current process group. Returns 0 when non-distributed
- [get_backend](mmengine.dist.get_backend): Returns the communication backends used by current process group. Returns `None` when non-distributed
- [get_local_rank](mmengine.dist.get_local_rank): Returns the local rank of current process in current process group. Returns 0 when non-distributed
- [get_local_size](mmengine.dist.get_local_size): Returns the number of processes which are both in current process group and on the same machine as the current process. Returns 1 when non-distributed
- [get_dist_info](mmengine.dist.get_dist_info): Returns the world_size and rank of the current process group. Returns world_size = 1, rank = 0 when non-distributed
- [is_main_process](mmengine.dist.is_main_process): Returns `True` if current process is rank 0 in current process group, otherwise `False` . Always returns `True` when non-distributed
- [master_only](mmengine.dist.master_only): A function decorator. Functions decorated by `master_only` will only execute on rank 0 process.
- [barrier](mmengine.dist.barrier): A synchronization primitive. Every process will hold until all processes in the current process group reach the same barrier location

## Collective communication

Collective communication functions are used for data transfer between processes in the same process group.
We provide the following APIs based on PyTorch native functions including all_reduce, all_gather, gather, broadcast.
These APIs are compatible with non-distributed environment and support more data types apart from Tensor.

- [all_reduce](mmengine.dist.all_reduce): AllReduce operation on Tensors in the current process group
- [all_gather](mmengine.dist.all_gather): AllGather operation on Tensors in the current process group
- [gather](mmengine.dist.gather): Gather Tensors in the current process group to a destinated rank
- [broadcast](mmengine.dist.broadcast): Broadcast a Tensor to all processes in the current process group
- [sync_random_seed](mmengine.dist.sync_random_seed): Synchronize random seed between processes in the current process group
- [broadcast_object_list](mmengine.dist.broadcast_object_list): Broadcast a list of Python objects. It requires the object can be serialized by Pickle.
- [all_reduce_dict](mmengine.dist.all_reduce_dict): AllReduce operation on dict. It is based on broadcast and all_reduce.
- [all_gather_object](mmengine.dist.all_gather_object): AllGather operations on any Python object than can be serialized by Pickle. It is based on all_gather
- [gather_object](mmengine.dist.gather_object): Gather Python objects that can be serialized by Pickle
- [collect_results](mmengine.dist.collect_results): Unified API for collecting a list of data in current process group. It support both CPU and GPU communication
