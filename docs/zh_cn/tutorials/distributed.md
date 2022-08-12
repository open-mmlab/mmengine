# 分布式通信原语

在分布式训练或测试的过程中，不同进程有时需要根据分布式的环境信息执行不同的代码逻辑，同时不同进程之间也经常会有相互通信的需求，对一些数据进行同步等操作。
PyTorch 提供了一套基础的通信原语用于多进程之间张量的通信，基于这套原语，MMEngine 实现了更高层次的通信原语封装以满足更加丰富的需求。基于 MMEngine 的通信原语，算法库中的模块可以

1. 在使用通信原语封装时不显式区分分布式/非分布式环境
2. 进行除 Tensor 以外类型数据的多进程通信
3. 无需了解底层通信后端或框架

这些通信原语封装的接口和功能可以大致归类为如下三种，我们在后续章节中逐个介绍

1. 分布式初始化：`init_dist` 负责初始化执行器的分布式环境
2. 分布式信息获取与控制：包括 `get_world_size` 等函数获取当前的 `rank` 和 `world_size` 等信息
3. 分布式通信接口：包括如 `all_reduce` 等通信函数（collective functions）

## 分布式初始化

- [init_dist](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.init_dist)： 是分布式训练的启动函数，目前支持 pytorch，slurm，MPI 3 种分布式启动方式，同时允许设置通信的后端，默认使用 NCCL。

## 分布式信息获取与控制

分布式信息的获取与控制函数没有参数，这些函数兼容非分布式训练的情况，功能如下

- [get_world_size](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.get_world_size)：获取当前进程组的进程总数，非分布式情况下返回 1
- [get_rank](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.get_rank)：获取当前进程对应的全局 rank 数，非分布式情况下返回 0
- [get_backend](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.get_backend)：获取当前通信使用的后端，非分布式情况下返回 None
- [get_local_rank](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.get_local_rank)：获取当前进程对应到当前机器的 rank 数，非分布式情况下返回 0
- [get_local_size](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.get_local_size)：获取当前进程所在机器的总进程数，非分布式情况下返回 0
- [get_dist_info](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.get_dist_info)：获取当前任务的进程总数和当前进程对应到全局的 rank 数，非分布式情况下 word_size = 1，rank = 0
- [is_main_process](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.is_main_process)：判断是否为 0 号主进程，非分布式情况下返回 True
- [master_only](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.master_only)：函数装饰器，用于修饰只需要全局 0 号进程（rank 0 而不是 local rank 0）执行的函数
- [barrier](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.utils.barrier)：同步所有进程到达相同位置

## 分布式通信函数

通信函数 （Collective functions），主要用于进程间数据的通信，基于 PyTorch 原生的 all_reduce，all_gather，gather，broadcast 接口，MMEngine 提供了如下接口，兼容非分布式训练的情况，并支持更丰富数据类型的通信。

- [all_reduce](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.all_reduce): 对进程间 tensor 进行 AllReduce 操作
- [all_gather](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.all_gather)：对进程间 tensor 进行 AllGather 操作
- [gather](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.gather)：将进程的 tensor 收集到一个目标 rank
- [broadcast](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.broadcast)：对某个进程的 tensor 进行广播
- [sync_random_seed](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.sync_random_seed)：同步进程之间的随机种子
- [broadcast_object_list](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.broadcast_object_list)：支持对任意可被 Pickle 序列化的 Python 对象列表进行广播，基于 broadcast 接口实现
- [all_reduce_dict](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.all_reduce_dict)：对 dict 中的内容进行 all_reduce 操作，基于 broadcast 和 all_reduce 接口实现
- [all_gather_object](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.all_gather_object)：基于 all_gather 实现对任意可以被 Pickle 序列化的 Python 对象进行 all_tather 操作
- [gather_object](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.gather_object)：将 group 里每个 rank 的  可被 Pickle 序列化的 Python 对象 gather 到一个目标 rank，且支持多种方式
- [collect_results](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.dist.collect_results)：支持基于 CPU 通信或者 GPU 通信对不同进程间的列表数据进行收集
