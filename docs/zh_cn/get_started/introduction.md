# 介绍

MMEngine 是一个基于 PyTorch 实现的，用于训练深度学习模型的基础库，支持在 Linux、Windows、macOS 上运行。它的亮点如下：

**集成主流的大模型训练框架**

- [ColossalAI](../common_usage/large_model_training.md#colossalai)
- [DeepSpeed](../common_usage/large_model_training.md#deepspeed)
- [FSDP](../common_usage/large_model_training.md#fullyshardeddataparallel-fsdp)

**支持丰富的训练策略**

- [混合精度训练（Mixed Precision Training）](../common_usage/speed_up_training.md#混合精度训练)
- [梯度累积（Gradient Accumulation）](../common_usage/save_gpu_memory.md#梯度累加)
- [梯度检查点（Gradient Checkpointing）](../common_usage/save_gpu_memory.md#梯度检查点)

**提供易用的配置系统**

- [纯 Python 风格的配置文件，易于跳转](../advanced_tutorials/config.md#纯-python-风格的配置文件beta)
- [纯文本风格的配置文件，支持 JSON 和 YAML](../advanced_tutorials/config.md)

**覆盖主流的训练监测平台**

- [TensorBoard](../common_usage/visualize_training_log.md#tensorboard) | [WandB](../common_usage/visualize_training_log.md#wandb) | [MLflow](../common_usage/visualize_training_log.md#mlflow-wip)
- [ClearML](../common_usage/visualize_training_log.md#clearml) | [Neptune](../common_usage/visualize_training_log.md#neptune) | [DVCLive](../common_usage/visualize_training_log.md#dvclive) | [Aim](../common_usage/visualize_training_log.md#aim)

**兼容主流的训练芯片**

- 英伟达 CUDA | 苹果 MPS
- 华为 Ascend | 寒武纪 MLU | 摩尔线程 MUSA

## 架构

![openmmlab-2 0-arch](https://user-images.githubusercontent.com/40779233/187065730-1e9af236-37dc-4dbd-b448-cce3b72b0109.png)

上图展示了 MMEngine 在 OpenMMLab 2.0 中的层次。MMEngine 实现了 OpenMMLab 算法库的新一代训练架构，为 OpenMMLab 中的 30 多个算法库提供了统一的执行基座。其核心组件包含训练引擎、评测引擎和模块管理等。

## 模块介绍

<img src="https://user-images.githubusercontent.com/40779233/187156277-7c5d020b-7ba6-421b-989d-2990034ff8cc.png" width = "300" alt="模块关系" align=center />

MMEngine 将训练过程中涉及的组件和它们的关系进行了抽象，如上图所示。不同算法库中的同类型组件具有相同的接口定义。

### 核心模块与相关组件

训练引擎的核心模块是[执行器（Runner）](../tutorials/runner.md)。执行器负责执行训练、测试和推理任务并管理这些过程中所需要的各个组件。在训练、测试、推理任务执行过程中的特定位置，执行器设置了[钩子（Hook）](../tutorials/hook.md)来允许用户拓展、插入和执行自定义逻辑。执行器主要调用如下组件来完成训练和推理过程中的循环：

- [数据集（Dataset）](../tutorials/dataset.md)：负责在训练、测试、推理任务中构建数据集，并将数据送给模型。实际使用过程中会被数据加载器（DataLoader）封装一层，数据加载器会启动多个子进程来加载数据。
- [模型（Model）](../tutorials/model.md)：在训练过程中接受数据并输出 loss；在测试、推理任务中接受数据，并进行预测。分布式训练等情况下会被模型的封装器（Model Wrapper，如 `MMDistributedDataParallel`）封装一层。
- [优化器封装（Optimizer）](../tutorials/optim_wrapper.md)：优化器封装负责在训练过程中执行反向传播优化模型，并且以统一的接口支持了混合精度训练和梯度累加。
- [参数调度器（Parameter Scheduler）](../tutorials/param_scheduler.md)：训练过程中，对学习率、动量等优化器超参数进行动态调整。

在训练间隙或者测试阶段，[评测指标与评测器（Metrics & Evaluator）](../tutorials/evaluation.md)会负责对模型性能进行评测。其中评测器负责基于数据集对模型的预测进行评估。评测器内还有一层抽象是评测指标，负责计算具体的一个或多个评测指标（如召回率、正确率等）。

为了统一接口，OpenMMLab 2.0 中各个算法库的评测器，模型和数据之间交流的接口都使用了[数据元素（Data Element）](../advanced_tutorials/data_element.md)来进行封装。

在训练、推理执行过程中，上述各个组件都可以调用日志管理模块和可视化器进行结构化和非结构化日志的存储与展示。[日志管理（Logging Modules）](../advanced_tutorials/logging.md)：负责管理执行器运行过程中产生的各种日志信息。其中消息枢纽（MessageHub）负责实现组件与组件、执行器与执行器之间的数据共享，日志处理器（Log Processor）负责对日志信息进行处理，处理后的日志会分别发送给执行器的日志器（Logger）和可视化器（Visualizer）进行日志的管理与展示。[可视化器（Visualizer）](../advanced_tutorials/visualization.md)：可视化器负责对模型的特征图、预测结果和训练过程中产生的结构化日志进行可视化，支持 Tensorboard 和 WanDB 等多种可视化后端。

### 公共基础模块

MMEngine 中还实现了各种算法模型执行过程中需要用到的公共基础模块，包括

- [配置类（Config）](../advanced_tutorials/config.md)：在 OpenMMLab 算法库中，用户可以通过编写 config 来配置训练、测试过程以及相关的组件。
- [注册器（Registry）](../advanced_tutorials/registry.md)：负责管理算法库中具有相同功能的模块。MMEngine 根据对算法库模块的抽象，定义了一套根注册器，算法库中的注册器可以继承自这套根注册器，实现模块的跨算法库调用。
- [文件读写（File I/O）](../advanced_tutorials/fileio.md)：为各个模块的文件读写提供了统一的接口，以统一的形式支持了多种文件读写后端和多种文件格式，并具备扩展性。
- [分布式通信原语（Distributed Communication Primitives）](../advanced_tutorials/distributed.md)：负责在程序分布式运行过程中不同进程间的通信。这套接口屏蔽了分布式和非分布式环境的区别，同时也自动处理了数据的设备和通信后端。
- [其他工具（Utils）](../advanced_tutorials/manager_mixin.md)：还有一些工具性的模块，如 ManagerMixin，它实现了一种全局变量的创建和获取方式，执行器内很多全局可见对象的基类就是 ManagerMixin。

用户可以进一步阅读[教程](../tutorials/runner.md)来了解这些模块的高级用法，也可以参考[设计文档](../design/hook.md) 了解它们的设计思路与细节。
