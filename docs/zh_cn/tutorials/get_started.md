# 使用 MMEngine 来训练模型

MMEngine 实现了 OpenMMLab 算法库的新一代训练架构，为算法模型的训练、测试、推理和可视化定义了一套基类与接口。
和 OpenMMLab 算法库的上一代训练架构相比，它具有如下三个特点：

- 统一：为不同方向算法模型的训练、测试、推理、和可视化过程进行了抽象并定义了一套统一的接口
- 清晰：封装的层次与逻辑清晰简单，抽象的定义与接口更加清晰，模块的拆分与边界更加清晰
- 灵活：在统一的基础框架内，模块可以灵活拓展和插拔，支持各类型算法和学习范式，包括少样本和零样本学习，自监督、半监督、和弱监督学习，和模型的蒸馏、剪枝、与量化。

## 组件

MMEngine 将算法模型训练、推理、测试和可视化过程中的各个组件进行了抽象，定义了如下几个组件和他们的相关接口，这些组件的关系如下图所示：

![runner_modules](https://user-images.githubusercontent.com/40779233/165018333-c54a3405-a566-4de6-a1d5-f2a3a836bd41.jpeg)

以下根据上图简述这些模块的功能与联系，用户可以通过各个组件的用户文档了解他们。

- [执行器（Runner）](./runner.md)：负责执行训练、测试和推理任务并管理这些过程中所需要的各个组件。
- [钩子（Hook）](./hook.md)：负责在训练、测试、推理任务执行过程中的特定位置执行自定义逻辑。
- [数据集（Dataset）](./basedataset.md)：负责在训练、测试、推理任务中构建数据集，并将数据送给模型。实际使用过程中会被数据加载器（DataLoader）封装一层，数据加载器会启动多个子进程来加载数据。
- [模型（Model）](./model.md)：在训练过程中接受数据、输出 loss，在测试、推理任务中接受数据，并进行预测。分布式训练等情况下会被模型的封装器（Model Wrapper，如 .`nn.DistributedDataParallel`）封装一层。
- [评测指标与评测器（Metrics & Evaluator）](./metric_and_evaluator.md)：评测器负责基于数据集对模型的预测进行评估。评测器内还有一层抽象是评测指标，负责计算具体的一个或多个评测指标（如召回率、正确率等）。
- [数据元素（Data Element）](./data_element.md)：评测器，模型和数据之间交流的接口使用数据元素进行封装。
- [参数调度器（Parameter Scheduler）](./param_scheduler.md)：训练过程中，对学习率、动量等参数进行动态调整。
- [优化器（Optimizer）](./optimizer_wrapper.md)：优化器负责在训练过程中执行反向传播优化模型。
- [日志管理（Logging Modules）](./logging.md)：负责管理 Runner 运行过程中产生的各种日志信息。其中消息枢纽 （MessageHub）负责实现组件与组件、执行器与执行器之间的数据共享，日志处理器（Log Processor）负责对日志信息进行处理，处理后的日志会分别发送给执行器的日志器（Logger）和可视化器（Visualizer）进行日志的管理与展示。
- [配置类（Config）](./config.md)：在 OpenMMLab 算法库中，用户可以通过编写 config 来配置训练、测试过程以及相关的组件。
- [注册器（Registry）](./registry.md)：负责管理算法库中具有相同功能的模块。MMEngine 根据对算法库模块的抽象，定义了一套根注册器，算法库中的注册器可以继承自这套根注册器，实现模块的跨算法库调用。
- [分布式通信原语（Distributed Communication Primitives）](./distributed.md)：负责在程序分布式运行过程中不同进程间的通信。这套接口屏蔽了分布式和非分布式环境的区别，同时也自动处理了数据的设备和通信后端。
- [其他工具（Utils）](./utils.md)：还有一些工具性的模块，如管理器混入（ManagerMixin），它实现了一种全局变量的创建和获取方式，Runner 内很多全局可见对象的基类就是 ManagerMixin。
