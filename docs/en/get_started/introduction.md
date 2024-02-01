# Introduction

MMEngine is a foundational library for training deep learning models based on
PyTorch. It supports running on Linux, Windows, and macOS. Its highlights are as follows:

**Integrate mainstream large-scale model training frameworks**

- [ColossalAI](../common_usage/large_model_training.md#colossalai)
- [DeepSpeed](../common_usage/large_model_training.md#deepspeed)
- [FSDP](../common_usage/large_model_training.md#fullyshardeddataparallel-fsdp)

**Supports a variety of training strategies**

- [Mixed Precision Training](../common_usage/speed_up_training.md#mixed-precision-training)
- [Gradient Accumulation](../common_usage/save_gpu_memory.md#gradient-accumulation)
- [Gradient Checkpointing](../common_usage/save_gpu_memory.md#gradient-checkpointing)

**Provides a user-friendly configuration system**

- [Pure Python-style configuration files, easy to navigate](../advanced_tutorials/config.md#a-pure-python-style-configuration-file-beta)
- [Plain-text-style configuration files, supporting JSON and YAML](../advanced_tutorials/config.html)

**Covers mainstream training monitoring platforms**

- [TensorBoard](../common_usage/visualize_training_log.md#tensorboard) | [WandB](../common_usage/visualize_training_log.md#wandb) | [MLflow](../common_usage/visualize_training_log.md#mlflow-wip)
- [ClearML](../common_usage/visualize_training_log.md#clearml) | [Neptune](../common_usage/visualize_training_log.md#neptune) | [DVCLive](../common_usage/visualize_training_log.md#dvclive) | [Aim](../common_usage/visualize_training_log.md#aim)

## Architecture

![openmmlab-2.0-arch](https://user-images.githubusercontent.com/40779233/187065730-1e9af236-37dc-4dbd-b448-cce3b72b0109.png)

The above diagram illustrates the hierarchy of MMEngine in OpenMMLab 2.0.
MMEngine implements a next-generation training architecture for the OpenMMLab
algorithm library, providing a unified execution foundation for over 30
algorithm libraries within OpenMMLab. Its core components include the training
engine, evaluation engine, and module management.

## Module Introduction

MMEngine abstracts the components involved in the training process and their
relationships. Components of the same type in different algorithm libraries
share the same interface definition.

### Core Modules and Related Components

The core module of the training engine is the
[`Runner`](../tutorials/runner.md). The `Runner` is responsible for executing
training, testing, and inference tasks and managing the various components
required during these processes. In specific locations throughout the
execution of training, testing, and inference tasks, the `Runner` sets up Hooks
to allow users to extend, insert, and execute custom logic. The `Runner`
primarily invokes the following components to complete the training and
inference loops:

- [Dataset](../tutorials/dataset.md): Responsible for constructing datasets in
  training, testing, and inference tasks, and feeding the data to the model.
  In usage, it is wrapped by a PyTorch DataLoader, which launches multiple
  subprocesses to load the data.
- [Model](../tutorials/model.md): Accepts data and outputs the loss during the
  training process; accepts data and performs predictions during testing and
  inference tasks. In a distributed environment, the model is wrapped by a
  Model Wrapper (e.g., `MMDistributedDataParallel`).
- [Optimizer Wrapper](../tutorials/optim_wrapper.md): The optimizer wrapper
  performs backpropagation to optimize the model during the training process
  and supports mixed-precision training and gradient accumulation through a
  unified interface.
- [Parameter Scheduler](../tutorials/param_scheduler.md): Dynamically adjusts
  optimizer hyperparameters such as learning rate and momentum during the
  training process.

During training intervals or testing phases, the [Metrics &
Evaluator](../tutorials/evaluation.md) are responsible for evaluating the
performance of the model. The `Evaluator` evaluates the model's predictions
based on the dataset. Within the `Evaluator`, there is an abstraction called
`Metrics`, which calculates various metrics such as recall, accuracy, and
others.

To ensure a unified interface, the communication interfaces between the
evaluators, models, and data in various algorithm libraries within OpenMMLab
2.0 are encapsulated using
[Data Elements](../advanced_tutorials/data_element.md).

During training and inference execution, the aforementioned components can
utilize the logging management module and visualizer for structured and
unstructured logging storage and visualization. [Logging
Modules](../advanced_tutorials/logging.md): Responsible for managing various
log information generated during the execution of the Runner. The Message Hub
implements data sharing between components, runners, and log processors, while
the Log Processor processes the log information. The processed logs are then
sent to the `Logger` and `Visualizer` for management and display. The
[`Visualizer`](../advanced_tutorials/visualization.md) is responsible for
visualizing the model's feature maps, prediction results, and structured logs
generated during the training process. It supports multiple visualization
backends such as TensorBoard and WanDB.

### Common Base Modules

MMEngine also implements various common base modules required during the
execution of algorithmic models, including:

- [Config](../advanced_tutorials/config.md): In the OpenMMLab algorithm library, users can configure the training, testing process,
  and related components by writing a configuration file (config).
- [Registry](../advanced_tutorials/registry.md): Responsible for managing
  modules within the algorithm library that have similar functionality. Based on the abstraction of algorithm library modules, MMEngine defines a set of root registries. Registries within the algorithm library can inherit from these root registries, enabling cross-algorithm library module invocations and interactions. This allows for seamless integration and utilization of modules across different algorithms within the OpenMMLab framework.
- [File I/O](../advanced_tutorials/fileio.md): Provides a unified interface
  for file read/write operations in various modules, supporting multiple file
  backend systems and formats in a consistent manner, with extensibility.
- [Distributed Communication Primitives](../advanced_tutorials/distributed.md):
  Handles communication between different processes during distributed program
  execution. This interface abstracts the differences between distributed and
  non-distributed environments and automatically handles data devices and
  communication backends.
- [Other Utilities](../advanced_tutorials/manager_mixin.md): There are also
  utility modules, such as `ManagerMixin`, which implements a way to create
  and access global variables. The base class for many globally accessible
  objects within the `Runner` is `ManagerMixin`.

Users can further read the [tutorials](../tutorials/runner.md) to understand the advanced usage of these
modules or refer to the [design documents](../design/hook.md) to understand their design principles
and details.
