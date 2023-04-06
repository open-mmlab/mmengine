# 可视化

## 1 总体设计

可视化可以给深度学习的模型训练和测试过程提供直观解释。在 OpenMMLab 算法库中，我们期望可视化功能的设计能满足以下需求：

- 提供丰富的开箱即用可视化功能，能够满足大部分计算机视觉可视化任务
- 高扩展性，可视化功能通常多样化，应该能够通过简单扩展实现定制需求
- 能够在训练和测试流程的任意点位进行可视化
- OpenMMLab 各个算法库具有统一可视化接口，利于用户理解和维护

基于上述需求，OpenMMLab 2.0 引入了可视化对象 Visualizer 和各个可视化存储后端 VisBackend 如 `LocalVisBackend`、`WandbVisBackend` 和 `TensorboardVisBackend` 等。此处的可视化不仅仅包括图片数据格式，还包括配置内容、标量和模型图等数据的可视化。

- 为了方便调用，Visualizer 提供的接口实现了绘制和存储的功能。可视化存储后端 VisBackend 作为 Visualizer 的内部属性，会在需要的时候被 Visualizer 调用，将数据存到不同的后端
- 考虑到绘制后会希望存储到多个后端，Visualizer 可以配置多个 VisBackend，当用户调用 Visualizer 的存储接口时候，Visualizer 内部会遍历的调用 VisBackend 存储接口

两者的 UML 关系图如下

<div align="center">
 <img src="https://user-images.githubusercontent.com/17425982/163327736-f7cb3b16-ef07-46bc-982a-3cc7495e6c82.png" >
</div>

## 2 可视化器 Visualizer

可视化对象 Visualizer 对外提供了所有接口。可以将其接口分成 3 大类，如下所示

**(1) 绘制相关接口**

- [draw_bboxes](mmengine.visualization.Visualizer.draw_bboxes) 绘制单个或多个边界框
- [draw_points](mmengine.visualization.Visualizer.draw_points) 绘制单个或多个点
- [draw_texts](mmengine.visualization.Visualizer.draw_texts) 绘制单个或多个文本框
- [draw_lines](mmengine.visualization.Visualizer.draw_lines) 绘制单个或多个线段
- [draw_circles](mmengine.visualization.Visualizer.draw_circles) 绘制单个或多个圆
- [draw_polygons](mmengine.visualization.Visualizer.draw_polygons) 绘制单个或多个多边形
- [draw_binary_masks](mmengine.visualization.Visualizer.draw_binary_masks) 绘制单个或多个二值掩码
- [draw_featmap](mmengine.visualization.Visualizer.draw_featmap) 绘制特征图，静态方法

上述接口除了 `draw_featmap` 外都可以链式调用，因为该方法调用后可能会导致图片尺寸发生改变。为了避免给用户带来困扰， `draw_featmap` 被设置为静态方法。

**(2) 存储相关接口**

- [add_config](mmengine.visualization.Visualizer.add_config) 写配置到特定存储后端
- [add_graph](mmengine.visualization.Visualizer.add_graph) 写模型图到特定存储后端
- [add_image](mmengine.visualization.Visualizer.add_image) 写图片到特定存储后端
- [add_scalar](mmengine.visualization.Visualizer.add_scalar) 写标量到特定存储后端
- [add_scalars](mmengine.visualization.Visualizer.add_scalars) 一次性写多个标量到特定存储后端
- [add_datasample](mmengine.visualization.Visualizer.add_datasample) 各个下游库绘制 datasample 数据的抽象接口

以 add 前缀开头的接口表示存储接口。datasample 是 OpenMMLab 2.0 架构中设计的各个下游库统一的抽象数据接口，而 `add_datasample` 接口可以直接处理该数据格式，例如可视化预测结果、可视化 Dataset 或者 DataLoader 输出、可视化中间预测结果等等都可以直接调用下游库重写的 `add_datasample` 接口。
所有下游库都必须要继承 Visualizer 并实现 `add_datasample` 接口。以 MMDetection 为例，应该继承并通过该接口实现目标检测中所有预置任务的可视化功能，例如目标检测、实例分割、全景分割任务结果的绘制和存储。

**(3) 其余功能性接口**

- [set_image](mmengine.visualization.Visualizer.set_image) 设置原始图片数据，默认输入图片格式为 RGB
- [get_image](mmengine.visualization.Visualizer.get_image) 获取绘制后的 Numpy 格式图片数据，默认输出格式为 RGB
- [show](mmengine.visualization.Visualizer.show) 可视化
- [get_backend](mmengine.visualization.Visualizer.get_backend) 通过 name 获取特定存储后端
- [close](mmengine.visualization.Visualizer.close) 关闭所有已经打开的资源，包括 VisBackend

关于其用法，可以参考 [可视化器用户教程](../advanced_tutorials/visualization.md)。

## 3 可视化存储后端 VisBackend

在绘制后可以将绘制后的数据存储到多个可视化存储后端中。为了统一接口调用，MMEngine 提供了统一的抽象类 `BaseVisBackend`，和一些常用的 VisBackend 如 `LocalVisBackend`、`WandbVisBackend` 和 `TensorboardVisBackend`。
BaseVisBackend 定义了对外调用的接口规范，主要接口和属性如下：

- [add_config](mmengine.visualization.BaseVisBackend.add_config) 写配置到特定存储后端
- [add_graph](mmengine.visualization.BaseVisBackend.add_graph) 写模型图到特定后端
- [add_image](mmengine.visualization.BaseVisBackend.add_image) 写图片到特定后端
- [add_scalar](mmengine.visualization.BaseVisBackend.add_scalar) 写标量到特定后端
- [add_scalars](mmengine.visualization.BaseVisBackend.add_scalars) 一次性写多个标量到特定后端
- [close](mmengine.visualization.BaseVisBackend.close) 关闭已经打开的资源
- [experiment](mmengine.visualization.BaseVisBackend.experiment) 写后端对象，例如 WandB 对象和 Tensorboard 对象

`BaseVisBackend` 定义了 5 个常见的写数据接口，考虑到某些写后端功能非常强大，例如 WandB，其具备写表格，写视频等等功能，针对这类需求用户可以直接获取 `experiment` 对象，然后调用写后端对象本身的 API 即可。而 `LocalVisBackend`、`WandbVisBackend` 和 `TensorboardVisBackend` 等都是继承自 `BaseVisBackend`，并根据自身特性实现了对应的存储功能。用户也可以继承 `BaseVisBackend` 从而扩展存储后端，实现自定义存储需求。
关于其用法，可以参考 [存储后端用户教程](../advanced_tutorials//visualization.md)。
