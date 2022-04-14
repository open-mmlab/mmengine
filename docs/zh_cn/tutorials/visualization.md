# 可视化 (Visualization)

## 概述

可视化可以给深度学习的模型训练和测试过程提供直观解释。在 OpenMMLab 算法库中，我们期望可视化功能的设计能满足以下需求：

- 提供丰富的开箱即用可视化功能，能够满足大部分计算机视觉可视化任务
- 高扩展性，可视化功能通常多样化，应该能够通过简单扩展实现定制需求
- 能够在训练和测试流程的任意点位进行可视化
- OpenMMLab 各个算法库具有统一可视化接口，利于用户理解和维护

基于上述需求，OpenMMLab 2.0 引入了可视化对象 Visualizer 和各个可视化存储后端 VisBackend 如 `LocalVisBackend`、`WandbVisBackend` 和 `TensorboardVisBackend` 等。此处的可视化不仅仅包括图片数据格式，还包括配置内容、标量和模型图等数据的可视化。

- 为了方便调用，Visualizer 提供了包括存储的全部接口, 可视化存储后端 VisBackend 作为 Visualizer 的内部属性。在通常情况下用户直接调用 Visualizer 中提供的方法即可，无需手动调用存储后端
- 考虑到绘制后会希望存储到多个后端，Visualizer 可以配置多个 VisBackend，当用户调用 Visualizer 的存储接口时候，Visualizer 内部会遍历的调用 VisBackend 存储接口

两者的 UML 关系图如下

<div align="center">
 <img src="https://user-images.githubusercontent.com/17425982/163327736-f7cb3b16-ef07-46bc-982a-3cc7495e6c82.png" >
</div>
## 可视化对象 Visualizer

### 接口说明

可视化对象 Visualizer 对外提供了所有接口。可以将其接口分成 3 大类，如下所示

**(1) 绘制相关接口**

- [draw_bboxes](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_bboxes) 绘制单个或多个边界框
- [draw_points](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_points) 绘制单个或多个点
- [draw_texts](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_texts) 绘制单个或多个文本框
- [draw_lines](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.lines) 绘制单个或多个线段
- [draw_circles](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_circles) 绘制单个或多个圆
- [draw_polygons](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_polygons) 绘制单个或多个多边形
- [draw_binary_masks](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_binary_mask) 绘制单个或多个二值掩码
- [draw_featmap](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_featmap) 绘制特征图，静态方法

以 draw 开头的接口表示绘制接口，默认都是用 matplotlib 绘制。除了 `draw_featmap` 外都可以链式调用，原因是该方法调用后可能会导致图片尺寸改变而无法保证链式调用的可行性。为了避免给用户带来困扰，将其设置为静态方法。

当用户想先绘制边界框，在此基础上绘制文本，绘制线段的时候，可以通过链式调用实现：

```python
visualizer.set_image(image)
visualizer.draw_bboxes(...).draw_texts(...).draw_lines(...)
visualizer.show() # 可视化绘制结果
```

特征图可视化是一个常见的功能，用户通过调用 `draw_featmap` 可视化特征图，其参数定义为：

```python
@staticmethod
def draw_featmap(featmap: torch.Tensor, # 输入格式要求为 CHW
                 overlaid_image: Optional[np.ndarray] = None, # 如果同时输入了 image 数据，则特征图会叠加到 image 上绘制
                 overlaid_bboxes: Optional[np.ndarray] = None, # 如果设置 bbox，则会同时绘制 bbox
                 channel_reduction: Optional[str] = 'squeeze_mean', # 多个通道压缩为单通道的策略
                 topk: int = 10, # 可选择激活度最高的 topk 个特征图显示
                 arrangement: Tuple[int, int] = (5, 2), # 多通道展开为多张图时候布局
                 resize_shape：Optional[tuple] = None, # 可以指定 resize_shape 参数来缩放特征图
                 alpha: float = 0.3) -> np.ndarray: # 图片和特征图绘制的叠加比例
```

特征图可视化功能较多，目前不支持 Batch 输入，其功能可以归纳如下

- 输入的 Tensor 一般是包括多个通道的，channel_reduction 参数可以将多个通道压缩为单通道，然后和图片或者 bboxes 进行叠加显示
  - `squeeze_mean` 将输入的 C 维度采用 mean 函数压缩为一个通道，输出维度变成 (1,H,W)
  - `select_max` 从输入的 C 维度中先在空间维度 sum，维度变成 (C,)，然后选择值最大的通道
  - `None` 表示不需要压缩，此时可以通过 topk 参数可选择激活度最高的 topk 个特征图显示

- 在 channel_reduction 参数为 None 的情况下，topk 参数生效，其会按照激活度排序选择 topk 个通道，然后和图片或者 bboxes 进行叠加显示，并且此时会通过 arrangement 参数指定显示的布局
  - 如果 topk 不是 -1，则会按照激活度排序选择 topk 个通道显示
  - 如果 topk = -1，此时通道 C 必须是 1 或者 3 表示输入数据是图片，否则报错提示用户应该设置 `channel_reduction`来压缩通道。

- 考虑到输入的特征图通常非常小，因此用户可以额外输入 `resize_shape` 参数，将特征图进行上采样进行可视化。需要注意的是**当 image 和 featmap 叠加显示时候，两者空间尺寸必须相同**

**(2) 存储相关接口**

- [add_config](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_config) 写配置到特定存储后端
- [add_graph](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_graph) 写模型图到特定存储后端
- [add_image](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_image) 写图片到特定存储后端
- [add_scalar](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_scalar) 写标量到特定存储后端
- [add_scalars](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_scalars) 一次性写多个标量到特定存储后端
- [add_datasample](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_datasample) 各个下游库绘制 datasample 数据的抽象接口

以 add 前缀开头的接口表示存储接口。datasample 是 OpenMMLab 2.0 架构中设计的各个下游库统一的抽象数据接口，而 `add_datasample` 接口可以直接处理该数据格式，例如可视化预测结果、可视化 Dataset 或者 DataLoader 输出、可视化中间预测结果等等都可以直接调用下游库重写的 `add_datasample` 接口。

所有下游库都必须要继承 Visualizer 并实现 `add_datasample` 接口。以 MMDetection 为例，应该继承并通过该接口实现目标检测中所有预置任务的可视化功能，例如目标检测、实例分割、全景分割任务结果的绘制和存储。

**(3) 其余功能性接口**

- [set_image](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.set_image) 设置原始图片数据，默认输入图片格式为 RGB
- [get_image](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.get_image) 获取绘制后的 Numpy 格式图片数据，默认输出格式为 RGB
- [show](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.show) 可视化
- [get_backend](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.get_backend) 通过 name 获取特定存储后端
- [close](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.close) 关闭所有已经打开的资源，包括 VisBackend

**为了确保可视化对象 Visualizer 能够在任何地方被调用，将其继承自 ManagerMixin 类，从而转变为全局唯一对象，在实例化必须要通过 `visualizer.get_instance()` 方式来初始化才能具备全局唯一功能。一旦初始化后可以在任意代码位置通过 `Visualizer.get_current_instance()` 来获取可视化对象。**

### 使用样例

以 MMDetection 为例，假设 `DetLocalVisualizer` 类继承自 `Visualizer`，并实现了 `add_datasample` 接口。配置文件写法为：

```python
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# 内部会调用 get_instance() 进行全局唯一实例化
VISUALIZERS.build(cfg.visualizer)

# 任意代码位置获取 visualizer
visualizer = Visualizer.get_current_instance()

# 绘制 datasample，并保存到本地存储后端
visualizer.add_datasample('demo_image',image, gt_sample, pred_sample, step=1)
# 直接本地窗口显示，而无需存储
visualizer.add_datasample('demo_image',image, gt_sample, pred_sample, show=True)

# 写图片
visualizer.add_image('demo_image', image, datasample, step=1)
# 绘制特征图并保存
featmap_image = visualizer.draw_featmap(tensor_chw)
visualizer.add_image('featmap', featmap_image)

# 写模型精度值
visualizer.add_scalar('mAP', 0.9, step=1)
visualizer.add_scalars({'loss': 1.2, 'acc': 0.8}, step=1)

# 写配置文件
visualizer.add_config(cfg)

# 写模型图
visualizer.add_graph(model, data_batch)
```

对于一些用户需要的自定义绘制需求或者上述接口无法满足的需求，用户可以通过 `get_backend` 方法获取具体对象来实现特定需求，假设同时存储多个存储后端，配置为：

```python
vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
```

```python
# 内部会调用 get_instance() 进行全局唯一实例化
VISUALIZERS.build(cfg.visualizer)
# 任意代码位置获取 visualizer
visualizer = Visualizer.get_current_instance()

# 扩展 add 功能，例如利用 Wandb 对象绘制表格
wandb = visualizer.get_backend('WandbVisBackend').experiment
val_table = wandb.Table(data=my_data, columns=column_names)
wandb.log({'my_val_table': val_table})
```

假设用户想使用 Wandb 优异的前端绘制功能，则可以使用 DetWandbVisualizer，典型配置如下

```python
vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]
visualizer = dict(
    type='DetWandbVisualizer', vis_backends=vis_backends, name='visualizer')
```

注意：由于 Wandb 绘制的数据无法和 `LocalVisBackend` 后端兼容，所以当 `vis_backends` 存在多个可视化存储后端时候只有 `WandbVisBackend` 才是有效的，用法和上面完全一致。

## 可视化存储后端 VisBackend

在绘制后可以将绘制后的数据存储到多个可视化存储后端中。为了统一接口调用，MMEngine 提供了统一的抽象类 `BaseVisBackend`，和一些常用的 VisBackend 如 `LocalVisBackend`、`WandbVisBackend` 和 `TensorboardVisBackend`。

BaseVisBackend 定义了对外调用的接口规范，主要接口和属性如下：

- [add_config](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.vis_backend.BaseVisBackend.add_config) 写配置到特定存储后端
- [add_graph](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.vis_backend.BaseVisBackend.add_graph) 写模型图到特定后端
- [add_image](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.vis_backend.BaseVisBackend.add_image) 写图片到特定后端
- [add_scalar](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.vis_backend.BaseVisBackend.add_scalar) 写标量到特定后端
- [add_scalars](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.vis_backend.BaseVisBackend.add_scalars) 一次性写多个标量到特定后端
- [close](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.vis_backend.BaseVisBackend.close) 关闭已经打开的资源
- [experiment](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.vis_backend.BaseVisBackend.experiment) 写后端对象，例如 Wandb 对象和 Tensorboard 对象

`BaseVisBackend` 定义了 5 个常见的写数据接口，考虑到某些写后端功能非常强大，例如 Wandb，其具备写表格，写视频等等功能，针对这类需求用户可以直接获取 experiment 对象，然后调用写后端对象本身的 API 即可。而 `LocalVisBackend`、`WandbVisBackend` 和 `TensorboardVisBackend` 等都是继承自 `BaseVisBackend`，并根据自身特性实现了对应的存储功能。
