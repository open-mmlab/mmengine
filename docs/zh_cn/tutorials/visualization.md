# 可视化 (Visualization)

## 概述

**(1) 总体介绍**

可视化可以给深度学习的模型训练和测试过程提供直观解释。在 OpenMMLab 算法库中，我们期望可视化功能的设计能满足以下需求：

- 提供丰富的开箱即用可视化功能，能够满足大部分计算机视觉可视化任务
- 高扩展性，可视化功能通常多样化，应该能够通过简单扩展实现定制需求
- 能够在训练和测试流程的任意点位进行可视化
- OpenMMLab 各个算法库具有统一可视化接口，利于用户理解和维护

基于上述需求，OpenMMLab 2.0 引入了绘制对象 Visualizer 和写端对象 Writer 的概念

- **Visualizer 负责单张图片的绘制功能**

  MMEngine 提供了以 Matplotlib 库为绘制后端的 `Visualizer` 类，其具备如下功能：

  - 提供了一系列和视觉任务无关的基础方法，例如 `draw_bboxes` 和 `draw_texts` 等
  - 各个基础方法支持链式调用，方便叠加绘制显示
  - 通过 `draw_featmap` 提供绘制特征图功能

  各个下游算法库可以继承 `Visualizer` 并在 `draw` 接口中实现所需的可视化功能，例如 MMDetection 中的 `DetVisualizer` 继承自 `Visualizer` 并在 `draw` 接口中实现可视化检测框、实例掩码和语义分割图等功能。Visualizer 类的 UML 关系图如下

  <div align="center">
   <img src="https://user-images.githubusercontent.com/17425982/154475592-7208a34b-f6cb-4171-b0be-9dbb13306862.png" >
  </div>

- **Writer 负责将各类数据写入到指定后端**

  为了统一接口调用，MMEngine 提供了统一的抽象类 `BaseWriter`，和一些常用的 Writer 如 `LocalWriter` 来支持将数据写入本地，`TensorboardWriter` 来支持将数据写入 Tensorboard，`WandbWriter` 来支持将数据写入 Wandb。用户也可以自定义 Writer 来将数据写入自定义后端。写入的数据可以是图片，模型结构图，标量如模型精度指标等。

  考虑到在训练或者测试过程中可能同时存在多个 Writer 对象，例如同时想进行本地和远程端写数据，为此设计了 `ComposedWriter` 负责管理所有运行中实例化的 Writer 对象，其会自动管理所有 Writer 对象，并遍历调用所有 Writer 对象的方法。Writer 类的 UML 关系图如下
  <div align="center">
   <img src="https://user-images.githubusercontent.com/17425982/157000633-9f552539-f722-44b1-b253-1abaf4a8eba6.png" >
  </div>

**(2) Writer 和 Visualizer 关系**

Writer 对象的核心功能是写各类数据到指定后端中，例如写图片、写模型图、写超参和写模型精度指标等，后端可以指定为本地存储、Wandb 和 Tensorboard 等等。在写图片过程中，通常希望能够将预测结果或者标注结果绘制到图片上，然后再进行写操作，为此在 Writer 内部维护了 Visualizer 对象，将 Visualizer 作为 Writer 的一个属性。需要注意的是：

- 只有调用了 Writer 中的 `add_image` 写图片功能时候才可能会用到 Visualizer 对象，其余接口和 Visualizer 没有关系
- 考虑到某些 Writer 后端本身就具备绘制功能例如 `WandbWriter`，此时 `WandbWriter` 中的 Visualizer 属性就是可选的，如果用户在初始化时候传入了 Visualizer 对象，则在  `add_image` 时候会调用 Visualizer 对象，否则会直接调用 Wandb 本身 API 进行图片绘制
- `LocalWriter` 和 `TensorboardWriter` 由于绘制功能单一，目前强制由 Visualizer 对象绘制，所以这两个 Writer 必须传入 Visualizer 或者子类对象

`WandbWriter` 的一个简略的演示代码如下

```python
# 为了方便理解，没有继承 BaseWriter
class WandbWriter:
    def __init__(self, visualizer=None):
        self._visualizer = None
        if visualizer:
            # 示例配置 visualizer=dict(type='DetVisualizer')
            self._visualizer = VISUALIZERS.build(visualizer)

    @property
    def visualizer(self):
        return self._visualizer

    def add_image(self, name, image, gt_sample=None, pred_sample=None, draw_gt=True, draw_pred=True, step=0, **kwargs):
        if self._visualize:
           self._visualize.draw(image, gt_sample, pred_sample, draw_gt, draw_pred)
           # 调用 Writer API 写图片到后端
           self.wandb.log({name: self.visualizer.get_image()}, ...)
           ...
        else:
           # 调用 Writer API 汇总并写图片到后端
           ...

    def add_scalar(self, name, value, step):
         self.wandb.log({name: value}, ...)
```


## 绘制对象 Visualizer

绘制对象 Visualizer 负责单张图片的各类绘制功能，默认绘制后端为 Matplotlib。为了统一 OpenMMLab 各个算法库的可视化接口，MMEngine 定义提供了基础绘制功能的 `Visualizer` 类，下游库可以继承 `Visualizer` 并实现 `draw` 接口来满足自己的绘制需求。

### Visualizer

`Visualizer` 提供了基础而通用的绘制功能，主要接口如下：

**(1) 绘制无关的功能性接口**

- [set_image](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.set_image) 设置原始图片数据，默认输入图片格式为 RGB
- [get_image](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.get_image) 获取绘制后的 Numpy 格式图片数据，默认输出格式为 RGB
- [show](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.show) 可视化
- [register_task](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.register_task) 注册绘制函数(其作用在 *自定义 Visualizer* 小节描述)

**(2) 绘制相关接口**

- [draw](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw) 用户使用的抽象绘制接口
- [draw_featmap](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_featmap) 绘制特征图
- [draw_bboxes](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_bboxes) 绘制单个或者多个边界框
- [draw_texts](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_texts) 绘制单个或者多个文本框
- [draw_lines](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.lines) 绘制单个或者多个线段
- [draw_circles](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_circles) 绘制单个或者多个圆
- [draw_polygons](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_polygons) 绘制单个或者多个多边形
- [draw_binary_masks](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.Visualizer.draw_binary_mask) 绘制单个或者多个二值掩码

用户除了可以单独调用 `Visualizer` 中基础绘制接口，同时也提供了链式调用功能和特征图可视化功能。`draw` 函数是抽象接口，内部没有任何实现，继承了 Visualizer 的类可以实现该接口，从而对外提供统一的绘制功能，而 `draw_xxx` 等目的是提供最基础的绘制功能，用户一般无需重写。

**(1) 链式调用**

例如用户先绘制边界框，在此基础上绘制文本，绘制线段，则调用过程为：

```python
visualizer.set_image(image)
visualizer.draw_bboxes(...).draw_texts(...).draw_lines(...)
visualizer.show() # 可视化绘制结果
```

**(2) 可视化特征图**

特征图可视化是一个常见的功能，通过调用 `draw_featmap` 可以直接可视化特征图，其参数定义为：

```python
@staticmethod
def draw_featmap(tensor_chw: torch.Tensor, # 输入格式要求为 CHW
                 image: Optional[np.ndarray] = None, # 如果同时输入了 image 数据，则特征图会叠加到 image 上绘制
                 mode: Optional[str] = 'mean', # 多个通道压缩为单通道的策略
                 topk: int = 10, # 可选择激活度最高的 topk 个特征图显示
                 arrangement: Tuple[int, int] = (5, 2), # 多通道展开为多张图时候布局
                 alpha: float = 0.3) -> np.ndarray: # 图片和特征图绘制的叠加比例
```

特征图可视化功能较多，目前不支持 Batch 输入

- mode 不是 None，topk 无效，会将多个通道输出采用 mode 模式函数压缩为单通道，变成单张图片显示，目前 mode 仅支持 None、'mean'、'max' 和 'min' 参数输入
- mode 是 None，topk 有效，如果 topk 不是 -1，则会按照激活度排序选择 topk 个通道显示，此时可以通过 arrangement 参数指定显示的布局
- mode 是 None，topk 有效，如果 `topk = -1`，此时通道 C 必须是 1 或者 3 表示输入数据是图片，可以直接显示，否则报错提示用户应该设置 mode 来压缩通道

```python
featmap=visualizer.draw_featmap(tensor_chw,image)
```

### 自定义 Visualizer

自定义的 Visualizer 中大部分情况下只需要实现 `get_image` 和 `draw` 接口。`draw` 是最高层的用户调用接口，`draw` 接口负责所有绘制功能，例如绘制检测框、检测掩码 mask 和 检测语义分割图等等。依据任务的不同，`draw` 接口实现的复杂度也不同。

以目标检测可视化需求为例，可能需要同时绘制边界框 bbox、掩码 mask 和语义分割图 seg_map，如果如此多功能全部写到 `draw` 方法中会难以理解和维护。为了解决该问题，`Visualizer` 基于 OpenMMLab 2.0 抽象数据接口规范支持了 `register_task` 函数。假设 MMDetection 中需要同时绘制预测结果中的 instances 和 sem_seg，可以在 MMDetection 的 `DetVisualizer` 中实现 `draw_instances` 和 `draw_sem_seg` 两个方法，用于绘制预测实例和预测语义分割图， 我们希望只要输入数据中存在 instances 或 sem_seg 时候，对应的两个绘制函数  `draw_instances` 和 `draw_sem_seg` 能够自动被调用，而用户不需要手动调用。为了实现上述功能，可以通过在 `draw_instances` 和 `draw_sem_seg` 两个函数加上 `@Visualizer.register_task` 装饰器，此时 `task_dict` 中就会存储字符串和函数的映射关系，在调用 `draw` 方法时候就可以通过 `self.task_dict`获取到已经被注册的函数。一个简略的实现如下所示

```python
class DetVisualizer(Visualizer):

    def draw(self, image, gt_sample=None, pred_sample=None, draw_gt=True, draw_pred=True):
        # 将图片和 matplotlib 布局关联
        self.set_image(image)

        if draw_gt:
            # self.task_dict 内部存储如下信息：
            # dict(instances=draw_instance 方法,sem_seg=draw_sem_seg 方法)
            for task in self.task_dict:
                task_attr = 'gt_' + task
                if task_attr in gt_sample:
                    self.task_dict[task](self, gt_sample[task_attr], 'gt')
        if draw_pred:
            for task in self.task_dict:
                task_attr = 'pred_' + task
                if task_attr in pred_sample:
                    self.task_dict[task](self, pred_sample[task_attr], 'pred')

    # data_type 用于区分当前绘制的内容是标注还是预测结果
    @Visualizer.register_task('instances')
    def draw_instance(self, instances, data_type):
        ...

    # data_type 用于区分当前绘制的内容是标注还是预测结果
    @Visualizer.register_task('sem_seg')
    def draw_sem_seg(self, pixel_data, data_type):
        ...
```

注意：是否使用 `register_task` 装饰器函数不是必须的，如果用户自定义 Visualizer，并且 `draw` 实现非常简单，则无需考虑 `register_task`。

在使用 Jupyter notebook 或者其他地方不需要写数据到指定后端的情形下，用户可以自己实例化 visualizer。一个简单的例子如下

```python
# 实例化 visualizer
visualizer=dict(type='DetVisualizer')
visualizer = VISUALIZERS.build(visualizer)
visualizer.draw(image, datasample)
visualizer.show() # 可视化绘制结果
```

## 写端 Writer

Visualizer 只实现了单张图片的绘制功能，但是在训练或者测试过程中，对一些关键指标或者模型训练超参的记录非常重要，此功能通过写端 Writer 实现。为了统一接口调用，MMEngine 提供了统一的抽象类 `BaseWriter`，和一些常用的 Writer 如 `LocalWriter` 、`TensorboardWriter`  和 `WandbWriter` 。

### BaseWriter

BaseWriter 定义了对外调用的接口规范，主要接口和属性如下：

- [add_config](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_config) 写超参到特定后端，常见的训练超参如初始学习率 LR、权重衰减系数和批大小等等
- [add_graph](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_graph) 写模型图到特定后端
- [add_image](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_image) 写图片到特定后端
- [add_scalar](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_scalar) 写标量到特定后端
- [add_scalars](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.add_scalars) 一次性写多个标量到特定后端
- [visualizer](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.visualizer) 绘制对象
- [experiment](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.BaseWriter.experiment) 写后端对象，例如 Wandb 对象和 Tensorboard 对象

`BaseWriter` 定义了 5 个常见的写数据接口，考虑到某些写后端功能非常强大，例如 Wandb，其具备写表格，写视频等等功能，针对这类需求用户可以直接获取 experiment 对象，然后调用写后端对象本身的 API 即可。

### LocalWriter、TensorboardWriter 和 WandbWriter

`LocalWriter` 提供了将数据写入到本地磁盘功能。如果用户需要写图片到硬盘，则**必须要通过初始化参数提供 Visualizer对象**。其典型用法为：

```python
# 配置文件
writer=dict(type='LocalWriter', save_dir='demo_dir', visualizer=dict(type='DetVisualizer'))
# 实例化和调用
local_writer=WRITERS.build(writer)
# 写模型精度值
local_writer.add_scalar('mAP', 0.9)
local_writer.add_scalars({'loss': 1.2, 'acc': 0.8})
# 写超参
local_writer.add_config(dict(lr=0.1, mode='linear'))
# 写图片
local_writer.add_image('demo_image', image, datasample)
```

如果用户有自定义绘制需求，则可以通过获取内部的 visualizer 属性来实现，如下所示

```python
# 配置文件
writer=dict(type='LocalWriter', save_dir='demo_dir', visualizer=dict(type='DetVisualizer'))
# 实例化和调用
local_writer=WRITERS.build(writer)
# 写图片
local_writer.visualizer.draw_bboxes(np.array([0, 0, 1, 1]))
local_writer.add_image('img', local_writer.visualizer.get_image())

# 绘制特征图并保存到本地
featmap_image=local_writer.visualizer.draw_featmap(tensor_chw)
local_writer.add_image('featmap', featmap_image)
```

`TensorboardWriter` 提供了将各类数据写入到 Tensorboard 功能，其用法和 LocalWriter 非常类似。 注意如果用户需要写图片到 Tensorboard，则**必须要通过初始化参数提供 Visualizer对象**。

`WandbWriter` 提供了将各类数据写入到 Wandb 功能。考虑到 Wandb 本身具备强大的图片功能，在调用 `WandbWriter` 的 `add_image` 方法时 Visualizer 对象是可选的，如果用户指定了 Visualizer 对象，则会调用  Visualizer 对象的绘制方法，否则直接调用 Wandb 自带的图片处理功能。

## 组合写端 ComposedWriter

考虑到在训练或者测试过程中，可能需要同时调用多个 Writer，例如想同时写到本地和 Wandb 端，为此设计了对外的 `ComposedWriter` 类，在训练或者测试过程中 `ComposedWriter` 会依次调用各个 Writer 的接口，其接口和 `BaseWriter` 一致，主要接口如下：

- [add_config](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.ComposedWriter.add_config) 写超参到所有已经加入的后端中，常见的训练超参如初始学习率 LR、权重衰减系数和批大小等等
- [add_graph](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.ComposedWriter.add_graph) 写模型图到所有已经加入的后端中
- [add_image](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.ComposedWriter.add_image) 写图片到所有已经加入的后端中
- [add_scalar](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.ComposedWriter.add_scalar) 写标量到所有已经加入的后端中
- [add_scalars](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.ComposedWriter.add_scalars) 一次性写多个标量到所有已经加入的后端中
- [get_writer](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.ComposedWriter.get_writer) 获取指定索引的 Writer，任何一个 Writer 中包括了 experiment 和 visualizer 属性
- [get_experiment](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.ComposedWriter.get_experiment) 获取指定索引的 experiment
- [get_visualizer](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.ComposedWriter.get_visualizer) 获取指定索引的 visualizer
- [close](https://mmengine.readthedocs.io/zh/latest/api.html#mmengine.visualization.writer.ComposedWriter.close) 调用所有 Writer 的 close 方法

为了让用户可以在代码的任意位置进行数据可视化，`ComposedWriter` 类继承至 [全局可访问基类 BaseGlobalAccessible](./logging.md/#全局可访问基类baseglobalaccessible)。一旦继承了全局可访问基类, 用户就可以通过调用 `ComposedWriter` 对象的 `get_instance` 来获取全局对象。其基本用法如下

```python
# 创建实例
writers=[dict(type='LocalWriter', save_dir='temp_dir', visualizer=dict(type='DetVisualizer')), dict(type='WandbWriter')]

ComposedWriter.create_instance('composed_writer', writers=writers)
```

一旦创建实例后，可以在代码任意位置获取 `ComposedWriter` 对象

```python
composed_writer=ComposedWriter.get_instance('composed_writer')

# 写模型精度值
composed_writer.add_scalar('mAP', 0.9)
composed_writer.add_scalars({'loss': 1.2, 'acc': 0.8})
# 写超参
composed_writer.add_config(dict(lr=0.1, mode='linear'))
# 写图片
composed_writer.add_image('demo_image', image, datasample)
# 写模型图
composed_writer.add_graph(model, input_array)
```

对于一些用户需要的自定义绘制需求或者上述接口无法满足的需求，用户可以通过 `get_xxx` 方法获取具体对象来实现特定需求

```python
composed_writer=ComposedWriter.get_instance('composed_writer')

# 绘制特征图，获取 LocalWriter 中的 visualizer
visualizer=composed_writer.get_visualizer(0)
featmap_image=visualizer.draw_featmap(tensor_chw)
composed_writer.add_image('featmap', featmap_image)

# 扩展 add 功能，例如利用 Wandb 对象绘制表格
wandb=composed_writer.get_experiment(1)
val_table = wandb.Table(data=my_data, columns=column_names)
wandb.log({'my_val_table': val_table})

# 配置中存在多个 Writer，在不想改动配置情况下只使用 LocalWriter
local_writer=composed_writer.get_writer(0)
local_writer.add_image('demo_image', image, datasample)
```
