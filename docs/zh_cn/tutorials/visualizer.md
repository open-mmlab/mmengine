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

  - 提供了一系列和视觉任务无关的基础方法，例如  `draw_bboxes` 和 `draw_texts` 等
  - 上述各个基础方法支持链式调用，方便叠加绘制显示
  - 提供了绘制特征图功能

  各个下游算法库可以继承 `Visualizer` 并在 `draw` 接口实现所需的可视化功能，例如 MMDetection 中的 `DetVisualizer` 继承至 `Visualizer` 并在 `draw` 接口实现可视化检测框、掩码 mask 和语义分割图等功能。Visualizer 类的 UML 关系图如下

  <div align="center">
   <img src="https://user-images.githubusercontent.com/17425982/154475592-7208a34b-f6cb-4171-b0be-9dbb13306862.png" >
  </div>

- **Writer 负责将各类数据写入到指定后端**

  为了统一接口调用，MMEngine 提供了统一的抽象类 `BaseWriter`，和一些常用的 Writer 如 `LocalWriter` 来支持将数据写入本地，`TensorboardWriter` 来支持将数据写入 Tensorboard，`WandbWriter` 来支持将数据写入 Wandb。用户也可以自定义 Writer 来将数据写入自定义后端。写入的数据可以是图片，模型结构图，标量如模型精度指标等。

  考虑到在训练或者测试过程中同时存在多个  Writer 对象，例如同时想进行本地和远程端写数据，为此设计了 `ComposedWriter`  负责管理所有运行中实例化的 Writer 对象，其会自动管理所有 Writer 对象，并遍历调用所有 Writer 对象的方法。Writer 类的 UML 关系图如下
<div align="center">
 <img src="https://user-images.githubusercontent.com/17425982/154474755-080b955b-436b-4cdb-9a49-16a9f231ce81.png" >
</div>

**(2) Writer 和 Visualizer 关系**

Writer 对象的核心功能是写各类数据到指定后端中，例如写图片、写模型图、写超参和写模型精度指标等，后端可以指定为本地存储、Wandb 和 Tensorboard 等等。在写图片过程中，通常希望能够将预测结果或者标注结果绘制到图片上，然后再进行写操作，为此在 Writer 内部维护了 Visualizer 对象，将 Visualizer 作为 Writer 的一个属性。当需要利用 Visualizer 对象来绘制结果到图片上时候，可以通过调用 Writer 的 Visualizer 属性对象进行绘制。一个简略的演示代码如下

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

    def add_image(self, name, image, datasample=None, step=0, **kwargs):
        if self._visualize:
           self._visualize.draw(image, datasample)
           # 调用 Writer API 写图片到后端
           self.wandb.log({name: self.visualizer.get_image()}, ...)
           ...
        else:
           # 调用 Writer API 汇总并写图片到后端
           ...

    def add_scaler(self, name, value, step):
         self.wandb.log({name: value}, ...)
```

对于非   `LocalWriter`  或者不需要调用写图片的 `add_image` 接口需求场景，visualizer 参数可以为 None。

注意 `Visualizer` 仅仅有单图绘制功能，如果想将绘制结果保存，例如保存到本地、Wandb 或者 Tensorboard，可以使用 Writer 写端对象。一个简单的例子如下

```python
visualizer=dict(type='DetVisualizer')
visualizer = VISUALIZERS.build(visualizer)
visualizer.draw(image, datasample)

# 保存到本地
# 直接实例化
local_writer=LocalWriter(save_dir='demo_dir')
# 绘制通过配置实例化
local_writer=WRITERS.build(dict(type='LocalWriter',save_dir='demo_dir'))
local_writer.add_image('demo_image',visualizer.get_image())

# 保存到 Wandb
wandb_writer=WandbWriter()
wandb_writer.add_image('demo_image',visualizer.get_image())

# 保存到 Tensorboard
tensorboard_writer=TensorboardWriter()
tensorboard_writer.add_image('demo_image',visualizer.get_image())
```

考虑到用户需要自己实例化 Visualizer 和 Writer，步骤较多，不推荐这种调用方法，推荐做法如下

```python
# 配置文件
writer=dict(type='LocalWriter',save_dir='demo_dir'，visualizer=dict(type='DetVisualizer'))
# 实例化和调用
writer_obj=WRITERS.build(writer)
writer_obj.add_image('demo_image',image, datasample)
```


## 绘制对象 Visualizer

绘制对象 Visualizer 负责单张图片的各类绘制功能，默认绘制后端为 Matplotlib。为了统一 OpenMMLab 各个算法库的可视化接口，MMEngine 定义提供了基于基础绘制功能的 `Visualizer` 类，下游库可以继承 `Visualizer` 并实现 `draw` 接口实现自己的可视化需求，例如 MMDetection 的 [`DetVisualizer`]()。

### Visualizer

`Visualizer` 提供了基础而通用的绘制功能，主要接口如下：

**(1) 绘制无关的功能性接口**

- set_image  设置原始图片数据
- get_image 获取绘制后的 Numpy 格式图片数据
- show  可视化
- register_task  注册绘制函数(其作用在 *自定义 Visualizer* 小节描述)

**(2) 绘制相关接口**

- draw  用户使用的抽象绘制接口
- draw_featmap  绘制特征图
- draw_bboxes  绘制单个或者多个边界框
- draw_texts  绘制单个或者多个文本框
- draw_lines  绘制单个或者多个线段
- draw_circles  绘制单个或者多个圆
- draw_polygons  绘制单个或者多个多边形
- draw_binary_masks  绘制单个或者多个二值掩码

**(1) 用例 1 - 链式调用**

例如用户先绘制边界框，在此基础上绘制文本，绘制线段，则调用过程为：

```python
visualizer.set_image(image)
visualizer.draw_bboxes(...).draw_texts(...).draw_lines(...)
```

**(2) 用例 2 - 可视化特征图**

特征图可视化是一个常见的功能，通过调用 `draw_featmap` 可以直接可视化特征图，目前该函数支持如下功能：

- 输入 4 维 BCHW 格式的 tensor，通道 C 是 1 或者 3 时候，展开成一张图片显示
- 输入 4 维 BCHW 格式的 tensor，通道 C 大于 3 时候，则支持选择激活度最高通道，展开成一张图片显示
- 输入 3 维 CHW 格式的 tensor，则选择激活度最高的 topk，然后拼接成一张图显示

```python
# 如果提前设置了图片，则特征图或者图片叠加显示，否则只显示特征图
visualizer.set_image(image)
visualizer.draw_featmap(...)
visualizer.save(...)
```

### 自定义 Visualizer

自定义的 Visualizer 中大部分情况下只需要实现 `get_image` 和 `draw` 接口。`draw` 是最高层的用户调用接口，`draw` 接口负责所有绘制功能，例如绘制检测框、检测掩码 mask 和 检测语义分割图等等。依据任务的不同，`draw` 接口实现的复杂度也不同。

以目标检测可视化需求为例，可能需要同时绘制边界框 bbox、掩码 mask 和语义分割图 seg_map，如果如此多功能全部写到 `draw` 方法中会难以理解和维护。为了解决该问题，`Visualizer` 基于 OpenMMLab 2.0 抽象数据接口规范支持了 `register_task` 函数。假设 MMDetection 中需要同时绘制预测结果中的 instances 和 sem_seg，可以在 MMDetection 的 `DetVisualizer` 中实现 `draw_instances` 和 `draw_sem_seg` 两个方法，用于绘制预测实例和预测语义分割图， 我们希望只要输入数据中存在 instances 或 sem_seg 时候，对应的两个绘制函数  `draw_instances` 和 `draw_sem_seg` 能够自动被调用，而用户不需要手动调用。为了实现上述功能，可以通过在 `draw_instances` 和 `draw_sem_seg` 两个函数加上 `@Visualizer.register_task` 装饰器。

```python
class DetVisualizer(Visualizer):

    def get_image(self):
        ...

    def draw(self,data_sample, image=None,show_gt=True, show_pred=True):
        if show_gt:
            for task in self.task_dict:
                task_attr = 'gt_' + task
                if task_attr in data_sample:
                    self.task_dict[task](self, data_sample[task_attr], DataType.GT)
        if show_pred:
            for task in self.task_dict:
                task_attr = 'pred_' + task
                if task_attr in data_sample:
                    self.task_dict[task](self, data_sample[task_attr], DataType.PRED)

    @Visualizer.register_task('instances')
    def draw_instance(self, instances, data_type):
        ...

    @Visualizer.register_task('sem_seg')
    def draw_sem_seg(self, pixel_data, data_type):
        ...
```

注意：是否使用 `register_task` 装饰器函数不是必须的，如果用户自定义 Visualizer，并且 `draw `实现非常简单，则无需考虑 `register_task`。

如果想使用 `DetVisualizer`，用户可以直接在 Python 代码中实例化，代码如下

```python
det_local_visualizer=DetVisualizer()
det_local_visualizer.draw(data_sample,image)
```

用户也可以使用注册器实例化，配置如下

```python
visualizer= dict(type='DetVisualizer')
det_local_visualizer=build_visualizer(visualizer)
det_local_visualizer.draw(data_sample,image)
```

## 写端 Writer

Visualizer 只是实现了单张图片的可视化功能，但是在训练或者测试过程中，对一些关键指标或者模型训练超参的记录非常重要，此功能通过写端 Writer 实现。

BaseWriter 定义了对外调用的接口规范，主要接口如下：

- add_hyperparams  写超参，常见的训练超参如初始学习率 LR、权重衰减系数和批大小等等
- add_image 写图片
- add_scalar 写标量
- add_graph 写模型图
- visualizer 绘制对象，可以为 None
- experiment 写后端对象，例如 Wandb 对象和 Tensorboard 对象

`BaseWriter` 定义了 4 种常见的写数据接口，考虑到某些写后端功能非常强大，例如 Wandb，其具备写表格，写视频等等功能，针对这类需求用户可以直接获取 experiment 对象，然后调用写后端对象本身的 API 即可。

由于 Visualizer 和 Writer 对象是解耦的，用户可以通过配置文件自由组合各种 Visualizer 和 Writer，例如 `WandbWriter` 绑定 `Visualizer`，表示图片上绘制结果功能由 `Visualizer` 提供，但是最终图片是写到了 Wandb 端，一个简单的例子如下所示

```python
# 配置文件
writer=dict(type='LocalWriter',save_dir='demo_dir'，visualizer=dict(type='DetVisualizer'))
# 实例化和调用
writer_obj=WRITERS.build(writer)
# 写图片
writer_obj.add_image('demo_image',image, datasample)
# 写模型精度值
writer_obj.add_scalar('mAP',0.9)
```

## 组合写端 ComposedWriter

考虑到在训练或者测试过程中，可能需要同时调用多个 Writer，例如想同时写到本地和 Wandb 端，为此设计了对外的 `ComposedWriter` 类，在训练或者测试过程中  `ComposedWriter` 会依次调用各个 Writer，主要接口如下：

- add_hyperparams  写超参，常见的训练超参如初始学习率 LR、权重衰减系数和批大小等等
- add_image 写图片
- add_scalar 写标量
- add_graph 写模型图
- setup_env 设置 work_dir 等必备的环境变量
- get_writer 获取某个 writer
- `__enter__` 上下文进入函数
- `__exit__` 上下文推出函数

为了让用户可以在代码的任意位置进行数据可视化，`ComposedWriter` 类实现 `__enter__` 和 ` __exit__`方法，并且在 `Runner` 中使上下文生效，从而在该上下文作用域内，用户可以通过 `get_writers` 工具函数获取 `ComposedWriter` 类实例，从而调用该类的各种可视化和写方法。一个简单粗略的实现和用例如下

```python
# 假设 writer 只有一个
visualizer=build_visualizer(cfg.visualizer,VISUALIZERS)
writer =build_writer(cfg.writer,WRITERS)
writer.bind_visualizer(visualizer)

# 假设在 epoch 训练过程中
with ComposedWriter(writer):
    while self.epoch < self._max_epochs:
         for i, flow in enumerate(workflow):
            ...
```

```python
# 配置文件写法
writer = dict(type='WandbWriter',init_kwargs=dict(project='demo'),
              visualizer= dict(type='DetLocalVisualizer', show=False))

# 在上下文作用域生效的任意位置
composed_writer=get_writers()
composed_writer.add_image('vis_image',image, datasample, iter=iter)
composed_writer.add_scalar('mAP', val, iter=iter)
```

在训练和测试过程中，用户可以在上下文生效的代码任意位置通过调用 `get_writers()` 获得 ComposedWriter 对象，然后通过该对象可以进行绘制或者写操作。
