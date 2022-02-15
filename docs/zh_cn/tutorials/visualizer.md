# 可视化 (Visualizer)

可视化可以给深度学习的模型训练和测试过程提供直观解释。在 OpenMMLab 算法库的早期设计中，可视化功能由一个个独立的函数实现，例如 `imshow_bboxes` 和 `imshow_det_bboxes`，该设计存在的主要问题可以总结为：

- 扩展性不足，功能单一，无法通过扩展实现定制可视化需求
- OpenMMLab 各个算法库没有统一可视化接口，不利于理解和维护
- 难以在训练和测试流程的任意点位进行可视化

为了解决上述问题，OpenMMLab 2.0 引入了可视化对象 Visualizer 和写端对象 Writer 的概念

- **Visualizer** 负责单张图片的各类绘制和可视化功能。MMEngine 提供了统一接口的抽象类 BaseVisualizer，各个下游算法库可以继承 BaseVisualizer 实现所需的可视化功能，例如 DetLocalVisualizer 实现检测相关的本地可视化功能，DetWandbVisualizer 实现检测相关的远程 Wandb 可视化功能
- **Writer** 负责将各类数据写入到指定后端，写入的数据可以是图片，模型结构图，也可以是标量例如模型精度指标。MMEngine 提供了统一接口的抽象类 BaseWriter，和一些常用的 Writer 如 LocalWriter 来支持将数据写入本地，TensorboardWriter 来支持将数据写入 Tensorboard，WandbWriter 来支持将数据写入 Wandb。用户也可以自定义 Writer 来将数据写入自定义后端。
- **RuntimeWriter** 负责管理所有运行中实例化的 Writer 对象。假设训练或者测试过程中同时存在多个  Writer 对象，RuntimeWriter 会自动管理所有 Writer 对象，并遍历调用所有 Writer 对象的方法

Visualizer、Writer 和 RuntimeWriter 三者联系如下图所示：

![Visualizer](https://user-images.githubusercontent.com/17425982/153836473-d6e1708d-20b8-433e-9fd7-880bfb4e42bf.png)

![Writer and RuntimeWriter](https://user-images.githubusercontent.com/17425982/153995219-fa3d57a2-83bc-490c-a16d-b9539dc3a030.png)

在训练或者测试过程中，常用的可视化流程为

- 利用 Visualizer 对象对当前图片进行绘制，例如绘制边界框 bbox 和 掩码 mask 等等
- 将 Visualizer 绘制后的图片传递给 Writer 对象，由 Writer 对象负责写入到指定后端
- 如果想可视化训练或者测试过程中的曲线，例如模型精度指标曲线，可以直接利用 Writer 对象接口实现，无需使用 Visualizer

在使用本地主机进行可视化的情况下，如用户单独写脚本可视化图片或者使用 jupyter notebook 时，用户只需要实例化 Visualizer 对象即可。用户可以使用 Visualizer 对当前图片进行绘制，然后直接调用 Visualizer 的 save/show 等接口对图片进行保存/显示。

## 可视化对象 Visualizer

可视化对象 Visualizer 负责单张图片的各类绘制和可视化功能。为了统一 OpenMMLab 各个算法库的可视化接口，MMEngine 定义了 Visualizer 的基类 BaseVisualizer，下游库可以继承 BaseVisualizer，实现自己的可视化需求，例如 MMDetection 的 DetLocalVisualizer、DetTensorboardVisualizer、DetWandbVisualizer，用于进行本地端可视化、Tensorboard 端可视化和 Wandb 端可视化等等。

### BaseVisualizer

BaseVisualizer 提供了基础而通用的可视化功能，主要接口如下：

**(1) 绘制无关的功能性接口**

- set_image  设置原始图片数据
- get_image 获取绘制后的图片数据
- save 保存图片
- show  可视化
- register_task  注册绘制函数

**(2) 绘制相关基础接口**

- draw  用户使用的抽象绘制接口
- draw_bbox  绘制边界框
- draw_text  绘制文本框
- draw_line  绘制线段
- draw_circle  绘制圆
- draw_polygon  绘制多边形
- draw_binary_mask  绘制二值掩码
- draw_featmap  绘制特征图

前面说过，Visualizer 接受的数据除了 image，还包括符合抽象数据接口规范的抽象数据封装。假设 MMDetection 中需要同时可视化预测结果中的 instances 和 sem_seg，可以在 MMDetection 中实现 `draw_instances` 和 `draw_sem_seg` 两个方法，用于绘制预测实例和预测语义分割图， 我们希望只要输入数据中存在 instances 或 sem_seg 时候，对应的两个绘制函数  `draw_instances` 和 `draw_sem_seg` 能够自动被调用，而用户不需要手动调用。为了实现上述功能，可以通过在 `draw_instances` 和 `draw_sem_seg` 两个函数加上 `@BaseVisualizer.register_task` 装饰器。

```python
class DetLocalVisualizer(BaseVisualizer):

    @BaseVisualizer.register_task('instances')
    def draw_instance(self, instances, data_type):
        ...

    @BaseVisualizer.register_task('sem_seg')
    def draw_sem_seg(self, pixel_data, data_type):
        ...
```

除了常见的 draw_bbox 等可视化功能外，Visualizer 还提供了两个实用功能：

- **支持用户自定义组合调用进行可视化**

  例如用户可以先绘制边界框，在此基础上绘制文本，绘制线段，最后保存起来，则调用过程为：

  ```python
  visualizer.draw_bbox(...).draw_text(...).draw_line(...).save()
  ```

  所有的 draw_xx 函数返回都是对象本身，用户可以自由组合。

- **特征图可视化**

   特征图可视化是一个常见的功能，通过调用 `draw_featmap` 可以直接可视化特征图，目前该函数支持如下功能：

  - 输入 batch tensor，通道是 1 或者 3 时候，展开成一张图片显示
  - 输入 batch tensor，通道大于 3 时候，则支持选择激活度最高通道，展开成一张图片显示
  - 输入 3 维 tensor，则选择激活度最高的 topk，然后拼接成一张图显示

### 自定义 Visualizer

自定义的 Visualizer 中大部分情况下只需要实现 get_image 和 draw 接口。以检测任务中可视化 instances 和 sem_seg 为例，则本地端可视化核心代码如下：

```python
class DetLocalVisualizer(BaseVisualizer):

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

    @BaseVisualizer.register_task('instances')
    def draw_instance(self, instances, data_type):
        ...

    @BaseVisualizer.register_task('sem_seg')
    def draw_sem_seg(self, pixel_data, data_type):
        ...
```

如果想使用 DetLocalVisualizer，用户可以直接在 python 代码中实例化，代码如下

```python
det_local_visualizer=DetLocalVisualizer()
```

用户也可以使用注册器实例化，配置如下

```python
visualizer= dict(type='DetLocalVisualizer')
```

## 写端 Writer

Visualizer 只是实现了单张图片的可视化功能，但是在训练或者测试过程中，对一些关键指标或者模型训练超参的记录非常重要，此功能通过写端 Writer 实现。同时写端 Writer 也可以通过 `bind_visualizer` 方法绑定 Visualizer 对象，从而通过 Writer 实现写图片、写指标等功能。

### BaseWriter

BaseWriter 定义了对外调用的接口规范，主要接口如下：

- add_hyperparams  写超参
- add_image 写图片
- add_scalar 写标量
- add_graph 写模型图
- bind_visualizer 绑定可视化对象
- experiment 写后端对象，例如 Wandb 对象和 Tensorboard 对象

BaseWriter 定义了 4 种常见的写数据接口，考虑到某些写后端功能非常强大，例如 Wandb，其具备写表格，写视频等等功能，针对这类需求用户可以直接获取 experiment 对象，然后调用写后端对象本身的 API 即可。

由于 Visualizer 和 Writer 对象是解耦的，用户可以通过配置文件自由组合各种 Visualizer 和 Writer，例如 WandbWriter 绑定 LocalVisualizer，表示图片可视化功能由 LocalVisualizer 提供，但是最终图片是写到了 Wandb 端，也可以 LocalWriter 绑定 WandbVisualizer，表示图片可视化功能由 WandbVisualizer 提供，然后直接保存到本地端。一个简单的例子如下所示

```python
# 1 实例化
det_local_visualizer=DetLocalVisualizer()
wandb_writer=WandbWriter()
# 2 绑定
wandb_writer.bind_visualizer(det_local_visualizer)

# 3 在任意代码位置，使用方式 1 (推荐)
wandb_writer.visualizer.draw(data_sample，image)
wandb_writer.add_image('vis_image', wandb_writer.visualizer.get_image())

# 3 在任意代码位置，使用方式 2 (如果你可以直接获取到 visualizer 实例)
det_local_visualizer.draw(data_sample，image)
wandb_writer.add_image('vis_image', det_local_visualizer.get_image())
```

## RuntimeWriter

考虑到在训练或者测试过程中，可能需要同时调用多个 Writer，例如想同时写到本地和 Wandb 端，为此设计了对外的 RuntimeWriter 类，在训练或者测试过程中  RuntimeWriter 会依次调用各个 Writer，主要接口如下：

- add_hyperparams  写超参
- add_image 写图片
- add_scalar 写标量
- add_graph 写模型图
- get_writer 获取某个 writer
- `__enter__`
- `__exit__`

为了让用户可以在代码的任意位置进行数据可视化，RuntimeWriter 类实现 `__enter__` 和 ` __exit__`方法，并且在 Runner 中使上下文生效，从而在该上下文作用域内，用户可以通过 `get_writers` 工具函数获取 RuntimeWriter 类实例，从而调用该类的各种可视化和写方法。一个简单粗略的实现和用例如下

```python
# 假设 writer 只有一个
visualizer=build_visualizer(cfg.visualizer,VISUALIZERS)
writer =build_writer(cfg.writer,WRITERS)
writer.bind_visualizer(visualizer)

# 假设在 epoch 训练过程中
with RuntimeWriter(writer):
    while self.epoch < self._max_epochs:
         for i, flow in enumerate(workflow):
            ...
```

```python
# 配置文件写法
visualizer= dict(type='DetLocalVisualizer', show=False)
writer = dict(type='WandbWriter',init_kwargs=dict(project='demo'))

# 在上下文作用域生效的任意位置
runtime_writer=get_writers()
runtime_writer.visualizer.draw(data_sample，image)
runtime_writer.add_image('vis_image',runtime_writer.visualizer.get_image(),iter=iter)
runtime_writer.add_scalar('acc',val,iter=iter)
```
