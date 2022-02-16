# 可视化 (Visualizer)

可视化可以给深度学习的模型训练和测试过程提供直观解释。在 OpenMMLab 算法库中，一个合格的可视化功能设计，需要满足以下需求：

- 提供丰富的开箱即用可视化功能，能够满足大部分计算机视觉可视化任务
- 高扩展性，可视化功能通常多样化，应该能够通过简单扩展实现定制需求
- 能够在训练和测试流程的任意点位进行可视化
- OpenMMLab 各个算法库具有统一可视化接口，利于用户理解和维护

基于上述需求，OpenMMLab 2.0 引入了可视化对象 Visualizer 和写端对象 Writer 的概念

- **Visualizer 负责单张图片的各类绘制和可视化功能**

  MMEngine 提供了统一接口的抽象类 BaseVisualizer，同时考虑到本地端可视化需要更细粒度的接口调用功能，例如在当前绘制图片基础上追加一个文本说明功能，为此在 BaseVisualizer 基础上构建了 BaseLocalVisualizer，其提供了诸如 `draw_bbox` 和 `draw_text` 等链式调用接口，所有本地端可视化类都继承至 BaseLocalVisualizer。

  各个下游算法库可以继承 BaseVisualizer 或者 BaseLocalVisualizer 实现所需的可视化功能，例如 DetLocalVisualizer 继承至 BaseLocalVisualizer 实现检测相关的本地可视化功能，DetWandbVisualizer 继承至 BaseVisualizer 实现检测相关的远程 Wandb 可视化功能。Visualizer类的 UML 关系图如下
<div align="center">
  <img src="https://user-images.githubusercontent.com/17425982/154222901-8f7bea1b-5f8e-456c-8e0b-bd75c7cb602e.png" >
</div>
- **Writer 负责将各类数据写入到指定后端**

  写入的数据可以是图片，模型结构图，也可以是标量例如模型精度指标，**如果是写图片数据，则是先调用 Visualizer 接口进行绘制，再调用 Writer 实例的 add_image 接口接收绘制后的图片，从而写入，写其他数据则是调用 Writer 本身接口实现**。MMEngine 提供了统一接口的抽象类 BaseWriter，和一些常用的 Writer 如 LocalWriter 来支持将数据写入本地，TensorboardWriter 来支持将数据写入 Tensorboard，WandbWriter 来支持将数据写入 Wandb。用户也可以自定义 Writer 来将数据写入自定义后端。

  考虑到在训练或者测试过程中同时存在多个  Writer 对象，例如同时想想本地和远程端写数据，为此设计了 **ComposeWriter**  负责管理所有运行中实例化的 Writer 对象，其会自动管理所有 Writer 对象，并遍历调用所有 Writer 对象的方法。Writer类的 UML 关系图如下
<div align="center">
 <img src="https://user-images.githubusercontent.com/17425982/154225398-7e478f68-58ae-46fd-ae23-47ad37ffb176.png" >
</div>

在训练或者测试过程中，常用的可视化流程为：

- 利用 Visualizer 对象对当前图片进行绘制，例如绘制边界框 bbox 和 掩码 mask 等等
- 将 Visualizer 绘制后的图片传递给 Writer 对象，由 Writer 对象负责写入到指定后端
- 如果想可视化训练或者测试过程中的曲线，例如模型精度指标曲线，可以直接利用 Writer 对象接口实现，无需使用 Visualizer

在使用本地主机进行可视化的情况下，如用户单独写脚本可视化图片或者使用 jupyter notebook 时，用户只需要实例化 Visualizer 对象即可。用户可以使用 Visualizer 对当前图片进行绘制，然后直接调用 Visualizer 的 save/show 等接口对图片进行保存/显示。

## 可视化对象 Visualizer

可视化对象 Visualizer 负责单张图片的各类绘制和可视化功能。为了统一 OpenMMLab 各个算法库的可视化接口，MMEngine 定义了 Visualizer 的基类 BaseVisualizer 和更细粒度调用接口 BaseLocalVisualizer，下游库可以继承 BaseVisualizer 或者 BaseLocalVisualizer，实现自己的可视化需求，例如 MMDetection 的 DetLocalVisualizer、DetTensorboardVisualizer、DetWandbVisualizer，用于进行本地端可视化、Tensorboard 端可视化和 Wandb 端可视化等等。

### BaseVisualizer

BaseVisualizer 提供了基础而通用的可视化功能，主要接口如下：

**(1) 绘制无关的功能性接口**

- set_image  设置原始图片数据
- get_image 获取绘制后的 Numpy 格式图片数据
- save 保存图片
- show  可视化
- register_task  注册绘制函数(其作用在 *自定义 Visualizer* 小节描述)
- setup_env 设置 work_dir 等必备的环境变量

**(2) 绘制相关基础接口**

- draw  用户使用的抽象绘制接口
- draw_featmap  绘制特征图

特征图可视化是一个常见的功能，通过调用 `draw_featmap` 可以直接可视化特征图，目前该函数支持如下功能(在子类实现)：

- 输入 4 维 BCHW 格式的 tensor，通道 C 是 1 或者 3 时候，展开成一张图片显示
- 输入 4 维 BCHW 格式的 tensor，通道 C 大于 3 时候，则支持选择激活度最高通道，展开成一张图片显示
- 输入 3 维 CHW 格式的 tensor，则选择激活度最高的 topk，然后拼接成一张图显示

### BaseLocalVisualizer

本地端可视化相比远程端 Wandb 等可视化对象应该具备更丰富更细粒度的功能，最常见的需求是链式调用，例如用户可以先绘制边界框，在此基础上绘制文本，绘制线段，最后保存起来，则调用过程为：

```python
visualizer.set_image(image)
visualizer.draw_bbox(...).draw_text(...).draw_line(...)
visualizer.save()
```

通常 Wandb 和 Tensorboard 等远程可视化类不具备链式调用功能，为此额外设计了 BaseLocalVisualizer，其相比 BaseVisualizer 多了链式调用功能，用户可以链式的调用 BaseLocalVisualizer 中的 `draw_xxx` 接口绘制。相比 BaseLocalVisualizer 其新增的绘制接口如下：

- draw_bbox  绘制边界框
- draw_text  绘制文本框
- draw_line  绘制线段
- draw_circle  绘制圆
- draw_polygon  绘制多边形
- draw_binary_mask  绘制二值掩码

上述接口的返回值都是 BaseLocalVisualizer 对象本身。

### 自定义 Visualizer

自定义的 Visualizer 中大部分情况下只需要实现 get_image 和 draw 接口。如果是本地端 Visualizer 并想具备链式调用功能，则应该继承 BaseLocalVisualizer，如果不具备链式调用功能，则直接继承 BaseVisualizer 即可。

以目标检测可视化需求为例，可能需要同时可视化边界框 bbox、掩码 mask 和语义分割图 seg_map，如果如此多功能全部写到 draw 方法中会难以理解和维护。基于该问题，并基于 OpenMMLab 2.0 抽象数据接口规范，在 BaseVisualizer 新增了 register_task 函数。假设 MMDetection 中需要同时可视化预测结果中的 instances 和 sem_seg，可以在 MMDetection 的 DetLocalVisualizer 中实现 `draw_instances` 和 `draw_sem_seg` 两个方法，用于绘制预测实例和预测语义分割图， 我们希望只要输入数据中存在 instances 或 sem_seg 时候，对应的两个绘制函数  `draw_instances` 和 `draw_sem_seg` 能够自动被调用，而用户不需要手动调用。为了实现上述功能，可以通过在 `draw_instances` 和 `draw_sem_seg` 两个函数加上 `@BaseVisualizer.register_task` 装饰器。

```python
class DetLocalVisualizer(BaseLocalVisualizer):

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

注意：是否使用 register_task 装饰器函数不是必须的，如果用户自定义 Visualizer，并且 draw 实现非常简单，则无需考虑 register_task。



如果想使用 DetLocalVisualizer，用户可以直接在 python 代码中实例化，代码如下

```python
det_local_visualizer=DetLocalVisualizer()
```

用户也可以使用注册器实例化，配置如下

```python
visualizer= dict(type='DetLocalVisualizer')
det_local_visualizer=build_visualizer(visualizer)
```

## 写端 Writer

Visualizer 只是实现了单张图片的可视化功能，但是在训练或者测试过程中，对一些关键指标或者模型训练超参的记录非常重要，此功能通过写端 Writer 实现。同时写端 Writer 也可以通过 `bind_visualizer` 方法绑定 Visualizer 对象，从而通过 Writer 实现写图片、写指标等功能。

BaseWriter 定义了对外调用的接口规范，主要接口如下：

- add_hyperparams  写超参
- add_image 写 Numpy 格式的图片
- add_scalar 写标量
- add_graph 写模型图
- bind_visualizer 绑定可视化对象(可选)
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

需要强调 Writer 和 Visualizer 对象是完全解耦的，也就是说  `bind_visualizer` 不是必须要实现的接口，如果用户不需要通过 Writer 获取 Visualizer 实例，那么   `bind_visualizer` 可以忽略，提供该接口只是为了方便使用而已。

## 组合写端 ComposeWriter

考虑到在训练或者测试过程中，可能需要同时调用多个 Writer，例如想同时写到本地和 Wandb 端，为此设计了对外的 ComposeWriter 类，在训练或者测试过程中  ComposeWriter 会依次调用各个 Writer，主要接口如下：

- add_hyperparams  写超参
- add_image 写图片
- add_scalar 写标量
- add_graph 写模型图
- setup_env 设置 work_dir 等必备的环境变量
- get_writer 获取某个 writer
- `__enter__` 上下文进入函数
- `__exit__` 上下文推出函数

为了让用户可以在代码的任意位置进行数据可视化，ComposeWriter 类实现 `__enter__` 和 ` __exit__`方法，并且在 Runner 中使上下文生效，从而在该上下文作用域内，用户可以通过 `get_writers` 工具函数获取 ComposeWriter 类实例，从而调用该类的各种可视化和写方法。一个简单粗略的实现和用例如下

```python
# 假设 writer 只有一个
visualizer=build_visualizer(cfg.visualizer,VISUALIZERS)
writer =build_writer(cfg.writer,WRITERS)
writer.bind_visualizer(visualizer)

# 假设在 epoch 训练过程中
with ComposeWriter(writer):
    while self.epoch < self._max_epochs:
         for i, flow in enumerate(workflow):
            ...
```

```python
# 配置文件写法
visualizer= dict(type='DetLocalVisualizer', show=False)
writer = dict(type='WandbWriter',init_kwargs=dict(project='demo'))

# 在上下文作用域生效的任意位置
compose_writer=get_writers()
compose_writer.visualizer.draw(data_sample，image)
compose_writer.add_image('vis_image',compose_writer.visualizer.get_image(),iter=iter)
compose_writer.add_scalar('acc',val,iter=iter)
```
