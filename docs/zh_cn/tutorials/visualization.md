## 可视化

可视化可以给深度学习的模型训练和测试过程提供直观解释。在 OpenMMLab 算法库中提供了如下功能：

- 丰富的开箱即用可视化功能，能够满足大部分计算机视觉可视化任务
- 高扩展性，用户可以通过简单扩展实现定制需求
- 能够在训练和测试流程的任意点位进行可视化
- OpenMMLab 各个算法库具有统一可视化接口，利于用户理解和维护

下面具体说明

### 1 绘制并本地窗口显示

**(1) 绘制检测框、掩码和文本等**
可视化器 Visualizer 提供了需要常用对象的绘制接口，例如绘制**检测框、点、文本、线、圆、多边形和二值掩码**。常见用法如下：

```python
import torch
from mmengine.visualization import Visualizer

# image 为 rgb 格式数据
visualizer = Visualizer(image=image)
# 绘制单个检测框, xyxy 格式
visualizer.draw_bboxes(torch.tensor([10, 5, 20, 40]))
visualizer.show()

# 或者绘制多个检测框
visualizer.set_image(image)
visualizer.draw_bboxes(torch.tensor([[10, 5, 20, 40], [50, 20, 80, 40]]))
visualizer.show()
```

上述绘制接口中除了绘制文本接收字符串数据外都可以接受 tensor 或者 np.array 格式的数据。

```python
# image 为 rgb 格式数据
visualizer = Visualizer(image=image)
visualizer.draw_texts("hello world!", torch.tensor([50, 50]))
visualizer.show()
```

上述绘制接口可以多次调用，从而实现叠加显示需求

```python
# image 为 rgb 格式数据
visualizer = Visualizer(image=image)

visualizer.draw_bboxes(torch.tensor([[10, 5, 20, 40], [50, 20, 80, 40]]))
visualizer.draw_texts("hello world!", torch.tensor([50, 50]))
visualizer.draw_circles(torch.tensor([20, 50]), torch.tensor([5]))

visualizer.show()
```

用户可以通过各个绘制接口中提供的参数来定制绘制对象的颜色和宽度等等。

**(2) 绘制特征图**
特征图可视化功能较多，目前不支持 batch 输入，为了方便理解，将其对外接口梳理如下：

```python
@staticmethod
def draw_featmap(featmap: torch.Tensor, # 输入格式要求为 CHW
                 overlaid_image: Optional[np.ndarray] = None, # 如果同时输入了 image 数据，则特征图会叠加到 image 上绘制
                 channel_reduction: Optional[str] = 'squeeze_mean', # 多个通道压缩为单通道的策略
                 topk: int = 10, # 可选择激活度最高的 topk 个特征图显示
                 arrangement: Tuple[int, int] = (5, 2), # 多通道展开为多张图时候布局
                 resize_shape：Optional[tuple] = None, # 可以指定 resize_shape 参数来缩放特征图
                 alpha: float = 0.5) -> np.ndarray: # 图片和特征图绘制的叠加比例
```

其功能可以归纳如下

- 输入的 Tensor 一般是包括多个通道的，channel_reduction 参数可以将多个通道压缩为单通道，然后和图片进行叠加显示

  - `squeeze_mean` 将输入的 C 维度采用 mean 函数压缩为一个通道，输出维度变成 (1, H, W)
  - `select_max` 从输入的 C 维度中先在空间维度 sum，维度变成 (C, )，然后选择值最大的通道
  - `None` 表示不需要压缩，此时可以通过 topk 参数可选择激活度最高的 topk 个特征图显示

- 在 channel_reduction 参数为 None 的情况下，topk 参数生效，其会按照激活度排序选择 topk 个通道，然后和图片进行叠加显示，并且此时会通过 arrangement 参数指定显示的布局

  - 如果 topk 不是 -1，则会按照激活度排序选择 topk 个通道显示
  - 如果 topk = -1，此时通道 C 必须是 1 或者 3 表示输入数据是图片，否则报错提示用户应该设置 `channel_reduction`来压缩通道。

- 考虑到输入的特征图通常非常小，函数支持输入 `resize_shape` 参数，方便将特征图进行上采样后进行可视化。

**常见功能 1**：将多通道特征图采用 `select_max` 参数压缩为单通道并显示

```python
visualizer = Visualizer()
# feat 为 CHW 格式的 tensor
drawn_img = visualizer.draw_featmap(feat, channel_reduction='select_max')
visualizer.show(drawn_img)
```

**常见功能 2**： 将多通道特征图采用 `select_max` 参数压缩为单通道，将其指定尺寸输出。如果输入的特征图比较小，可以使用 `resize_shape` 对特征图上采样可视化

```python
visualizer = Visualizer()
# feat 为 CHW 格式的 tensor
drawn_img = visualizer.draw_featmap(feat, channel_reduction='select_max'，resize_shape=(200, 100))
visualizer.show(drawn_img)
```

**常见功能 3**： 将多通道特征图采用 `select_max` 参数压缩为单通道并叠加原图显示

```python
visualizer = Visualizer()
# feat 为 CHW 格式的 tensor
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction='select_max')
visualizer.show(drawn_img)
```

**常见功能 4**： 利用 `topk=5` 参数选择多通道特征图中激活度最高的 5 个通道并采用 2x3 布局显示

```python
visualizer = Visualizer()
# feat 为 CHW 格式且 shape 为 (10, 50, 50) tensor
drawn_img = visualizer.draw_featmap(feat, channel_reduction=None, topk=5, arrangement=(2, 3))
assert drawn_img.shape == (2 * 50, 3 * 50, 3)
visualizer.show(drawn_img)
```

用户可以通过 `arrangement` 参数选择自己想要的布局

```python
visualizer = Visualizer()
# feat 为 CHW 格式且 shape 为 (10, 50, 50) tensor
drawn_img = visualizer.draw_featmap(feat, channel_reduction=None, topk=5, arrangement=(4, 2))
assert drawn_img.shape == (4 * 50, 2 * 50, 3)
visualizer.show(drawn_img)
```

### 2 绘制并选择不同存储后端

在绘制完成后，可以选择本地窗口显示，也可以存储到不同后端中，目前 MMEngine 内置了本地存储、Tensorboard 存储和 WandB 存储 3 个后端。

**(1) 单存储后端**
如果选择本地存储后端，如下所示：

```python
# image 为 rgb 格式数据
visualizer = Visualizer(image=image, vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')

visualizer.draw_bboxes(torch.tensor([[10, 5, 20, 40], [50, 20, 80, 40]]))
visualizer.draw_texts("hello world!", torch.tensor([50, 50]))
visualizer.draw_circles(torch.tensor([20, 50]), torch.tensor([5]))

# 会生成 temp_dir/vis_data/vis_image/demo_0.png
visualizer.add_image('demo', visualizer.get_image())

# 保存特征图
drawn_img = visualizer.draw_featmap(feat, channel_reduction=None, topk=5, arrangement=(4, 2))
# 会生成 temp_dir/vis_data/vis_image/feat_0.png
visualizer.add_image('demo', drawn_img)
```

其中生成的后缀 0 是用来区分不同 step 场景

```python
# 会生成 temp_dir/vis_data/vis_image/demo_1.png
visualizer.add_image('demo', visualizer.get_image(), step=1)
# 会生成 temp_dir/vis_data/vis_image/demo_3.png
visualizer.add_image('demo', visualizer.get_image(), step=3)
```

选择其他后端也是完全相同的用法

```python
# image 为 rgb 格式数据
visualizer = Visualizer(image=image, vis_backends=[dict(type='TensorboardVisBackend')], save_dir='temp_dir')
# visualizer = Visualizer(image=image, vis_backends=[dict(type='WandbVisBackend')], save_dir='temp_dir')

visualizer.draw_bboxes(torch.tensor([[10, 5, 20, 40], [50, 20, 80, 40]]))
visualizer.draw_texts("hello world!", torch.tensor([50, 50]))
visualizer.draw_circles(torch.tensor([20, 50]), torch.tensor([5]))

# 会生成 temp_dir/vis_data/events.out.tfevents.xx 文件
visualizer.add_image('demo', visualizer.get_image())
```

**(2) 多存储后端**
用户也可以配置多个存储后端，将绘制后结果保存到不同的位置

```python
# image 为 rgb 格式数据
visualizer = Visualizer(image=image, vis_backends=[dict(type='TensorboardVisBackend'),
                                                   dict(type='LocalVisBackend')],
                        save_dir='temp_dir')

visualizer.draw_bboxes(torch.tensor([[10, 5, 20, 40], [50, 20, 80, 40]]))
visualizer.draw_texts("hello world!", torch.tensor([50, 50]))
visualizer.draw_circles(torch.tensor([20, 50]), torch.tensor([5]))

visualizer.add_image('demo', visualizer.get_image())
```

注意：如果多个存储后端中存在同一个类的多个后端，那么必须指定 name 字段，否则无法区分是哪个存储后端

```python
# image 为 rgb 格式数据
visualizer = Visualizer(image=image, vis_backends=[dict(type='TensorboardVisBackend', name='tb_1', save_dir='temp_dir_1'),
                                                   dict(type='TensorboardVisBackend', name='tb_2', save_dir='temp_dir_2'),
                                                   dict(type='LocalVisBackend', name='local')],
                        save_dir='temp_dir')
```

### 3 保存配置和标量到不同存储后端

存储后端除了可以保存图片相关信息，还可以保存 OpenMMLab 训练过程中的配置信息和标量数据。

```python
# 保存配置
visualizer = Visualizer(vis_backends=[dict(type='TensorboardVisBackend')], save_dir='temp_dir')
visualizer.add_config(config)
```

```python
# 保存标量，例如评估指标
visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')
visualizer.add_scalar('map', 0.7, step=0)
visualizer.add_scalar('map', 0.8, step=1)
```

也可以一次性保存多个标量数据

```python
visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')
visualizer.add_scalars({'loss': [1, 2, 3], 'acc': 0.8})
```

同样的，用户也可以配置多个存储后端。

### 4 任意点位进行可视化

在深度学习过程中，会存在在某些代码位置插入可视化函数，并将其保存到不同后端的需求，这类需求主要用于可视化分析和调试阶段。MMEngine 设计的可视化器支持在任意点位获取同一个可视化器然后进行可视化的功能。
用户只需要在初始化时候通过 `get_instance` 接口实例化可视化对象，此时该可视化对象即为全局可获取唯一对象，后续通过  `Visualizer.get_current_instance()` 即可在代码任意位置获取。

```python
# 在程序初始化时候调用
visualizer1 = Visualizer.get_instance(name='vis', vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')

# 在任何代码位置都可调用
visualizer2 = Visualizer.get_current_instance()
visualizer2.add_scalar('map', 0.7, step=0)

assert id(visualizer1) == id(visualizer2)
```

也可以通过字段配置方式全局初始化

```python
visualizer_cfg=dict(
                name='vis',
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir='temp_dir')
VISUALIZERS.build(visualizer_cfg)
```

### 5 扩展存储后端和可视化器

**(1) 调用特定存储后端功能**
目前存储后端仅仅提供了保存配置、保存标量等功能，但是由于 WandB 和 Tensorboard 这类存储后端功能非常强大，用户可能会希望利用到这类存储后端的扩展类功能。为此存储后端提供了  `experiment` 属性来获取后端对象，从而满足各类定制化功能。
例如用户想将自定义数据保存为表格显示，而 WandB 提供了该类 API 接口，此时用户可以通过 `experiment`属性获取 WandB 对象，然后调用特定的 API

```python
# 全局初始化
Visualizer.get_instance(name='vis', vis_backends=[dict(type='WandbVisBackend')], save_dir='temp_dir')

# 任意代码位置
visualizer = Visualizer.get_current_instance()
# 获取 wandb 对象
wandb = visualizer.get_backend('WandbVisBackend').experiment
# 追加表格数据
table = wandb.Table(columns=["step", "mAP"])
table.add_data(1, 0.2)
table.add_data(2, 0.5)
table.add_data(3, 0.9)
# 保存
wandb.log({"table": table})
```

**(2) 扩展存储后端**
用户可以方便快捷的扩展存储后端。只需要继承自 `BaseVisBackend` 并实现各类 `add_xx` 方法即可

```python
from mmengine.registry import VISBACKENDS
from mmengine.visualization import BaseVisBackend

@VISBACKENDS.register_module()
class DemoVisBackend(BaseVisBackend):
    def add_image(self, **kwargs):
        pass

visualizer = Visualizer(vis_backends=[dict(type='DemoVisBackend')], save_dir='temp_dir')
visualizer.add_image('demo',image)
```

**(3) 扩展可视化器**
同样的，用户可以方便快捷的扩展可视化器。只需要继承自 Visualizer 并实现想覆写的函数即可。大部分情况下用户扩展可视化器，只需要覆写  `add_datasample`即可，该接口为各个下游库绘制 datasample 数据的抽象接口，以 MMDetection 为例，datasample 数据中通常包括 标注 bbox、标注 mask 、预测 bbox 或者预测 mask 等数据，MMDetection 会继承 Visualizer 并实现 `add_datasample` 接口，在该接口内部会针对检测任务相关数据进行可视化绘制，从而简化检测任务可视化需求。

```python
from mmengine.registry import VISUALIZERS

@VISUALIZERS.register_module()
class DetLocalVisualizer(Visualizer):
    def add_datasample(self,
                       name,
                       image: np.ndarray,
                       gt_sample: Optional['BaseDataElement'] = None,
                       pred_sample: Optional['BaseDataElement'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        pass

visualizer_cfg = dict(
    type='DetLocalVisualizer', vis_backends=[dict(type='WandbVisBackend')], name='visualizer')

# 全局初始化
VISUALIZERS.build(visualizer_cfg)

# 任意代码位置
det_local_visualizer = Visualizer.get_current_instance()
det_local_visualizer.add_datasample('det', image, gt_sample, pred_sample)
```
