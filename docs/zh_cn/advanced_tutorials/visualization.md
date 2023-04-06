# 可视化

可视化可以给深度学习的模型训练和测试过程提供直观解释。

MMEngine 提供了 `Visualizer` 可视化器用以可视化和存储模型训练和测试过程中的状态以及中间结果，具备如下功能：

- 支持基础绘图接口以及特征图可视化
- 支持本地、TensorBoard 以及 WandB 等多种后端，可以将训练状态例如 loss 、lr 或者性能评估指标以及可视化的结果写入指定的单一或多个后端
- 允许在代码库任意位置调用，对任意位置的特征、图像和状态等进行可视化和存储。

## 基础绘制接口

可视化器提供了常用对象的绘制接口，例如绘制**检测框、点、文本、线、圆、多边形和二值掩码**。这些基础 API 支持以下特性：

- 可以多次调用，实现叠加绘制需求
- 均支持多输入，除了要求文本输入的绘制接口外，其余接口同时支持 Tensor 以及 Numpy array 的输入

常见用法如下：

(1) 绘制检测框、掩码和文本等

```python
import torch
import mmcv
from mmengine.visualization import Visualizer

# https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/zh_cn/_static/image/cat_and_dog.png
image = mmcv.imread('docs/en/_static/image/cat_and_dog.png',
                    channel_order='rgb')
visualizer = Visualizer(image=image)
# 绘制单个检测框, xyxy 格式
visualizer.draw_bboxes(torch.tensor([72, 13, 179, 147]))
# 绘制多个检测框
visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220], [72, 13, 179, 147]]))
visualizer.show()
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186052649-8611ae43-1bb9-46e8-b6a1-dfc5063407c7.png" width="400"/>
</div>

```python
visualizer.set_image(image=image)
visualizer.draw_texts("cat and dog", torch.tensor([10, 20]))
visualizer.show()
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186052726-bd8f1571-b34b-471a-9876-9f0ae8c4e2be.png" width="400"/>
</div>

你也可以通过各个绘制接口中提供的参数来定制绘制对象的颜色和宽度等等

```python
visualizer.set_image(image=image)
visualizer.draw_bboxes(torch.tensor([72, 13, 179, 147]),
                       edge_colors='r',
                       line_widths=3)
visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220]]),line_styles='--')
visualizer.show()
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186053154-ddd9d6ec-56e0-45cc-86a8-b349812ba2e7.png" width="400"/>
</div>

(2) 叠加显示

上述绘制接口可以多次调用，从而实现叠加显示需求

```python
visualizer.set_image(image=image)
visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220], [72, 13, 179, 147]]))
visualizer.draw_texts("cat and dog",
                      torch.tensor([10, 20])).draw_circles(torch.tensor([40, 50]),
                      torch.tensor([20]))
visualizer.show()
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186053209-de5f57e5-4ccb-45af-9370-1788d257123b.png" width="400"/>
</div>

## 特征图绘制

特征图可视化功能较多，目前只支持单张特征图的可视化，为了方便理解，将其对外接口梳理如下：

```python
@staticmethod
def draw_featmap(
    # 输入格式要求为 CHW
    featmap: torch.Tensor,
    # 如果同时输入了 image 数据，则特征图会叠加到 image 上绘制
    overlaid_image: Optional[np.ndarray] = None,
    # 多个通道压缩为单通道的策略
    channel_reduction: Optional[str] = 'squeeze_mean',
    # 可选择激活度最高的 topk 个特征图显示
    topk: int = 10,
    # 多通道展开为多张图时候布局
    arrangement: Tuple[int, int] = (5, 2),
    # 可以指定 resize_shape 参数来缩放特征图
    resize_shape: Optional[tuple] = None,
    # 图片和特征图绘制的叠加比例
    alpha: float = 0.5,
) -> np.ndarray:
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

常见用法如下：以预训练好的 ResNet18 模型为例，通过提取 layer4 层输出进行特征图可视化

(1) 将多通道特征图采用 `select_max` 参数压缩为单通道并显示

```python
import numpy as np
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor

def preprocess_image(img, mean, std):
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

model = resnet18(pretrained=True)

def _forward(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x1 = model.layer1(x)
    x2 = model.layer2(x1)
    x3 = model.layer3(x2)
    x4 = model.layer4(x3)
    return x4

model.forward = _forward

image_norm = np.float32(image) / 255
input_tensor = preprocess_image(image_norm,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
feat = model(input_tensor)[0]

visualizer = Visualizer()
drawn_img = visualizer.draw_featmap(feat, channel_reduction='select_max')
visualizer.show(drawn_img)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186053367-61202d57-2402-4056-a651-1a9e0953b9b8.png" width="400"/>
</div>

由于输出的 feat 特征图尺寸为 7x7，直接可视化效果不佳，用户可以通过叠加输入图片或者 `resize_shape` 参数来缩放特征图。如果传入图片尺寸和特征图大小不一致，会强制将特征图采样到和输入图片相同空间尺寸

```python
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction='select_max')
visualizer.show(drawn_img)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186053413-4731e487-dd23-4ed4-b001-f958009280ad.png" width="400"/>
</div>

(2) 利用 `topk=5` 参数选择多通道特征图中激活度最高的 5 个通道并采用 2x3 布局显示

```python
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction=None,
                                    topk=5, arrangement=(2, 3))
visualizer.show(drawn_img)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186053463-6997f904-9d1e-4680-a2de-d4656e81e24d.png" width="400"/>
</div>

用户可以通过 `arrangement` 参数选择自己想要的布局

```python
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction=None,
                                    topk=5, arrangement=(4, 2))
visualizer.show(drawn_img)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186053653-d4d33e5d-f02e-4727-951c-880131d873dc.png" width="400"/>
</div>

## 基础存储接口

在绘制完成后，可以选择本地窗口显示，也可以存储到不同后端中，目前 MMEngine 内置了本地存储、Tensorboard 存储和 WandB 存储 3 个后端，且支持存储绘制后的图片、loss 等标量数据和配置文件。

**(1) 存储绘制后的图片**

假设存储后端为本地存储

```python
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='LocalVisBackend')],
                        save_dir='temp_dir')
visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220], [72, 13, 179, 147]]))
visualizer.draw_texts("cat and dog", torch.tensor([10, 20]))
visualizer.draw_circles(torch.tensor([40, 50]), torch.tensor([20]))

# 会生成 temp_dir/vis_data/vis_image/demo_0.png
visualizer.add_image('demo', visualizer.get_image())
```

其中生成的后缀 0 是用来区分不同 step 场景

```python
# 会生成 temp_dir/vis_data/vis_image/demo_1.png
visualizer.add_image('demo', visualizer.get_image(), step=1)
# 会生成 temp_dir/vis_data/vis_image/demo_3.png
visualizer.add_image('demo', visualizer.get_image(), step=3)
```

如果想使用其他后端，则只需要修改配置文件即可

```python
# TensorboardVisBackend
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='TensorboardVisBackend')],
                        save_dir='temp_dir')
# 或者 WandbVisBackend
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='WandbVisBackend')],
                        save_dir='temp_dir')
```

**(2) 存储特征图**

```python
visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
                        save_dir='temp_dir')
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction=None,
                                    topk=5, arrangement=(2, 3))
# 会生成 temp_dir/vis_data/vis_image/feat_0.png
visualizer.add_image('feat', drawn_img)
```

**(3) 存储 loss 等标量数据**

```python
# 会生成 temp_dir/vis_data/scalars.json
# 保存 loss
visualizer.add_scalar('loss', 0.2, step=0)
visualizer.add_scalar('loss', 0.1, step=1)
# 保存 acc
visualizer.add_scalar('acc', 0.7, step=0)
visualizer.add_scalar('acc', 0.8, step=1)
```

也可以一次性保存多个标量数据

```python
# 会将内容追加到 temp_dir/vis_data/scalars.json
visualizer.add_scalars({'loss': 0.3, 'acc': 0.8}, step=3)
```

**(4) 保存配置文件**

```python
from mmengine import Config
cfg=Config.fromfile('tests/data/config/py_config/config.py')
# 会生成 temp_dir/vis_data/config.py
visualizer.add_config(cfg)
```

## 多后端存储

实际上，任何一个可视化器都可以配置任意多个存储后端，可视化器会循环调用配置好的多个存储后端，从而将结果保存到多后端中。

```python
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='TensorboardVisBackend'),
                                      dict(type='LocalVisBackend')],
                        save_dir='temp_dir')
# 会生成 temp_dir/vis_data/events.out.tfevents.xxx 文件
visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220], [72, 13, 179, 147]]))
visualizer.draw_texts("cat and dog", torch.tensor([10, 20]))
visualizer.draw_circles(torch.tensor([40, 50]), torch.tensor([20]))

visualizer.add_image('demo', visualizer.get_image())
```

注意：如果多个存储后端中存在同一个类的多个后端，那么必须指定 name 字段，否则无法区分是哪个存储后端

```python
visualizer = Visualizer(
    image=image,
    vis_backends=[
        dict(type='TensorboardVisBackend', name='tb_1', save_dir='temp_dir_1'),
        dict(type='TensorboardVisBackend', name='tb_2', save_dir='temp_dir_2'),
        dict(type='LocalVisBackend', name='local')
    ],
    save_dir='temp_dir')
```

## 任意点位进行可视化

在深度学习过程中，会存在在某些代码位置插入可视化函数，并将其保存到不同后端的需求，这类需求主要用于可视化分析和调试阶段。MMEngine 设计的可视化器支持在任意点位获取同一个可视化器然后进行可视化的功能。
用户只需要在初始化时候通过 `get_instance` 接口实例化可视化对象，此时该可视化对象即为全局可获取唯一对象，后续通过  `Visualizer.get_current_instance()` 即可在代码任意位置获取。

```python
# 在程序初始化时候调用
visualizer1 = Visualizer.get_instance(
    name='vis',
    vis_backends=[dict(type='LocalVisBackend')]
)

# 在任何代码位置都可调用
visualizer2 = Visualizer.get_current_instance()
visualizer2.add_scalar('map', 0.7, step=0)

assert id(visualizer1) == id(visualizer2)
```

也可以通过字段配置方式全局初始化

```python
from mmengine.registry import VISUALIZERS

visualizer_cfg = dict(type='Visualizer',
                      name='vis_new',
                      vis_backends=[dict(type='LocalVisBackend')])
VISUALIZERS.build(visualizer_cfg)
```

## 扩展存储后端和可视化器

**(1) 调用特定存储后端**

目前存储后端仅仅提供了保存配置、保存标量等基本功能，但是由于 WandB 和 Tensorboard 这类存储后端功能非常强大， 用户可能会希望利用到这类存储后端的其他功能。因此，存储后端提供了 `experiment` 属性来方便用户获取后端对象，满足各类定制化功能。
例如 WandB 提供了表格显示的 API 接口，用户可以通过 `experiment`属性获取 WandB 对象，然后调用特定的 API 来将自定义数据保存为表格显示

```python
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='WandbVisBackend')],
                        save_dir='temp_dir')

# 获取 WandB 对象
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

visualizer = Visualizer(vis_backends=[dict(type='DemoVisBackend')],
                        save_dir='temp_dir')
visualizer.add_image('demo',image)
```

**(3) 扩展可视化器**

同样的，用户可以通过继承 Visualizer 并实现想覆写的函数来方便快捷的扩展可视化器。大部分情况下，用户需要覆写 `add_datasample`来进行拓展。数据中通常包括标注或模型预测的检测框和实例掩码，该接口为各个下游库绘制 datasample 数据的抽象接口。以 MMDetection 为例，datasample 数据中通常包括标注 bbox、标注 mask 、预测 bbox 或者预测 mask 等数据，MMDetection 会继承 Visualizer 并实现 `add_datasample` 接口，在该接口内部会针对检测任务相关数据进行可视化绘制，从而简化检测任务可视化需求。

```python
from mmengine.registry import VISUALIZERS

@VISUALIZERS.register_module()
class DetLocalVisualizer(Visualizer):
    def add_datasample(self,
                       name,
                       image: np.ndarray,
                       data_sample: Optional['BaseDataElement'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        pass

visualizer_cfg = dict(type='DetLocalVisualizer',
                      vis_backends=[dict(type='WandbVisBackend')],
                      name='visualizer')

# 全局初始化
VISUALIZERS.build(visualizer_cfg)

# 任意代码位置
det_local_visualizer = Visualizer.get_current_instance()
det_local_visualizer.add_datasample('det', image, data_sample)
```
