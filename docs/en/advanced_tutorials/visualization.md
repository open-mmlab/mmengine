# Visualization

Visualization provides an intuitive explanation of the training and testing process of the deep learning model.

MMEngine provides `Visualizer` to visualize and store the state and intermediate results of the model training and testing process, with the following features:

- It supports basic drawing interface and feature map visualization
- It enables recording training states (such as loss and lr), performance evaluation metrics, and visualization results to a specified or multiple backends, including local device, TensorBoard, and WandB.
- It can be used in any location in the code base.

## Basic Drawing APIs

`Visualizer` provides drawing APIs for common objects such as **detection bboxes, points, text, lines, circles, polygons, and binary masks**.

These APIs have the following features:

- Can be called multiple times to achieve overlay drawing requirements.
- All support multiple input types such as Tensor, Numpy array, etc.

Typical usages are as follows.

1. Draw detection bboxes, masks, text, etc.

```python
import torch
import mmcv
from mmengine.visualization import Visualizer

# https://raw.githubusercontent.com/open-mmlab/mmengine/main/docs/en/_static/image/cat_and_dog.png
image = mmcv.imread('docs/en/_static/image/cat_and_dog.png',
                    channel_order='rgb')
visualizer = Visualizer(image=image)
# single bbox formatted as [xyxy]
visualizer.draw_bboxes(torch.tensor([72, 13, 179, 147]))
# draw multiple bboxes
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

You can also customize things like color and width using the parameters in each API.

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

2. Overlay display

These APIs can be called multiple times to get an overlay result.

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

## Feature Map Visualization

Feature map visualization has many functions. Currently, we only support single feature map visualization.

```python
@staticmethod
def draw_featmap(
    # input format must be CHW
    featmap: torch.Tensor,
    # if image data is input at the same time,
    # the feature map will be overlaid on the image
    overlaid_image: Optional[np.ndarray] = None,
    # strategy to reduce multiple channels into a single channel
    channel_reduction: Optional[str] = 'squeeze_mean',
    # topk feature maps to show
    topk: int = 10,
    # the layout when multiple channels are expanded into multiple images
    arrangement: Tuple[int, int] = (5, 2),
    # scale the feature map
    resize_shape: Optional[tuple] = None,
    # overlay ratio between input image and generated feature map
    alpha: float = 0.5,
) -> np.ndarray:
```

The main features can be concluded as follows:

- As the input Tensor usually includes multiple channels, `channel_reduction` can reduce them into a single channel and overlay the result to the image.

  - `squeeze_mean` reduces the input channel C into a single channel using the mean function, so the output dimension becomes (1, H, W)
  - `select_max` select the channel with the maximum activation, where 'activation' refers to the sum across spatial dimensions of a channel.
  - `None` indicates that no reduction is needed, which allows the user to select the top k feature maps with the highest activation degree through the `topk` parameter.

- `topk` is only valid when the `channel_reduction` is `None`. It selects the top k channels according to the activation degree and then displays them overlaid with the image. The display layout can be specified using the `--arrangement` parameter.

  - If `topk` is not -1, `topk` channels with the largest activation will be selected for display.
  - If `topk` is -1, channel number C must be either 1 or 3 to indicate if the input is a picture. Otherwise, an error will be raised to prompt the user to reduce the channel with `channel_reduction`.

- Considering that the input feature map is usually very small, the function can upsample the feature map through `resize_shape` before the visualization.

For example, we would like to get the feature map from the layer4 output of a pre-trained ResNet18 model and visualize it.

1. Reduce the multi-channel feature map into a single channel using `select_max` and display it.

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

Since the output feat feature map size is 7x7, the visualization effect is not good if we directly work on it. Users can scale the feature map by overlaying the input image or the `resize_shape` parameter. If the size of the incoming image is not the same as the size of the feature map, the feature map will be forced to be resampled to the same spatial size as the input image.

```python
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction='select_max')
visualizer.show(drawn_img)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186053413-4731e487-dd23-4ed4-b001-f958009280ad.png" width="400"/>
</div>

2. Select the top five channels with the highest activation in the multi-channel feature map by setting `topk=5`, then format them into a 2x3 layout.

```python
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction=None,
                                    topk=5, arrangement=(2, 3))
visualizer.show(drawn_img)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186053463-6997f904-9d1e-4680-a2de-d4656e81e24d.png" width="400"/>
</div>

Users can set their own desired layout through `arrangement`.

```python
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction=None,
                                    topk=5, arrangement=(4, 2))
visualizer.show(drawn_img)
```

<div align=center>
<img src="https://user-images.githubusercontent.com/17425982/186053653-d4d33e5d-f02e-4727-951c-880131d873dc.png" width="400"/>
</div>

## Basic Storage APIs

Once the drawing is completed, users can choose to display the result directly or save it to different backends. The backends currently supported by MMEngine include local storage, `Tensorboard` and `WandB`. The data supported include drawn pictures, scalars, and configurations.

1. Save the result image

Suppose you want to save to your local device.

```python
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='LocalVisBackend')],
                        save_dir='temp_dir')
visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220], [72, 13, 179, 147]]))
visualizer.draw_texts("cat and dog", torch.tensor([10, 20]))
visualizer.draw_circles(torch.tensor([40, 50]), torch.tensor([20]))

# temp_dir/vis_data/vis_image/demo_0.png will be generated
visualizer.add_image('demo', visualizer.get_image())
```

The zero in the result file name is used to distinguish different steps.

```python
# temp_dir/vis_data/vis_image/demo_1.png will be generated
visualizer.add_image('demo', visualizer.get_image(), step=1)
# temp_dir/vis_data/vis_image/demo_3.png will be generated
visualizer.add_image('demo', visualizer.get_image(), step=3)
```

If you want to switch to other backends, you can change the configuration file like this:

```python
# TensorboardVisBackend
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='TensorboardVisBackend')],
                        save_dir='temp_dir')
# WandbVisBackend
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='WandbVisBackend')],
                        save_dir='temp_dir')
```

2. Store feature maps

```python
visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
                        save_dir='temp_dir')
drawn_img = visualizer.draw_featmap(feat, image, channel_reduction=None,
                                    topk=5, arrangement=(2, 3))
# temp_dir/vis_data/vis_image/feat_0.png will be generated
visualizer.add_image('feat', drawn_img)
```

3. Save scalar data such as loss

```python
# temp_dir/vis_data/scalars.json will be generated
# save loss
visualizer.add_scalar('loss', 0.2, step=0)
visualizer.add_scalar('loss', 0.1, step=1)
# save acc
visualizer.add_scalar('acc', 0.7, step=0)
visualizer.add_scalar('acc', 0.8, step=1)
```

Multiple scalar data can also be saved at once.

```python
# New contents will be added to the temp_dir/vis_data/scalars.json
visualizer.add_scalars({'loss': 0.3, 'acc': 0.8}, step=3)
```

4. Save configurations

```python
from mmengine import Config
cfg=Config.fromfile('tests/data/config/py_config/config.py')
# temp_dir/vis_data/config.py will be saved
visualizer.add_config(cfg)
```

## Various Storage Backends

Any `Visualizer` can be configured with any number of storage backends. `Visualizer` will loop through all the configured backends and save the results to each one.

```python
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='TensorboardVisBackend'),
                                      dict(type='LocalVisBackend')],
                        save_dir='temp_dir')
# temp_dir/vis_data/events.out.tfevents.xxx files will be generated
visualizer.draw_bboxes(torch.tensor([[33, 120, 209, 220], [72, 13, 179, 147]]))
visualizer.draw_texts("cat and dog", torch.tensor([10, 20]))
visualizer.draw_circles(torch.tensor([40, 50]), torch.tensor([20]))

visualizer.add_image('demo', visualizer.get_image())
```

Note: If there are multiple backends used at the same time, the `name` field must be specified. Otherwise, it is impossible to distinguish which backend it is.

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

## Visualize at Anywhere

During the development, users may need to add visualization functions somewhere in their codes and save the results to different backends, which is very common for analysis and debugging. `Visualizer` in MMEngine can obtain the data from the same visualizers and then visualize them.

Users only need to instantiate the visualizer through `get_instance` during initialization. The visualizer obtained this way is unique and globally accessible. Then it can be accessed anywhere in the code through `Visualizer.get_current_instance()`.

```python
# call during the initialization stage
visualizer1 = Visualizer.get_instance(
    name='vis',
    vis_backends=[dict(type='LocalVisBackend')]
)

# call anywhere
visualizer2 = Visualizer.get_current_instance()
visualizer2.add_scalar('map', 0.7, step=0)

assert id(visualizer1) == id(visualizer2)
```

It can also be initialized globally through the config field.

```python
from mmengine.registry import VISUALIZERS

visualizer_cfg = dict(type='Visualizer',
                      name='vis_new',
                      vis_backends=[dict(type='LocalVisBackend')])
VISUALIZERS.build(visualizer_cfg)
```

## Customize Storage Backends and Visualizers

1. Call a specific storage backend

The storage backend only provides basic functions such as saving configurations and scalars. However, users may want to utilize other powerful backend features like WandB and Tensorboard. Therefore, the storage backend provides the `experiment` attribute to facilitate users to obtain backend objects and meet various customized functions.

For example, WandB provides an API to display tables. Users can obtain the WandB objects through the `experiment` attribute and then call a specific API to save the data as a table to show.

```python
visualizer = Visualizer(image=image,
                        vis_backends=[dict(type='WandbVisBackend')],
                        save_dir='temp_dir')

# get WandB object
wandb = visualizer.get_backend('WandbVisBackend').experiment
# add data to the table
table = wandb.Table(columns=["step", "mAP"])
table.add_data(1, 0.2)
table.add_data(2, 0.5)
table.add_data(3, 0.9)
# save
wandb.log({"table": table})
```

2. Customize storage backends

Users only need to inherit `BaseVisBackend` and implement various `add_xx` methods to customize the storage backend easily.

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

3. Customize visualizers

Similarly, users can easily customize the visualizer by inheriting `Visualizer` and implementing the functions they want to override.

In most cases, users need to override `add_datasample`. The data usually includes detection bboxes and instance masks from annotations or model predictions. This interface is for drawing `datasample` data for various downstream libraries. Taking MMDetection as an example, the `datasample` data usually includes labeled bboxs, labeled masks, predicted bboxs, or predicted masks. MMDetection will inherit `Visualizer` and implement the `add_datasample` interface, drawing the data related to the detection task.

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

# global initialize
VISUALIZERS.build(visualizer_cfg)

# call anywhere in your code
det_local_visualizer = Visualizer.get_current_instance()
det_local_visualizer.add_datasample('det', image, data_sample)
```
