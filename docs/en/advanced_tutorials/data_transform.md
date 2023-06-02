# Data transform

In the OpenMMLab repositories, dataset construction and data preparation are decoupled from each other.
Usually, the dataset construction only parses the dataset and records the basic information of each sample,
while the data preparation is performed by a series of data transforms, such as data loading, preprocessing,
and formatting based on the basic information of the samples.

## To use Data Transforms

In MMEngine, we use various callable data transforms classes to perform data manipulation. These data
transformation classes can accept several configuration parameters for instantiation and then process the
input data dictionary by calling. Also, all data transforms accept a dictionary as input and output the
processed data as a dictionary. A simple example is as belows:

```{note}
In MMEngine, we don't have the implementations of data transforms. you can find the base data transform class
and many other data transforms in MMCV. So you need to install MMCV before learning this tutorial, see the
{external+mmcv:doc}`MMCV installation guide <get_started/installation>`.
```

```python
>>> import numpy as np
>>> from mmcv.transforms import Resize
>>>
>>> transform = Resize(scale=(224, 224))
>>> data_dict = {'img': np.random.rand(256, 256, 3)}
>>> data_dict = transform(data_dict)
>>> print(data_dict['img'].shape)
(224, 224, 3)
```

## To use in Config Files

In config files, we can compose multiple data transforms as a list, called a data pipeline. And the data
pipeline is an argument of the dataset.

Usually, a data pipeline consists of the following parts:

1. Data loading, use [`LoadImageFromFile`](mmcv.transforms.LoadImageFromFile) to load image files.
2. Label loading, use [`LoadAnnotations`](mmcv.transforms.LoadAnnotations) to load the bboxes, semantic segmentation and keypoint annotations.
3. Data processing and augmentation, like [`RandomResize`](mmcv.transforms.RandomResize).
4. Data formatting, we use different data transforms for different tasks. And the data transform for specified
   task is implemented in the corresponding repository. For example, the data formatting transform for image
   classification task is `PackClsInputs` and it's in MMPretrain.

Here, taking the classification task as an example, we show a typical data pipeline in the figure below. For
each sample, the basic information stored in the dataset is a dictionary as shown on the far left side of the
figure, after which, every blue block represents a data transform, and in every data transform, we add some new fields (marked in green) or update some existing fields (marked in orange) in the data dictionary.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/206081993-d5351151-466c-4b13-bf6d-9441c0c896c8.png" width="90%" style="background-color: white;padding: 10px;"/>
</div>

If want to use the above data pipeline in our config file, use the below settings:

```python
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=256, keep_ratio=True),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs'),
        ]
    )
)
```

## Common Data Transforms

According to the functionality, the data transform classes can be divided into data loading, data
pre-processing & augmentation and data formatting.

### Data Loading

To support loading large-scale dataset, usually we won't load all dense data during dataset construction, but
only load the file path of these data. Therefore, we need to load these data in the data pipeline.

|                     Data Transforms                      |                                     Functionality                                     |
| :------------------------------------------------------: | :-----------------------------------------------------------------------------------: |
| [`LoadImageFromFile`](mmcv.transforms.LoadImageFromFile) |                          Load images according to the path.                           |
|  [`LoadAnnotations`](mmcv.transforms.LoadImageFromFile)  | Load and format annotations information, including bbox, segmentation map and others. |

### Data Pre-processing & Augmentation

Data transforms for pre-processing and augmentation usually manipulate the image and annotation data, like
cropping, padding, resizing and others.

|                      Data Transforms                       |                         Functionality                          |
| :--------------------------------------------------------: | :------------------------------------------------------------: |
|                [`Pad`](mmcv.transforms.Pad)                |                   Pad the margin of images.                    |
|         [`CenterCrop`](mmcv.transforms.CenterCrop)         |            Crop the image and keep the center part.            |
|          [`Normalize`](mmcv.transforms.Normalize)          |                  Normalize the image pixels.                   |
|             [`Resize`](mmcv.transforms.Resize)             |         Resize images to the specified scale or ratio.         |
|       [`RandomResize`](mmcv.transforms.RandomResize)       |    Resize images to a random scale in the specified range.     |
| [`RandomChoiceResize`](mmcv.transforms.RandomChoiceResize) | Resize images to a random scale from several specified scales. |
|    [`RandomGrayscale`](mmcv.transforms.RandomGrayscale)    |                   Randomly grayscale images.                   |
|         [`RandomFlip`](mmcv.transforms.RandomFlip)         |                     Randomly flip images.                      |

### Data Formatting

Data formatting transforms will convert the data to some specified type.

|                 Data Transforms                  |                     Functionality                     |
| :----------------------------------------------: | :---------------------------------------------------: |
|      [`ToTensor`](mmcv.transforms.ToTensor)      | Convert the data of specified field to `torch.Tensor` |
| [`ImageToTensor`](mmcv.transforms.ImageToTensor) |  Convert images to `torch.Tensor` in PyTorch format.  |

## Custom Data Transform Classes

To implement a new data transform class, the class needs to inherit `BaseTransform` and implement `transform`
method. Here, we use a simple flip transforms (`MyFlip`) as example:

```python
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyFlip(BaseTransform):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
```

Then, we can instantiate a `MyFlip` object and use it to process our data dictionary.

```python
import numpy as np

transform = MyFlip(direction='horizontal')
data_dict = {'img': np.random.rand(224, 224, 3)}
data_dict = transform(data_dict)
processed_img = data_dict['img']
```

Or, use it in the data pipeline by modifying our config file:

```python
pipeline = [
    ...
    dict(type='MyFlip', direction='horizontal'),
    ...
]
```

Please note that to use the class in our config file, we need to confirm the `MyFlip` class will be imported
during running.
