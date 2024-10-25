# Abstract Data Element

During the model training and testing, there will be a large amount of data passed through different components, and different algorithms usually have different kinds of data. For example, single-stage detectors may only need ground truth bounding boxes and ground truth box labels, whereas Mask R-CNN also requires the instance masks.

The training codes can be shown as:

```python
for img, img_metas, gt_bboxes, gt_labels in data_loader:
    loss = retinanet(img, img_metas, gt_bboxes, gt_labels)
```

```python
for img, img_metas, gt_bboxes, gt_masks, gt_labels in data_loader:
    loss = mask_rcnn(img, img_metas, gt_bboxes, gt_masks, gt_labels)
```

We can see that without encapsulation, the inconsistency of data required by different algorithms leads to the inconsistency of interfaces among different algorithm modules, which affects the extensibility of the whole algorithm library. Moreover, the modules within one algorithm library often need redundant interfaces in order to maintain compatibility.

These disadvantages are more obvious among different algorithm libraries, which makes it difficult to reuse modules and expand interfaces when implementing multi-task perception models (multiple tasks such as semantic segmentation, detection, key point detection, etc.).

To solve the above problems, MMEngine defines a set of abstract data interfaces to encapsulate various data during the implementation of the model. Suppose the above different data are encapsulated into `data_sample`, the training of different algorithms can be abstracted and unified into the following code:

```python
for img, data_sample in dataloader:
    loss = model(img, data_sample)
```

The abstracted interface unifies and simplifies the interface between modules in the algorithm library, and can be used to pass data between datasets, models, visualizers, evaluates, or even within different modules in one model.

Besides the basic add, delete, update, and query functions, this interface also supports transferring data between different devices and the operation of `dict` and `torch.Tensor`, which can fully satisfy the requirements of the algorithm library.

Those algorithm libraries based on MMEngine can inherit from this design and implement their own interfaces to meet the characteristics and custom needs of data in different algorithms, improving the expandability while maintaining a unified interface.

During the implementation, there are two types of data interfaces for the algorithm libraries:

- A collection of all annotation information and prediction information for a training or testing sample, such as the output of a dataset, the inputs of model and visualizer, typically constitutes all the information of an individual training or testing sample. MMEngine defines this as a `DataSample`.
- A single type of prediction or annotation, typically the output of a sub-module in an algorithm model, such as the output of the RPN in two-stage detection, the output of a semantic segmentation model, the output of a keypoint branch, or the output of the generator in GANs, is defined by MMEngine as a data element (`XXXData`).

The following section first introduces the base class [BaseDataElement](mmengine.structures.BaseDataElement) for `DataSample` and `XXXData`.

## BaseDataElement

There are two types of data in `BaseDataElement`. One is `data` such as the bounding box, label, and the instance mask, etc., the other is `metainfo` which contains the meta information of the data to ensure the integrity of the data, including `img_shape`, `img_id`, and some other basic information of the images. These information facilitate the recovery and the use of the data in visualization and other cases. Therefore, users need to explicitly distinguish and declare the data of these two types of attributes while creating the `BaseDataElement`.

To make it easier to use `BaseDataElement`, the data in both `data` and `metainfo` are attributes of `BaseDataElement`. We can directly access the data and metainfo by accessing the class attributes. In addition, `BaseDataElement` provides several methods for manipulating the data in `data`.

- Add, delete, update, and query data in different fields of `data`.
- Copy `data` to target devices.
- Support accessing data in the same way as a dictionary or a tensor to fully satisfy the algorithm's requirements.

### 1. Create BaseDataElement

The data parameter of `BaseDataElement` can be freely added by means of `key=value`. The fields of `metainfo`, however, need to be explicitly specified using the keyword `metainfo`.

```python
import torch
from mmengine.structures import BaseDataElement
# declare an empty object
data_element = BaseDataElement()

bboxes = torch.rand((5, 4))  # suppose bboxes is a tensor in the shape of Nx4. N represents the number of the boxes
scores = torch.rand((5,))  # suppose scores is a tensor with N dimensions. N represents the number of the noxes.
img_id = 0  # image ID
H = 800  # image height
W = 1333  # image width

# Set the data parameter directly in BaseDataElement
data_element = BaseDataElement(bboxes=bboxes, scores=scores)

# Explicitly declare the metainfo in BaseDataElement
data_element = BaseDataElement(
    bboxes=bboxes,
    scores=scores,
    metainfo=dict(img_id=img_id, img_shape=(H, W)))
```

### 2. `new` and `clone`

Users can use the `new()` method to create an abstract data interface with the same state and data from an existing data interface. You can set `metainfo` and `data` while creating a new `BaseDataElement` to create an abstract interface with the same state and data as `data` or `metainfo`. For example, `new(metainfo=xx)` makes the new `BaseDataElement` has the same content as the cloned `BaseDataElement`, but `metainfo` is set to the newly specified content. You can also use `clone()` directly to get a deep copy. The behavior of the `clone()` is the same as the `clone()` in PyTorch Tensor operation.

```python
data_element = BaseDataElement(
    bboxes=torch.rand((5, 4)),
    scores=torch.rand((5,)),
    metainfo=dict(img_id=1, img_shape=(640, 640)))

# set metainfo and data while creating BaseDataElement
data_element1 = data_element.new(metainfo=dict(img_id=2, img_shape=(320, 320)))
print('bboxes is in data_element1:', 'bboxes' in data_element1) # True
print('bboxes in data_element1 is same as bbox in data_element', (data_element1.bboxes == data_element.bboxes).all())
print('img_id in data_element1 is', data_element1.img_id == 2) # True

data_element2 = data_element.new(label=torch.rand(5,))
print('bboxes is not in data_element2', 'bboxes' not in data_element2) # True
print('img_id in data_element2 is same as img_id in data_element', data_element2.img_id == data_element.img_id)
print('label in data_element2 is', 'label' in data_element2)

# create a new object using `clone`, which makes the new object has the same data, same metainfo, and the same status as the data_element
data_element2 = data_element1.clone()
```

```
bboxes is in data_element1: True
bboxes in data_element1 is same as bbox in data_element tensor(True)
img_id in data_element1 is True
bboxes is not in data_element2 True
img_id in data_element2 is same as img_id in data_element True
label in data_element2 is True
```

### 3. Add and query attributes

When it comes to adding attributes, users can add attributes to the `data` in the same way they add class attributes. For `metainfo`, it generally stores metadata about images and is not usually modified. If there is a need to add attributes to `metainfo`, users should use the `set_metainfo` interface to explicitly modify it.

For querying, users can access the key-value pairs that exist only in `data` using `keys`, `values`, and `items`. Similarly, they can access the key-value pairs that exist only in `metainfo` using `metainfo_keys`, `metainfo_values`, and `metainfo_items`. Users can also access all attributes of the BaseDataElement, regardless of their type, using `all_keys`, `all_values`, and `all_items`.

To facilitate usage, users can access the data within `data` and `metainfo` in the same way they access class attributes. Alternatively, they can use the `get()` interface in a dictionary-like manner to access the data.

**Note:**

1. `BaseDataElement` does not support having the same field names in both `metainfo` and `data` attributes. Therefore, users should avoid setting the same field names in them, as it would result in an error in `BaseDataElement`.

2. Considering that `InstanceData` and `PixelData` support slicing operations on the data, in order to maintain consistency with the use of `[]` and reduce the number of different methods for the same need, BaseDataElement does not support accessing and setting its attributes like a dictionary. Therefore, operations like `BaseDataElement[name]` for value assignment and retrieval are not supported.

```python
data_element = BaseDataElement()
# Set the `metainfo` field of the data_element using `set_metainfo`,
# with img_id and img_shape becoming attributes of the data_element.
data_element.set_metainfo(dict(img_id=9, img_shape=(100, 100)))
# check metainfo key, value, and item
print("metainfo'keys are ", data_element.metainfo_keys())
print("metainfo'values are ", data_element.metainfo_values())
for k, v in data_element.metainfo_items():
    print(f'{k}: {v}')

print("Check img_id and img_shape from class parameters")
print('img_id: ', data_element.img_id)
print('img_shape: ', data_element.img_shape)
```

```
metainfo'keys are ['img_id', 'img_shape']
metainfo'values are [9, (100, 100)]
img_id: 9
img_shape: (100, 100)
Check img_id and img_shape from class parameters
img_id: 9
img_shape: (100, 100)
```

```python

# directly set data field via class attributes in BaseDataElement
data_element.scores = torch.rand((5,))
data_element.bboxes = torch.rand((5, 4))

print("data's key is: ", data_element.keys())
print("data's value is: ", data_element.values())
for k, v in data_element.items():
    print(f'{k}: {v}')

print("Check scores and bboxes via class attributes")
print('scores: ', data_element.scores)
print('bboxes: ', data_element.bboxes)

print("Check scores and bboxes via get()")
print('scores: ', data_element.get('scores', None))
print('bboxes: ', data_element.get('bboxes', None))
print('fake: ', data_element.get('fake', 'not exist'))
```

```
data's key is: ['scores', 'bboxes']
data's value is: [tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515]), tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])]
scores: tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515])
bboxes: tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])
Check scores and bboxes via class attributes
scores: tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515])
bboxes: tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])
Check scores and bboxes via get()
scores: tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515])
bboxes: tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])
fake: not exist
```

```python

print("All keys in data_element is: ", data_element.all_keys())
print("The length of values in data_element is: ", len(data_element.all_values()))
for k, v in data_element.all_items():
    print(f'{k}: {v}')
```

```
All key in data_element is: ['img_id', 'img_shape', 'scores', 'bboxes']
The length of values in data_element is 4
img_id: 9
img_shape: (100, 100)
scores: tensor([0.7937, 0.6307, 0.3682, 0.4425, 0.8515])
bboxes: tensor([[0.9204, 0.2110, 0.2886, 0.7925],
        [0.7993, 0.8982, 0.5698, 0.4120],
        [0.7085, 0.7016, 0.3069, 0.3216],
        [0.0206, 0.5253, 0.1376, 0.9322],
        [0.2512, 0.7683, 0.3010, 0.2672]])
```

### 4. Delete and modify attributes

Users can modify the `data` attribute of `BaseDataElement` in the same way they modify instance attributes. As for `metainfo`, it generally stores metadata about images and is not usually modified. If there is a need to modify `metainfo`, users should use the `set_metainfo` interface to make explicit modifications.

For convenience in operations, `data` and `metainfo` can be directly deleted using del. Additionally, the pop method is supported to delete attributes after accessing them.

```python
data_element = BaseDataElement(
    bboxes=torch.rand((6, 4)), scores=torch.rand((6,)),
    metainfo=dict(img_id=0, img_shape=(640, 640))
)
for k, v in data_element.all_items():
    print(f'{k}: {v}')
```

```
img_id: 0
img_shape: (640, 640)
scores: tensor([0.8445, 0.6678, 0.8172, 0.9125, 0.7186, 0.5462])
bboxes: tensor([[0.5773, 0.0289, 0.4793, 0.7573],
        [0.8187, 0.8176, 0.3455, 0.3368],
        [0.6947, 0.5592, 0.7285, 0.0281],
        [0.7710, 0.9867, 0.7172, 0.5815],
        [0.3999, 0.9192, 0.7817, 0.2535],
        [0.2433, 0.0132, 0.1757, 0.6196]])
```

```python
# modify data attributes
data_element.bboxes = data_element.bboxes * 2
data_element.scores = data_element.scores * -1
for k, v in data_element.items():
    print(f'{k}: {v}')

# delete data attributes
del data_element.bboxes
for k, v in data_element.items():
    print(f'{k}: {v}')

data_element.pop('scores', None)
print('The keys in data is: ', data_element.keys())
```

```
scores: tensor([-0.8445, -0.6678, -0.8172, -0.9125, -0.7186, -0.5462])
bboxes: tensor([[1.1546, 0.0578, 0.9586, 1.5146],
        [1.6374, 1.6352, 0.6911, 0.6735],
        [1.3893, 1.1185, 1.4569, 0.0562],
        [1.5420, 1.9734, 1.4344, 1.1630],
        [0.7999, 1.8384, 1.5635, 0.5070],
        [0.4867, 0.0264, 0.3514, 1.2392]])
scores: tensor([-0.8445, -0.6678, -0.8172, -0.9125, -0.7186, -0.5462])
The keys in data is []
```

```python
# modify metainfo
data_element.set_metainfo(dict(img_shape = (1280, 1280), img_id=10))
print(data_element.img_shape)  # (1280, 1280)
for k, v in data_element.metainfo_items():
    print(f'{k}: {v}')

# use pop access and delete
del data_element.img_shape
for k, v in data_element.metainfo_items():
    print(f'{k}: {v}')

data_element.pop('img_id')
print('The keys in metainfo is ', data_element.metainfo_keys())
```

```
(1280, 1280)
img_id: 10
img_shape: (1280, 1280)
img_id: 10
The keys in metainfo is []
```

### 5. Tensor-like operations

Users can transform the data status in `BaseDataElement` like the operations in tensor.Tensor. Currently, we support `cuda`, `cpu`, `to`, and `numpy`, etc. `to` has the same interface as `torch.Tensor.to()`, which allows users to change the status of the encapsulted tensor freely.

**Note:** These interfaces only handle sequences types in `np.array`, `torch.Tensor`, and numbers. Data in other types will be skipped, such as strings.

```python
data_element = BaseDataElement(
    bboxes=torch.rand((6, 4)), scores=torch.rand((6,)),
    metainfo=dict(img_id=0, img_shape=(640, 640))
)
# copy data to GPU
cuda_element_1 = data_element.cuda()
print('cuda_element_1 is on the device of', cuda_element_1.bboxes.device)  # cuda:0
cuda_element_2 = data_element.to('cuda:0')
print('cuda_element_1 is on the device of', cuda_element_2.bboxes.device)  # cuda:0

# copy data to cpu
cpu_element_1 = cuda_element_1.cpu()
print('cpu_element_1 is on the device of', cpu_element_1.bboxes.device)  # cpu
cpu_element_2 = cuda_element_2.to('cpu')
print('cpu_element_2 is on the device of', cpu_element_2.bboxes.device)  # cpu

# convert data to FP16
fp16_instances = cuda_element_1.to(
    device=None, dtype=torch.float16, non_blocking=False, copy=False,
    memory_format=torch.preserve_format)
print('The type of bboxes in fp16_instances is', fp16_instances.bboxes.dtype)  # torch.float16

# detach all data gradients
cuda_element_3 = cuda_element_2.detach()
print('The data in cuda_element_3 requires grad: ', cuda_element_3.bboxes.requires_grad)
# transform data to numpy array
np_instances = cpu_element_1.numpy()
print('The type of cpu_element_1 is convert to', type(np_instances.bboxes))
```

```
cuda_element_1 is on the device of cuda:0
cuda_element_1 is on the device of cuda:0
cpu_element_1 is on the device of cpu
cpu_element_2 is on the device of cpu
The type of bboxes in fp16_instances is torch.float16
The data in cuda_element_3 requires grad:  False
The type of cpu_element_1 is convert to <class 'numpy.ndarray'>
```

### 6. Show properties

`BaseDataElement` also implements `__repr__` which allows users to get all the data information through `print`. Meanwhile, to facilitate debugging, all attributes in `BaseDataElement` are added to `__dict__`. Users can visualize the contents directly in their IDEs. A complete property display is as follows:

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = BaseDataElement(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([0, 1, 2, 3])
instance_data.det_scores = torch.Tensor([0.01, 0.1, 0.2, 0.3])
print(instance_data)
```

```
<BaseDataElement(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([0, 1, 2, 3])
    det_scores: tensor([0.0100, 0.1000, 0.2000, 0.3000])
) at 0x7f9f339f85b0>
```

## xxxData

MMEngine categorizes the data elements into three categories:

- InstanceData: mainly for high-level tasks that encapsulated all instance-related data in the image, such as bounding boxes, labels, instance masks, key points, polygons, tracking ids, etc. All instance-related data has the same **length**, which is the number of instances in the image.
- PixelData: mainly for low-level tasks and some high-level tasks that require pixel-level labels. It encapsulates pixel-level data such as segmentation map for semantic segmentations, flow map for optical flow tasks, panoptic segmentation map for panoramic segmentations, and various images generated by bottom-level tasks like super-resolution maps, denoising maps, and other various style maps generated. These data typically have three or four dimensions, with the last two dimensions representing the height and width of the data, which are consistent across the dataset.
- LabelData: mainly for encapsulating label-level data, such as class labels in image classification or multi-class classification, content categories for generated images in image generation, text in text recognition tasks, and more.

### InstanceData

[`InstanceData`](mmengine.structures.InstanceData) builds upon `BaseDataElement` and introduces restrictions on the data stored in `data`, requiring that the length of the data is consistent. For example, in object detection, assuming an image has N objects (instances), you can store all the bounding boxes and labels in InstanceData, where the lengths of bounding boxes and label in InstanceData are the same. Based on this assumption, InstanceData is extended to include the following features:

- length validation of the data stored in InstanceData's data.
- support for dictionary-like access and assignment of attributes in the `data`.
- support for basic indexing, slicing, and advanced indexing capabilities.
- support for concatenation of InstanceData with the same keys but different instances.

These extended features support basic data structures such as `torch.tensor`, `numpy.ndarray`, list, str, and tuple, as well as custom data structures, as long as the custom data structure implements `__len__`, `__getitem__`, and `cat` methods.

#### Data verification

All data stored in `InstanceData` must have the same length.

```python
from mmengine.structures import InstanceData
import torch
import numpy as np

img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data.det_scores = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print('The length of instance_data is', len(instance_data))  # 2

instance_data.bboxes = torch.rand((3, 4))
```

```
The length of instance_data is 2
AssertionError: the length of values 3 is not consistent with the length of this :obj:`InstanceData` 2
```

#### Dictionary-like operations for accessing and setting attributes

`InstanceData` supports dictionary-like operations on **data** attributes.

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data["det_labels"] = torch.LongTensor([2, 3])
instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print(instance_data)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2, 3])
    det_scores: tensor([0.8000, 0.7000])
    bboxes: tensor([[0.6576, 0.5435, 0.5253, 0.8273],
                [0.4533, 0.6848, 0.7230, 0.9279]])
) at 0x7f9f339f8ca0>
```

#### Indexing and slicing

`InstanceData` supports the list indexing and slicing operations similar to Python, meanwhile, it also supports advanced indexing operations like numpy.

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data.det_scores = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print(instance_data)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2, 3])
    det_scores: tensor([0.8000, 0.7000])
    bboxes: tensor([[0.1872, 0.1669, 0.7563, 0.8777],
                [0.3421, 0.7104, 0.6000, 0.1518]])
) at 0x7f9f312b4dc0>
```

1. Indexing

```python
print(instance_data[1])
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([3])
    det_scores: tensor([0.7000])
    bboxes: tensor([[0.3421, 0.7104, 0.6000, 0.1518]])
) at 0x7f9f312b4610>
```

2. Slicing

```python
print(instance_data[0:1])
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2])
    det_scores: tensor([0.8000])
    bboxes: tensor([[0.1872, 0.1669, 0.7563, 0.8777]])
) at 0x7f9f312b4e20>
```

3. Advanced indexing

- list indexing

```python
sorted_results = instance_data[instance_data.det_scores.sort().indices]
print(sorted_results)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([3, 2])
    det_scores: tensor([0.7000, 0.8000])
    bboxes: tensor([[0.3421, 0.7104, 0.6000, 0.1518],
                [0.1872, 0.1669, 0.7563, 0.8777]])
) at 0x7f9f312b4a90>
```

- bool indexing

```python
filter_results = instance_data[instance_data.det_scores > 0.75]
print(filter_results)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2])
    det_scores: tensor([0.8000])
    bboxes: tensor([[0.1872, 0.1669, 0.7563, 0.8777]])
) at 0x7fa061299dc0>
```

4. result is empty

```python
empty_results = instance_data[instance_data.det_scores > 1]
print(empty_results)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([], dtype=torch.int64)
    det_scores: tensor([])
    bboxes: tensor([], size=(0, 4))
) at 0x7f9f439cccd0>
```

#### Concatenate data

Users can concatenate two `InstanceData` with the same key into one new `InstanceData`. For two different `InstanceData` with different length as N and M, the length of the output `InstanceData` is N + M.

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data.det_scores = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
print('The length of instance_data is', len(instance_data))
cat_results = InstanceData.cat([instance_data, instance_data])
print('The length of instance_data is', len(cat_results))
print(cat_results)
```

```
The length of instance_data is 2
The length of instance_data is 4
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2, 3, 2, 3])
    det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
    bboxes: tensor([[0.5341, 0.8962, 0.9043, 0.2824],
                [0.3864, 0.2215, 0.7610, 0.7060],
                [0.5341, 0.8962, 0.9043, 0.2824],
                [0.3864, 0.2215, 0.7610, 0.7060]])
) at 0x7fa061d4a9d0>
```

#### Customize data structures

Users need to implement `__len__`, `__getitem__`, and `cat` in their customized data structures to achieve the above functions.

```python
import itertools

class TmpObject:
    def __init__(self, tmp) -> None:
        assert isinstance(tmp, list)
        self.tmp = tmp

    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, item):
        if type(item) == int:
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))
        return TmpObject(self.tmp[item])

    @staticmethod
    def cat(tmp_objs):
        assert all(isinstance(results, TmpObject) for results in tmp_objs)
        if len(tmp_objs) == 1:
            return tmp_objs[0]
        tmp_list = [tmp_obj.tmp for tmp_obj in tmp_objs]
        tmp_list = list(itertools.chain(*tmp_list))
        new_data = TmpObject(tmp_list)
        return new_data

    def __repr__(self):
        return str(self.tmp)
```

```python
img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
instance_data = InstanceData(metainfo=img_meta)
instance_data.det_labels = torch.LongTensor([2, 3])
instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
instance_data.bboxes = torch.rand((2, 4))
instance_data.polygons = TmpObject([[1, 2, 3, 4], [5, 6, 7, 8]])
print(instance_data)
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    det_labels: tensor([2, 3])
    polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
    det_scores: tensor([0.8000, 0.7000])
    bboxes: tensor([[0.4207, 0.0778, 0.9959, 0.1967],
                [0.4679, 0.7934, 0.5372, 0.4655]])
) at 0x7fa061b5d2b0>
```

```python
# advanced indexing
print(instance_data[instance_data.det_scores > 0.75])
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    bboxes: tensor([[0.4207, 0.0778, 0.9959, 0.1967]])
    det_labels: tensor([2])
    det_scores: tensor([0.8000])
    polygons: [[1, 2, 3, 4]]
) at 0x7f9f312716d0>
```

```python
# cat
print(InstanceData.cat([instance_data, instance_data]))
```

```
<InstanceData(

    META INFORMATION
    pad_shape: (800, 1216, 3)
    img_shape: (800, 1196, 3)

    DATA FIELDS
    bboxes: tensor([[0.4207, 0.0778, 0.9959, 0.1967],
                [0.4679, 0.7934, 0.5372, 0.4655],
                [0.4207, 0.0778, 0.9959, 0.1967],
                [0.4679, 0.7934, 0.5372, 0.4655]])
    det_labels: tensor([2, 3, 2, 3])
    det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
    polygons: [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]
) at 0x7f9f31271490>
```

### PixelData

[`PixelData`](mmengine.structures.PixelData) upon `BaseDataElement` and imposes restrictions on the stored `data`:

- All data must be three-dimension in the order of (Channel, Height, and Width).
- All data must have the same length and width.

MMEngine extends the `PixelData` according to these assumptions, including:

- Dimension validation on data stored
- Support indexing and slicing the data in spatial dimension

#### Data verification

`PixelData` checks the length and dimensions of all the data passed to it.

```python
from mmengine.structures import PixelData
import random
import torch
import numpy as np
metainfo = dict(
    img_id=random.randint(0, 100),
    img_shape=(random.randint(400, 600), random.randint(400, 600)))
image = np.random.randint(0, 255, (4, 20, 40))
featmap = torch.randint(0, 255, (10, 20, 40))
pixel_data = PixelData(metainfo=metainfo,
                       image=image,
                       featmap=featmap)
print('The shape of pixel_data is', pixel_data.shape)
# set
pixel_data.map3 = torch.randint(0, 255, (20, 40))
print('The shape of pixel_data is', pixel_data.map3.shape)
```

```
The shape of pixel_data is (20, 40)
The shape of pixel_data is torch.Size([1, 20, 40])
```

```python
pixel_data.map2 = torch.randint(0, 255, (3, 20, 30))
# AssertionError: the height and width of values (20, 30) is not consistent with the length of this :obj:`PixelData` (20, 40)
```

```
AssertionError: the height and width of values (20, 30) is not consistent with the length of this :obj:`PixelData` (20, 40)
```

```python
pixel_data.map2 = torch.randint(0, 255, (1, 3, 20, 40))
# AssertionError: The dim of value must be 2 or 3, but got 4
```

```
AssertionError: The dim of value must be 2 or 3, but got 4
```

#### Querying in spatial dimension

`PixelData` supports indexing and slicing in spatial dimension on part of the data instances. Users only need to pass in the index of the length and width.

```python
metainfo = dict(
    img_id=random.randint(0, 100),
    img_shape=(random.randint(400, 600), random.randint(400, 600)))
image = np.random.randint(0, 255, (4, 20, 40))
featmap = torch.randint(0, 255, (10, 20, 40))
pixel_data = PixelData(metainfo=metainfo,
                       image=image,
                       featmap=featmap)
print('The shape of pixel_data is: ', pixel_data.shape)
```

```
The shape of pixel_data is (20, 40)
```

- Indexing

```python
index_data = pixel_data[10, 20]
print('The shape of index_data is: ', index_data.shape)
```

```
The shape of index_data is (1, 1)
```

- Slicing

```python
slice_data = pixel_data[10:20, 20:40]
print('The shape of slice_data is: ', slice_data.shape)
```

```
The shape of slice_data is (10, 20)
```

### LabelData

[`LabelData`](mmengine.structures.LabelData) is mainly used to encapsulate label data such as classiciation labels, predicted text labels, etc. `LabelData` has no limitations to `data`, and it provides two extra features: `onehot` transformation and `index` transformation.

```python
from mmengine.structures import LabelData
import torch

item = torch.tensor([1], dtype=torch.int64)
num_classes = 10

```

```python
onehot = LabelData.label_to_onehot(label=item, num_classes=num_classes)
print(f'{num_classes} is convert to ', onehot)

index = LabelData.onehot_to_label(onehot=onehot)
print(f'{onehot} is convert to ', index)
```

```
10 is convert to  tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) is convert to tensor([1])
```

## xxxDataSample

There may be different types of labels in one sample, for example, there may be both instance-level labels (Box) and pixel-level labels (SegMap) in one image. Therefore, we need to have a higher-level encapsulation on top of PixelData, InstanceData, and PixelData to represent the image-level labels. This layer is named `XXXDataSample` across the OpenMMLab series algorithms. In MMDet we have `DetDataSample`. All the labels are encapsulated in `XXXDataSample` during the training process, so different deep learning tasks can maintain a uniform data flow and data processing method.

### Downstream library usage

We take MMDet as an example to illustrate the use of the `DataSample` in downstream libraries and its constraints and naming styles. MMDet defined `DetDataSample` and seven fields, which are:

- Annotation Information
  - gt_instance (InstanceData): Instance annotation information includes the instance class, bounding box, etc. The type constraint is `InstanceData`.
  - gt_panoptic_seg (PixelData): For panoptic segmentation annotation information, the required type is PixelData.
  - gt_semantic_seg (PixelData): Semantic segmentation annotation information. The type constraint is `PixelData`.
- Prediction Results
  - pred_instance (InstanceData): Instance prediction results include the instance class, bounding boxes, etc. The type constraint is `InstanceData`.
  - pred_panoptic_seg (PixelData): Panoptic segmentation prediction results. The type constraint is `PixelData`.
  - pred_semantic_seg (PixelData): Semantic segmentation prediction results. The type constraint is `PixelData`.
- Intermediate Results
  - proposal (InstanceData): Mostly used for the RPN results in the two-stage algorithms. The type constraint is `InstanceData`.

```python
from mmengine.structures import BaseDataElement
import torch

class DetDataSample(BaseDataElement):

    # annotation
    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_gt_panoptic_seg', dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    # prediction
    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_pred_panoptic_seg', dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    # intermediate result
    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

```

### Type constraint

`DetDataSample` is used in the following way. It will throw an error when the data type is invalid, for example, using `torch.Tensor` to define `proposals` instead of `InstanceData`.

```python
data_sample = DetDataSample()

data_sample.proposals = InstanceData(data=dict(bboxes=torch.rand((5,4))))
print(data_sample)
```

```
<DetDataSample(

    META INFORMATION

    DATA FIELDS
    proposals: <InstanceData(

            META INFORMATION

            DATA FIELDS
            data:
                bboxes: tensor([[0.7513, 0.9275, 0.6169, 0.5581],
                            [0.6019, 0.6861, 0.7915, 0.0221],
                            [0.5977, 0.8987, 0.9541, 0.7877],
                            [0.0309, 0.1680, 0.1374, 0.0556],
                            [0.3842, 0.9965, 0.0747, 0.6546]])
        ) at 0x7f9f1c090310>
) at 0x7f9f1c090430>
```

```python
data_sample.proposals = torch.rand((5, 4))
```

```
AssertionError: tensor([[0.4370, 0.1661, 0.0902, 0.8421],
        [0.4947, 0.1668, 0.0083, 0.1111],
        [0.2041, 0.8663, 0.0563, 0.3279],
        [0.7817, 0.1938, 0.2499, 0.6748],
        [0.4524, 0.8265, 0.4262, 0.2215]]) should be a <class 'mmengine.data.instance_data.InstanceData'> but got <class 'torch.Tensor'>
```

## Simpify the interfaces

In this section, we use MMDetection to demonstrate how to migrate the abstract data interfaces to simplify the module and component interfaces. We suppose both `DetDataSample` and `InstanceData` have been implemented in MMDetection and MMEngine.

### 1. Simplify the module interface

Detector's external interfaces can be significantly simplified and unified. In the training process of a single-stage detection and segmentation algorithm in MMDet 2.X, `SingleStageDetector` requires `img`, `img_metas`, `gt_bboxes`, `gt_labels` and `gt_bboxes_ignore` as the inputs, but `SingleStageInstanceSegmentor` requires `gt_masks` as well. This causes inconsistency in the training interface and affects flexibility.

```python
class SingleStageDetector(BaseDetector):
    ...

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):


class SingleStageInstanceSegmentor(BaseDetector):
    ...

    def forward_train(self,
                      img,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
```

In MMDet 3.X, the training interfaces of all the detectors can be unified as `img` and `data_samples` using `DetDataSample`. Different modules can use `data_samples` to encapsulate their own attributes.

```python
class SingleStageDetector(BaseDetector):
    ...

    def forward_train(self,
                      img,
                      data_samples):

class SingleStageInstanceSegmentor(BaseDetector):
    ...

    def forward_train(self,
                      img,
                      data_samples):

```

### 2. Simplify the model interfaces

In MMDet 2.X, `HungarianAssigner` and `MaskHungarianAssigner` will be used to assign bboxes and instance segment information with annotated instances, respectively. The assignment logics of these two modules are the same, and the only differences are the interface and the calculation of the loss functions. However, this difference makes the code of `HungarianAssigner` cannot be directly used in `MaskHungarianAssigner`, which caused the redundancy.

```python
class HungarianAssigner(BaseAssigner):

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):

class MaskHungarianAssigner(BaseAssigner):

    def assign(self,
               cls_pred,
               mask_pred,
               gt_labels,
               gt_mask,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
```

In MMDet 3.X, `InstanceData` can encapsulate the bounding boxes, scores, and masks. With this, we can simplify the core parameters of `HungarianAssigner` to `pred_instances`, `gt_instances`, and `gt_instances_ignore`. This unifies the two assigners into one `HungarianAssianger`.

```python
class HungarianAssigner(BaseAssigner):

    def assign(self,
               pred_instances,
               gt_instancess,
               gt_instances_ignore=None,
               eps=1e-7):
```
