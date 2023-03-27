# BaseDataset

## Introduction

The Dataset class in the algorithm toolbox is responsible for providing input data for the model during the training/testing process. The Dataset class in each algorithm toolbox under OpenMMLab projects has some common characteristics and requirements, such as the need for efficient internal data storage format, support for the concatenation of different datasets, dataset repeated sampling, and so on.

Therefore, **MMEngine** implements [BaseDataset](mmengine.dataset.BaseDataset) which provides some basic interfaces and implements some DatasetWrappers with the same interfaces. Most of the Dataset Classes in the OpenMMLab algorithm toolbox meet the interface defined by the `BaseDataset` and use the same DatasetWrappers.

The basic function of the BaseDataset is to load the dataset information. Here, we divide the dataset information into two categories. One is meta information, which represents the information related to the dataset itself and sometimes needs to be obtained by the model or other external components. For example, the meta information of the dataset generally includes the category information `classes` in the image classification task, since the classification model usually needs to record the category information of the dataset. The other is data information, which defines the file path and corresponding label information of specific data info. In addition, another function of the BaseDataset is to continuously send data into the data pipeline for data preprocessing.

### The standard data annotation file

In order to unify the dataset interface of different tasks and facilitate multiple tasks training in one model, OpenMMLab formulate the **OpenMMLab 2.0 dataset format specification**. Dataset annotation files should conform to this specification, and the `BaseDataset` reads and parses data annotation files based on this specification. If the data annotation file provided by the user does not conform to the specified format, the user can choose to convert it to the specified format and use OpenMMLab's algorithm toolbox to conduct algorithm training and testing based on the converted data annotation file.

The OpenMMLab 2.0 dataset format specification states that annotation files must be in the format of `json` or `yaml`, `yml` or `pickle`, `pkl`. The dictionary stored in the annotation file must contain two fields, `metainfo` and `data_list`. The `metainfo` is a dictionary containing meta information about the dataset. The `data_list` is a list in which each element is a dictionary and the dictionary defines a raw data info. Each raw data info contains one or more training/test samples.

Here is an example of a JSON annotation file (where each raw data info contains only one training/test sample):

```json

{
    "metainfo":
        {
            "classes": ["cat", "dog"]
        },
    "data_list":
        [
            {
                "img_path": "xxx/xxx_0.jpg",
                "img_label": 0
            },
            {
                "img_path": "xxx/xxx_1.jpg",
                "img_label": 1
            }
        ]
}
```

We assume that the data is stored in the following path:

```text
data
├── annotations
│   ├── train.json
├── train
│   ├── xxx/xxx_0.jpg
│   ├── xxx/xxx_1.jpg
│   ├── ...
```

### The initialization process of the BaseDataset

The initialization process of the `BaseDataset` is shown as follows:

<div align="center">
  <img src="https://user-images.githubusercontent.com/26813582/201585974-1360e2b5-f95f-4273-8cbf-6024e33204ab.png" height="500"/>
</div>

1. `load metainfo`: Obtain the meta information of the dataset. The meta information can be obtained from three sources with the priority from high to low:

- The dict of `metainfo` passed by the user in the `__init__()` function. The priority is high since the user can pass this argument when the `BaseDataset` is instantiated;

- The dict of `BaseDataset.METAINFO` in the class attributes of BaseDataset. The priority is medium since the user can change the class attributes `BaseDataset.METAINFO` in the custom dataset class;

- The dict of `metainfo` included in the annotation file. The priority is low since the annotation file is generally not changed.

If three sources have the same field, the source with the highest priority determines the value of the field. The priority comparison of these fields is: The fields in the `metainfo` dictionary passed by the user > The fields in the `BaseDataset.METAINFO` of BaseDataset > the fields in the `metainfo` of annotation file.

2. `join path`: Process the path of datainfo and annotating files;

3. `build pipeline`: Build data pipeline for the data preprocessing and data preparation;

4. `full init`: Fully initializes the BaseDataset. This step mainly includes the following operations:

- `load data list`: Read and parse the annotation files that meet the OpenMMLab 2.0 dataset format specification. In this step, the `parse_data_info()` method is called. This method is responsible for parsing each raw data info in the annotation file;

- `filter data` (optional): Filters unnecessary data based on `filter_cfg`, such as data samples that do not contain annotations. By default, there is no filtering operation, and downstream subclasses can override it according to their own needs.

- `get subset` (optional): Sample a subset of dataset based on a given index or an integer value, such as only the first 10 samples for training/testing. By default, all data samples are used.

- `serialize data` (optional): Serialize all data samples to save memory. Please see [Save memory](#save-memory) for more details. we serialize all data samples by default.

The `parse_data_info()` method in the BaseDataset is used to process a raw data info in the annotation file into one or more training/test data samples. The user needs to implement the `parse_data_info()` method if they want to customize dataset class.

### The interface of BaseDataset

Once the BaseDataset is initialized, it supports `__getitem__` method to index a data info and `__len__` method to get the length of dataset, just like `torch.utils.data.Dataset`. The Basedataset provides the following interfaces:

- `metainfo`: Return the meta information with a dictionary value.

- `get_data_info(idx)`: Return the full data information of the given `idx`, and the return value is a dictionary.

- `__getitem__(idx)`: Return the results of data pipeline(The input data of model) of the given 'idx', and the return value is a dictionary.

- `__len__()`: Return the length of the dataset. The return value is an integer.

- `get_subset_(indices)`: Modify the original dataset class **in inplace** according to `indices`. If `indices` is `int`, then the original dataset class contains only the first few data samples. If `indices` is `Sequence[int]`, the raw dataset class contains data samples specified according to `Sequence[int]`.

- `get_subset(indices)`: Return a **new** sub-dataset class according to indices, i.e., re-copies a sub-dataset. If `indices` is `int`, the returned sub-dataset object contains only the first few data samples. If `indices` is `Sequence[int]`, the returned sub-dataset object contains the data samples specified according to `Sequence[int]`.

## Customize dataset class based on BaseDataset

We can customize the dataset class based on BaseDataset, after we understand the initialization process of BaseDataset and the provided interfaces of BaseDataset.

### Annotation files that meet the OpenMMLab 2.0 dataset format specification

As mentioned above, users can overload `parse_data_info()` to load annotation files that meet the OpenMMLab 2.0 dataset format specification. Here is an example of using BaseDataset to implement a specific dataset.

```python
import os.path as osp

from mmengine.dataset import BaseDataset


class ToyDataset(BaseDataset):

    # Take the above annotation file as example. The raw_data_info represents a dictionary in the data_list list:
    # {
    #    'img_path': "xxx/xxx_0.jpg",
    #    'img_label': 0,
    #    ...
    # }
    def parse_data_info(self, raw_data_info):
        data_info = raw_data_info
        img_prefix = self.data_prefix.get('img_path', None)
        if img_prefix is not None:
            data_info['img_path'] = osp.join(
                img_prefix, data_info['img_path'])
        return data_info

```

#### Using Customized dataset class

The `ToyDataset` can be instantiated with the following configuration, once it has been defined:

```python

class LoadImage:

    def __call__(self, results):
        results['img'] = cv2.imread(results['img_path'])
        return results

class ParseImage:

    def __call__(self, results):
        results['img_shape'] = results['img'].shape
        return results

pipeline = [
    LoadImage(),
    ParseImage(),
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline)
```

At the same time, the external interface provided by the BaseDataset can be used to access specific data sample information:

```python
toy_dataset.metainfo
# dict(classes=('cat', 'dog'))

toy_dataset.get_data_info(0)
# {
#     'img_path': "data/train/xxx/xxx_0.jpg",
#     'img_label': 0,
#     ...
# }

len(toy_dataset)
# 2

toy_dataset[0]
# {
#     'img_path': "data/train/xxx/xxx_0.jpg",
#     'img_label': 0,
#     'img': a ndarray with shape (H, W, 3), which denotes the value of the image,
#     'img_shape': (H, W, 3) ,
#     ...
# }

# The `get_subset` interface does not modify the original dataset class, i.e. make a complete copy of it
sub_toy_dataset = toy_dataset.get_subset(1)
len(toy_dataset), len(sub_toy_dataset)
# 2, 1

# The `get_subset_` interface modify the original dataset class in inplace
toy_dataset.get_subset_(1)
len(toy_dataset)
# 1
```

Following the above steps, we can see how to customize a dataset based on the BaseDataset and how to use the customized dataset.

#### Customize dataset for videos

In the above examples, each raw data info of the annotation file contains only one training/test sample (usually in the image field). If each raw data info contains several training/test samples (usually in the video domain), we only need to ensure that the return value of `parse_data_info()` is `list[dict]`:

```python
from mmengine.dataset import BaseDataset


class ToyVideoDataset(BaseDataset):

    # raw_data_info is still a dict, but it contains multiple samples
    def parse_data_info(self, raw_data_info):
        data_list = []

        ...

        for ... :

            data_info = dict()

            ...

            data_list.append(data_info)

        return data_list

```

The usage of `ToyVideoDataset` is similar to that of `ToyDataset`, which will not be repeated here.

### Annotation files that do not meet the OpenMMLab 2.0 dataset format specification

For annotated files that do not meet the OpenMMLab 2.0 dataset format specification, there are two ways to use:

1. Convert the annotation files that do not meet the specifications into the annotation files that do meet the specifications, and then use the BaseDataset in the above way.

2. Implement a new dataset class that inherits from the `BaseDataset` and overloads the `load_data_list(self):` function of the `BaseDataset` to handle annotation files that don't meet the specification and guarantee a return value of `list[dict]`, where each `dict` represents a data sample.

## Other features of BaseDataset

The BaseDataset also contains the following features:

### lazy init

When the BaseDataset is instantiated, the annotation file needs to be read and parsed, therefore it will take some time. However, in some cases, such as the visualization of prediction, only the meta information of the BaseDataset is required, and reading and parsing the annotation file may not be necessary. To save time on instantiating the BaseDataset in this case, the BaseDataset supports lazy init:

```python
pipeline = [
    LoadImage(),
    ParseImage(),
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline,
    # Pass the lazy_init variable in here
    lazy_init=True)
```

When `lazy_init=True`, the initialization of ToyDataset's only performs steps 1, 2, and 3 of the BaseDataset initialization process. At this time, `toy_dataset` was not fully initialized, since `toy_dataset` will not read and parse the annotation file. The `toy_dataset` only set the meta information of the dataset (`metainfo`).

Naturally, if you need to access specific data information later, you can manually call the `toy_dataset.full_init()` interface to perform the complete initialization process, during which the data annotation file will be read and parsed. Calling the `get_data_info (independence idx)`, `__len__ ()`, `__getitem__ (independence idx)`, ` get_subset_ (indices)` and `get_subset(indices)` interface will also automatically call the `full_init()` interface to perform the full initialization process (only on the first call, later calls will not call the `full_init()` interface repeatedly):

```python
# Full initialization
toy_dataset.full_init()

# After initialization, you can now get the data info
len(toy_dataset)
# 2
toy_dataset[0]
# {
#     'img_path': "data/train/xxx/xxx_0.jpg",
#     'img_label': 0,
#     'img': a ndarray with shape (H, W, 3), which denotes the value the image,
#     'img_shape': (H, W, 3) ,
#     ...
# }
```

**Notice:**

Performing full initialization by calling the `__getitem__()` interface directly carries some risks: If a dataset object is not fully initialized by setting `lazy_init=True` firstly, then it is directly sent to the dataloader. Different dataloader workers will read and parse the annotation file at the same time in the subsequent data reading process. Although this may work normally, it consumes a lot of time and memory. **Therefore, it is recommended to manually call the `full_init()` interface to perform the full initialization process before you need to access specific data.**

The above is not fully initialized by setting `lazy_init=True`, and then complete initialization according to the demand, called lazy init.

### Save memory

In the specific process of reading data, the dataloader will usually prefetch data from multiple dataloader workers, and multiple workers have complete dataset object backup, so there will be multiple copies of the same `data_list` in the memory. In order to save this part of memory consumption, The `BaseDataset` can serialize `data_list` into memory in advance, so that multiple workers can share the same copy of `data_list`, so as to save memory.

By default, the BaseDataset stores the serialization of `data_list` into memory. It is also possible to control whether the data will be serialized into memory ahead of time by using the  `serialize_data` argument (default is `True`) :

```python
pipeline = [
    LoadImage(),
    ParseImage(),
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline,
    # Pass the serialize data argument in here
    serialize_data=False)
```

The above example does not store the `data_list` serialization into memory in advance, so it is not recommended to instantiate the dataset class, when using the dataloader to open multiple dataloader workers to load the data.

## DatasetWrappers

In addition to BaseDataset, MMEngine also provides several DatasetWrappers: `ConcatDataset`, `RepeatDataset`, `ClassBalancedDataset`. These dataset wrappers also support lazy init and have memory-saving features.

### ConcatDataset

MMEngine provides a `ConcatDataset` wrapper to concatenate datasets in the following way:

```python
from mmengine.dataset import ConcatDataset

pipeline = [
    LoadImage(),
    ParseImage(),
]

toy_dataset_1 = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline)

toy_dataset_2 = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='val/'),
    ann_file='annotations/val.json',
    pipeline=pipeline)

toy_dataset_12 = ConcatDataset(datasets=[toy_dataset_1, toy_dataset_2])

```

The above example combines the `train` set and the `val` set of the dataset into one large dataset.

### RepeatDataset

MMEngine provides `RepeatDataset` wrapper to repeat a dataset several times, as follows:

```python
from mmengine.dataset import RepeatDataset

pipeline = [
    LoadImage(),
    ParseImage(),
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline)

toy_dataset_repeat = RepeatDataset(dataset=toy_dataset, times=5)

```

The above example samples the `train` set of the dataset five times.

### ClassBalancedDataset

MMEngine provides `ClassBalancedDataset` wrapper to repeatedly sample the corresponding samples based on the frequency of category occurrence in the dataset.

**Notice:**

The `ClassBalancedDataset` wrapper assumes that the wrapped dataset class supports the `get_cat_ids(idx)` method, which returns a list. The list contains the categories of  `data_info` given by 'idx'. The usage is as follows:

```python
from mmengine.dataset import BaseDataset, ClassBalancedDataset

class ToyDataset(BaseDataset):

    def parse_data_info(self, raw_data_info):
        data_info = raw_data_info
        img_prefix = self.data_prefix.get('img_path', None)
        if img_prefix is not None:
            data_info['img_path'] = osp.join(
                img_prefix, data_info['img_path'])
        return data_info

    # The necessary method that needs to return the category of data sample
    def get_cat_ids(self, idx):
        data_info = self.get_data_info(idx)
        return [int(data_info['img_label'])]

pipeline = [
    LoadImage(),
    ParseImage(),
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline)

toy_dataset_repeat = ClassBalancedDataset(dataset=toy_dataset, oversample_thr=1e-3)

```

The above example resamples the `train` set of the dataset with `oversample_thr=1e-3`. Specifically, for categories whose frequency is less than `1e-3` in the dataset, samples corresponding to this category will be sampled repeatedly; otherwise, samples will not be sampled repeatedly. Please refer to the API documentation of `ClassBalancedDataset` for specific sampling policies.

### Customize DatasetWrapper

Since the BaseDataset support lazy init, some rules need to be followed when customizing the DatasetWrapper. Here is an example to show how to customize the DatasetWrapper:

```python
from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS


@DATASETS.register_module()
class ExampleDatasetWrapper:

    def __init__(self, dataset, lazy_init=False, ...):
        # Build the source dataset (self.dataset)
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        # Record the meta information of source dataset
        self._metainfo = self.dataset.metainfo

        '''
        1. Implement some code here to record some of the hyperparameters used to wrap the dataset.
        '''

        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    def full_init(self):
        if self._fully_initialized:
            return

        # Initialize the source dataset completely
        self.dataset.full_init()

        '''
        2. Implement some code here to wrap the source dataset.
        '''

        self._fully_initialized = True

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int):

        '''
        3. Implement some code here to map the wrapped index `idx` to the index of the source dataset 'ori_idx'.
        '''
        ori_idx = ...

        return ori_idx

    # Provide the same external interface as `self.dataset `.
    @force_full_init
    def get_data_info(self, idx):
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    # Provide the same external interface as `self.dataset `.
    def __getitem__(self, idx):
        if not self._fully_initialized:
            warnings.warn('Please call `full_init` method manually to '
                          'accelerate the speed.')
            self.full_init()

        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset[sample_idx]

    # Provide the same external interface as `self.dataset `.
    @force_full_init
    def __len__(self):

        '''
        4. Implement some code here to calculate the length of the wrapped dataset.
        '''
        len_wrapper = ...

        return len_wrapper

    # Provide the same external interface as `self.dataset `.
    @property
    def metainfo(self)
        return copy.deepcopy(self._metainfo)
```
