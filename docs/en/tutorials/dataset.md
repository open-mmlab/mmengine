# Dataset and DataLoader

```{hint}
If you have never been exposed to PyTorch's Dataset and DataLoader classes, you are recommended to read through [PyTorch official tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) to get familiar with some basic concepts.
```

Datasets and DataLoaders are necessary components in MMEngine's training pipeline. They are conceptually derived from and consistent with PyTorch. Typically, a dataset defines the quantity, parsing, and pre-processing of the data, while a dataloader iteratively loads data according to settings such as `batch_size`, `shuffle`, `num_workers`, etc. Datasets are encapsulated with dataloaders and they together constitute the data source.

In this tutorial, we will step through their usage in MMEngine runner from the outside (dataloader) to the inside (dataset) and give some practical examples. After reading through this tutorial, you will be able to:

- Master the configuration of dataloaders in MMEngine
- Learn to use existing datasets (e.g. those from `torchvision`) from config files
- Know about building and using your own dataset

## Details on dataloader

Dataloaders can be configured in MMEngine's `Runner` with 3 arguments:

- `train_dataloader`: Used in `Runner.train()` to provide training data for models
- `val_dataloader`: Used in `Runner.val()` or in `Runner.train()` at regular intervals for model evaluation
- `test_dataloader`: Used in `Runner.test()` for the final test

MMEngine has full support for PyTorch native `DataLoader` objects. Therefore, you can simply pass your valid, already built dataloaders to the runner, as shown in [getting started in 15 minutes](../get_started/15_minutes.md). Meanwhile, thanks to the [Registry Mechanism](../advanced_tutorials/registry.md) of MMEngine, those arguments also accept `dict`s as inputs, as illustrated in the following example (referred to as example 1). The keys in the dictionary correspond to arguments in DataLoader's init function.

```python
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=torchvision.datasets.CIFAR10(...),
        collate_fn=dict(type='default_collate')
    )
)
```

When passed to the runner in the form of a dict, the dataloader will be lazily built in the runner when actually needed.

```{note}
For more configurable arguments of the `DataLoader`, please refer to [PyTorch API documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
```

```{note}
If you are interested in the details of the building procedure, you may refer to [build_dataloader](mmengine.runner.Runner.build_dataloader)
```

You may find example 1 differs from that in [getting started in 15 minutes](../get_started/15_minutes.md) in some arguments. Indeed, due to some obscure conventions in MMEngine, you can't seamlessly switch it to a dict by simply replacing `DataLoader` with `dict`. We will discuss the differences between our convention and PyTorch's in the following sections, in case you run into trouble when using config files.

### sampler and shuffle

One obvious difference is that we add a `sampler` argument to the dict. This is because we **require `sampler` to be explicitly specified** when using a dict as a dataloader. Meanwhile, `shuffle` is also removed from `DataLoader` arguments, because it conflicts with `sampler` in PyTorch, as referred to in [PyTorch DataLoader API documentation](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

```{note}
In fact, `shuffle` is just a notation for convenience in PyTorch implementation. If `shuffle` is set to `True`, the dataloader will automatically switch to `RandomSampler`
```

With a `sampler` argument, codes in example 1 is **nearly** equivalent to code block below

```python
from mmengine.dataset import DefaultSampler

dataset = torchvision.datasets.CIFAR10(...)
sampler = DefaultSampler(dataset, shuffle=True)

runner = Runner(
    train_dataloader=DataLoader(
        batch_size=32,
        sampler=sampler,
        dataset=dataset,
        collate_fn=default_collate
    )
)
```

```{warning}
The equivalence of the above codes holds only if: 1) you are training with a single process, and 2) no `randomness` argument is passed to the runner. This is due to the fact that `sampler` should be built after distributed environment setup to be correct. The runner will guarantee the correct order and proper random seed by applying lazy initialization techniques, which is only possible for dict inputs. Instead, when building a sampler manually, it requires extra work and is highly error-prone. Therefore, the code block above is just for illustration and definitely not recommended. We **strongly suggest passing `sampler` as a `dict`** to avoid potential problems.
```

### DefaultSampler

The above example may make you wonder what a `DefaultSampler` is, why use it and whether there are other options. In fact, `DefaultSampler` is a built-in sampler in MMEngine which eliminates the gap between distributed and non-distributed training and thus enabling a seamless conversion between them. If you have the experience of using `DistributedDataParallel` in PyTorch, you may be impressed by having to change the `sampler` argument to make it correct. However, in MMEngine, you don't need to bother with this `DefaultSampler`.

`DefaultSampler` accepts the following arguments:

- `shuffle`: Set to `True` to load data in the dataset in random order
- `seed`: Random seed used to shuffle the dataset. Typically it doesn't require manual configuration here because the runner will handle it with `randomness` configuration
- `round_up`: When set this to `True`, this is the same behavior as setting `drop_last=False` in PyTorch `DataLoader`. You should take care of it when doing migration from PyTorch.

```{note}
For more details about `DefaultSampler`, please refer to [its API docs](mmengine.dataset.DefaultSampler)
```

`DefaultSampler` handles most of the cases. We ensure that error-prone details such as random seeds are handled properly when you are using it in a runner. This prevents you from getting into troubles with distributed training. Apart from `DefaultSampler`, you may also be interested in [InfiniteSampler](mmengine.dataset.InfiniteSampler) for iteration-based training pipelines. If you have more advanced demands, you may want to refer to the codes of these two built-in samplers to implement your own one and register it to `DATA_SAMPLERS` registry.

```python
@DATA_SAMPLERS.register_module()
class MySampler(Sampler):
    pass

runner = Runner(
    train_dataloader=dict(
        sampler=dict(type='MySampler'),
        ...
    )
)
```

### The obscure collate_fn

Among the arguments of PyTorch `DataLoader`, `collate_fn` is often ignored by users, but in MMEngine you must pay special attention to it. When you pass the dataloader argument as a dict, MMEngine will use the built-in [pseudo_collate](mmengine.dataset.pseudo_collate) by default, which is significantly different from that, [default_collate](https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate), in PyTorch. Therefore, when doing a migration from PyTorch, you have to explicitly specify the `collate_fn` in config files to be consistent in behavior.

```{note}
MMEngine uses `pseudo_collate` as default value is mainly due to historical compatibility reasons. You don't have to look deeply into it. You can just know about it and avoid potential errors.
```

MMEngine provides 2 built-in `collate_fn`:

- `pseudo_collate`: Default value in MMEngine. It won't concatenate data through `batch` index. Detailed explanations can be found in [pseudo_collate API doc](mmengine.dataset.pseudo_collate)
- `default_collate`: It behaves almost identically to PyTorch's `default_collate`. It will transfer data into `Tensor` and concatenate them through `batch` index. More details and slight differences from PyTorch can be found in [default_collate API doc](mmengine.dataset.default_collate)

If you want to use a custom `collate_fn`, you can register it to `FUNCTIONS` registry.

```python
@FUNCTIONS.register_module()
def my_collate_func(data_batch: Sequence) -> Any:
    pass

runner = Runner(
    train_dataloader=dict(
        ...
        collate_fn=dict(type='my_collate_func')
    )
)
```

## Details on dataset

Typically, datasets define the quantity, parsing, and pre-processing of the data. It is encapsulated in dataloader, allowing the latter to load data in batches. Since we fully support PyTorch `DataLoader`, the dataset is also compatible. Meanwhile, thanks to the registry mechanism, when a dataloader is given as a dict, its `dataset` argument can also be given as a dict, which enables lazy initialization in the runner. This mechanism allows for writing config files.

### Use torchvision datasets

`torchvision` provides various open datasets. They can be directly used in MMEngine as shown in [getting started in 15 minutes](../get_started/15_minutes.md), where a `CIFAR10` dataset is used together with torchvision's built-in data transforms.

However, if you want to use the dataset in config files, registration is needed. What's more, if you also require data transforms in torchvision, some more registrations are required. The following example illustrates how to do it.

```python
import torchvision.transforms as tvt
from mmengine.registry import DATASETS, TRANSFORMS
from mmengine.dataset.base_dataset import Compose

# register CIFAR10 dataset in torchvision
# data transforms should also be built here
@DATASETS.register_module(name='Cifar10', force=False)
def build_torchvision_cifar10(transform=None, **kwargs):
    if isinstance(transform, dict):
        transform = [transform]
    if isinstance(transform, (list, tuple)):
        transform = Compose(transform)
    return torchvision.datasets.CIFAR10(**kwargs, transform=transform)

# register data transforms in torchvision
DATA_TRANSFORMS.register_module('RandomCrop', module=tvt.RandomCrop)
DATA_TRANSFORMS.register_module('RandomHorizontalFlip', module=tvt.RandomHorizontalFlip)
DATA_TRANSFORMS.register_module('ToTensor', module=tvt.ToTensor)
DATA_TRANSFORMS.register_module('Normalize', module=tvt.Normalize)

# specify in runner
runner = Runner(
    train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
        dataset=dict(type='Cifar10',
            root='data/cifar10',
            train=True,
            download=True,
            transform=[
                dict(type='RandomCrop', size=32, padding=4),
                dict(type='RandomHorizontalFlip'),
                dict(type='ToTensor'),
                dict(type='Normalize', **norm_cfg)])
    )
)
```

```{note}
The above example makes extensive use of the registry mechanism and borrows the [Compose](mmengine.dataset.Compose) module from MMEngine. If you urge to use torchvision dataset in your config files, you can refer to it and make some slight modifications. However, we recommend you borrow datasets from downstream repos such as [MMDet](https://github.com/open-mmlab/mmdetection), [MMPretrain](https://github.com/open-mmlab/mmpretrain), etc. This may give you a better experience.
```

### Customize your dataset

You are free to customize your own datasets, as you would with PyTorch. You can also copy existing datasets from your previous PyTorch projects. If you want to learn how to customize your dataset, please refer to [PyTorch official tutorials](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)

### Use MMEngine BaseDataset

Apart from directly using PyTorch native `Dataset` class, you can also use MMEngine's built-in class `BaseDataset` to customize your own one, as referred to [BaseDataset tutorial](../advanced_tutorials/basedataset.md). It makes some conventions on the format of annotation files, which makes the data interface more unified and multi-task training more convenient. Meanwhile, `BaseDataset` can easily cooperate with built-in [data transforms](../advanced_tutorials/data_element.md) in MMEngine, which releases you from writing one from scratch.

Currently, `BaseDataset` has been widely used in downstream repos of OpenMMLab 2.0 projects.
