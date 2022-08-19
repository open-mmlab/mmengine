# 数据集基类（BaseDataset）

## 数据集基类的其它特性

数据集基类还包含以下特性：

### 懒加载（lazy init）

在数据集类实例化时，需要读取并解析标注文件，因此会消耗一定时间。然而在某些情况比如预测可视化时，往往只需要数据集类的元信息，可能并不需要读取与解析标注文件。为了节省这种情况下数据集类实例化的时间，数据集基类支持懒加载：

```python
pipeline = [
    dict(type='xxx', ...),
    dict(type='yyy', ...),
    ...
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline,
    # 在这里传入 lazy_init 变量
    lazy_init=True)
```

当 `lazy_init=True` 时，`ToyDataset` 的初始化方法只执行了[数据集基类的初始化流程](#%E6%95%B0%E6%8D%AE%E9%9B%86%E5%9F%BA%E7%B1%BB%E7%9A%84%E5%88%9D%E5%A7%8B%E5%8C%96%E6%B5%81%E7%A8%8B)中的 1、2、3 步骤，此时 `toy_dataset` 并未被完全初始化，因为 `toy_dataset` 并不会读取与解析标注文件，只会设置数据集类的元信息（`metainfo`）。

自然的，如果之后需要访问具体的数据信息，可以手动调用 `toy_dataset.full_init()` 接口来执行完整的初始化过程，在这个过程中数据标注文件将被读取与解析。调用 `get_data_info(idx)`, `__len__()`, `__getitem__(idx)`，`get_subset_(indices)`， `get_subset(indices)` 接口也会自动地调用 `full_init()` 接口来执行完整的初始化过程（仅在第一次调用时，之后调用不会重复地调用 `full_init()` 接口）：

```python
# 完整初始化
toy_dataset.full_init()

# 初始化完毕，现在可以访问具体数据
len(toy_dataset) # 2
toy_dataset[0] # dict(img=xxx, label=0)
```

**注意：**

通过直接调用 `__getitem__()` 接口来执行完整初始化会带来一定风险：如果一个数据集类首先通过设置 `lazy_init=True` 未进行完全初始化，然后直接送入数据加载器（dataloader）中，在后续读取数据的过程中，不同的 worker 会同时读取与解析标注文件，虽然这样可能可以正常运行，但是会消耗大量的时间与内存。**因此，建议在需要访问具体数据之前，提前手动调用 `full_init()` 接口来执行完整的初始化过程。**

以上通过设置 `lazy_init=True` 未进行完全初始化，之后根据需求再进行完整初始化的方式，称为懒加载。

### 节省内存

在具体的读取数据过程中，数据加载器（dataloader）通常会起多个 worker 来预取数据，多个 worker 都拥有完整的数据集类备份，因此内存中会存在多份相同的 `data_list`，为了节省这部分内存消耗，数据集基类可以提前将 `data_list` 序列化存入内存中，使得多个 worker 可以共享同一份 `data_list`，以达到节省内存的目的。

数据集基类默认是将 `data_list` 序列化存入内存，也可以通过 `serialize_data` 变量（默认为 `True`）来控制是否提前将 `data_list` 序列化存入内存中：

```python
pipeline = [
    dict(type='xxx', ...),
    dict(type='yyy', ...),
    ...
]

toy_dataset = ToyDataset(
    data_root='data/',
    data_prefix=dict(img_path='train/'),
    ann_file='annotations/train.json',
    pipeline=pipeline,
    # 在这里传入 serialize_data 变量
    serialize_data=False)
```

上面例子不会提前将 `data_list` 序列化存入内存中，因此不建议在使用数据加载器开多个 worker 加载数据的情况下，使用这种方式实例化数据集类。