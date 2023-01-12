# 文件读写

`MMEngine` 在 `fileio` 模块中实现了一套统一的文件读写接口。通过 `fileio` 模块，我们可以用同一个函数来处理不同的文件格式，如 `json`、`yaml` 和 `pickle`，并且可以方便地拓展其它的文件格式。

`fileio` 模块也支持从多种文件存储后端读写文件，包括本地磁盘、Petrel（内部使用）、Memcached、LMDB 和 HTTP。

## 读取和保存数据

`MMEngine` 为读取和保存数据提供了统一的 API 接口，目前支持的格式有 `json`、`yaml` 和 `pickle`。

### 从硬盘中读写文件

```python
from mmengine import load, dump

# 从文件中读取数据
data = load('test.json')
data = load('test.yaml')
data = load('test.pkl')
# 从文件对象中读取数据
with open('test.json', 'r') as f:
    data = load(f, file_format='json')

# 将数据序列化为字符串
json_str = dump(data, file_format='json')

# 将数据保存至文件 (根据文件名后缀反推文件类型)
dump(data, 'out.pkl')

# 将数据保存至文件对象
with open('test.yaml', 'w') as f:
    data = dump(data, f, file_format='yaml')
```

### 从其它文件存储后端读写文件

```python
from mmengine import load, dump

# 从 s3 文件中读取数据
data = load('s3://bucket-name/test.json')
data = load('s3://bucket-name/test.yaml')
data = load('s3://bucket-name/test.pkl')

# 将数据保存至 s3 文件（根据文件名后缀反推文件类型）
dump(data, 's3://bucket-name/out.pkl')
```

拓展 API 以支持更多的文件格式是很方便的。你所需要做的是写一个继承自 `BaseFileHandler` 的文件句柄，并使用一个或者多个文件格式来注册它。

```python
from mmengine import register_handler, BaseFileHandler

# 为了注册多个文件格式，可以使用列表作为参数。
# @register_handler(['txt', 'log'])
@register_handler('txt')
class TxtHandler1(BaseFileHandler):

    def load_from_fileobj(self, file):
        return file.read()

    def dump_to_fileobj(self, obj, file):
        file.write(str(obj))

    def dump_to_str(self, obj, **kwargs):
        return str(obj)
```

以 `PickleHandler` 为例：

```python
from mmengine import BaseFileHandler
import pickle

class PickleHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        return pickle.load(file, **kwargs)

    def load_from_path(self, filepath, **kwargs):
        return super(PickleHandler, self).load_from_path(
            filepath, mode='rb', **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('protocol', 2)
        return pickle.dumps(obj, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('protocol', 2)
        pickle.dump(obj, file, **kwargs)

    def dump_to_path(self, obj, filepath, **kwargs):
        super(PickleHandler, self).dump_to_path(
            obj, filepath, mode='wb', **kwargs)
```

## 读取文件并返回列表或字典

例如，`a.txt` 是文本文件，一共有 5 行内容。

```
a
b
c
d
e
```

### 从硬盘读取

使用 `list_from_file` 从 `a.txt` 中读取列表：

```python
from mmengine import list_from_file

print(list_from_file('a.txt'))
# ['a', 'b', 'c', 'd', 'e']
print(list_from_file('a.txt', offset=2))
# ['c', 'd', 'e']
print(list_from_file('a.txt', max_num=2))
# ['a', 'b']
print(list_from_file('a.txt', prefix='/mnt/'))
# ['/mnt/a', '/mnt/b', '/mnt/c', '/mnt/d', '/mnt/e']
```

例如，`b.txt` 是文本文件，一共有 3 行内容。

```
1 cat
2 dog cow
3 panda
```

使用 `dict_from_file` 从 `b.txt` 中读取字典：

```python
from mmengine import dict_from_file

print(dict_from_file('b.txt'))
# {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
print(dict_from_file('b.txt', key_type=int))
# {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```

### 从其他存储后端读取

使用 `list_from_file` 从 `s3://bucket-name/a.txt` 中读取列表：

```python
from mmengine import list_from_file

print(list_from_file('s3://bucket-name/a.txt'))
# ['a', 'b', 'c', 'd', 'e']
print(list_from_file('s3://bucket-name/a.txt', offset=2))
# ['c', 'd', 'e']
print(list_from_file('s3://bucket-name/a.txt', max_num=2))
# ['a', 'b']
print(list_from_file('s3://bucket-name/a.txt', prefix='/mnt/'))
# ['/mnt/a', '/mnt/b', '/mnt/c', '/mnt/d', '/mnt/e']
```

使用 `dict_from_file` 从 `s3://bucket-name/b.txt` 中读取字典：

```python
from mmengine import dict_from_file

print(dict_from_file('s3://bucket-name/b.txt'))
# {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
print(dict_from_file('s3://bucket-name/b.txt', key_type=int))
# {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```

## 读取和保存权重文件

通常情况下，我们可以通过下面的方式从磁盘或网络远端读取权重文件：

```python
import torch

filepath1 = '/path/of/your/checkpoint1.pth'
filepath2 = 'http://path/of/your/checkpoint3.pth'

# 从本地磁盘读取权重文件
checkpoint = torch.load(filepath1)
# 保存权重文件到本地磁盘
torch.save(checkpoint, filepath1)

# 从网络远端读取权重文件
checkpoint = torch.utils.model_zoo.load_url(filepath2)
```

在 `MMEngine` 中，在不同存储格式中读取权重文件可以通过 `load_checkpoint` 和 `save_checkpoint` 来统一实现：

```python
from mmengine import load_checkpoint, save_checkpoint

filepath1 = '/path/of/your/checkpoint1.pth'
filepath2 = 's3://bucket-name/path/of/your/checkpoint1.pth'
filepath3 = 'http://path/of/your/checkpoint3.pth'

# 从本地磁盘读取权重文件
checkpoint = load_checkpoint(filepath1)
# 保存权重文件到本地磁盘
save_checkpoint(checkpoint, filepath1)

# 从 s3 读取权重文件
checkpoint = load_checkpoint(filepath2)
# 保存权重文件到 s3
save_checkpoint(checkpoint, filepath2)

# 从网络远端读取权重文件
checkpoint = load_checkpoint(filepath3)
```
