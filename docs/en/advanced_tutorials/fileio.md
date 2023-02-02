# File IO

`MMEngine` implements a unified set of file reading and writing interfaces in `fileio` module. With the `fileio` module, we can use the same function to handle different file formats, such as `json`, `yaml` and `pickle`. Other file formats can also be easily extended.

The `fileio` module also supports reading and writing files from a variety of file storage backends, including disk, Petrel (for internal use), Memcached, LMDB, and HTTP.

## Load and dump data

`MMEngine` provides a universal API for loading and dumping data, currently supported formats are `json`, `yaml`, and `pickle`.

### Load from disk or dump to disk

```python
from mmengine import load, dump

# load data from a file
data = load('test.json')
data = load('test.yaml')
data = load('test.pkl')
# load data from a file-like object
with open('test.json', 'r') as f:
    data = load(f, file_format='json')

# dump data to a string
json_str = dump(data, file_format='json')

# dump data to a file with a filename (infer format from file extension)
dump(data, 'out.pkl')

# dump data to a file with a file-like object
with open('test.yaml', 'w') as f:
    data = dump(data, f, file_format='yaml')
```

### Load from other backends or dump to other backends

```python
from mmengine import load, dump

# load data from a file
data = load('s3://bucket-name/test.json')
data = load('s3://bucket-name/test.yaml')
data = load('s3://bucket-name/test.pkl')

# dump data to a file with a filename (infer format from file extension)
dump(data, 's3://bucket-name/out.pkl')
```

It is also very convenient to extend the API to support more file formats. All you need to do is to write a file handler inherited from `BaseFileHandler` and register it with one or several file formats.

```python
from mmengine import register_handler, BaseFileHandler

# To register multiple file formats, a list can be used as the argument.
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

Here is an example of `PickleHandler`:

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

## Load a text file as a list or dict

For example `a.txt` is a text file with 5 lines.

```
a
b
c
d
e
```

### Load from disk

Use `list_from_file` to load the list from `a.txt`:

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

For example `b.txt` is a text file with 3 lines.

```
1 cat
2 dog cow
3 panda
```

Then use `dict_from_file` to load the dict from `b.txt`:

```python
from mmengine import dict_from_file

print(dict_from_file('b.txt'))
# {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
print(dict_from_file('b.txt', key_type=int))
# {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```

### Load from other backends

Use `list_from_file` to load the list from `s3://bucket-name/a.txt`:

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

Use `dict_from_file` to load the dict from `s3://bucket-name/b.txt`.

```python
from mmengine import dict_from_file

print(dict_from_file('s3://bucket-name/b.txt'))
# {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
print(dict_from_file('s3://bucket-name/b.txt', key_type=int))
# {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}
```

## Load and dump checkpoints

We can read the checkpoints from disk or internet in the following way:

```python
import torch

filepath1 = '/path/of/your/checkpoint1.pth'
filepath2 = 'http://path/of/your/checkpoint3.pth'

# read checkpoints from disk
checkpoint = torch.load(filepath1)
# save checkpoints to disk
torch.save(checkpoint, filepath1)

# read checkpoints from internet
checkpoint = torch.utils.model_zoo.load_url(filepath2)
```

In `MMEngine`, reading and writing checkpoints in different storage forms can be uniformly implemented with `load_checkpoint` and `save_checkpoint`:

```python
from mmengine import load_checkpoint, save_checkpoint

filepath1 = '/path/of/your/checkpoint1.pth'
filepath2 = 's3://bucket-name/path/of/your/checkpoint1.pth'
filepath3 = 'http://path/of/your/checkpoint3.pth'

# read checkpoints from disk
checkpoint = load_checkpoint(filepath1)
# save checkpoints to disk
save_checkpoint(checkpoint, filepath1)

# read checkpoints from s3
checkpoint = load_checkpoint(filepath2)
# save checkpoints to s3
save_checkpoint(checkpoint, filepath2)

# read checkpoints from internet
checkpoint = load_checkpoint(filepath3)
```
