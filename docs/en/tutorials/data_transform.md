# Data Transform

Before the data is feed in models, it need some processing, for loading,
formatting or augmentation. Same as PyTorch, we use **data transforms** to
manipulate the data. And a series of data transforms compose a data pipeline.

Every transform in the data pipeline defines a step to manipulate the data dict.
And each transform takes a dict as input and also output a dict for the next
transform.

In MMEngine, we provide a `BaseTransform` and some useful transform wrappers to
help us build a flexible and powerful data pipeline.

## BaseTransform

The `BaseTransform` is an abstract class, and it only defines the inferface for
a data transform class.

A data transform class just needs to inherit the `BaseTransform` class and
implement the `transform` function. Here, we use a simplified `RandomFlip` as
an example.

```python
import random
import mmcv
from mmengine.data import BaseTransform

class RandomFlip(BaseTransform):
    def transform(self, results: dict) -> dict:
        img = results['img']
        flip = True if random.random() > 0.5 else False
        if flip:
            results['img'] = mmcv.imflip(img)
        return results
```

And then, we can instance a `RandomFlip` object and use it as a callable object
to process the data dict.

```python
import numpy as np
transform = RandomFlip()
data_info = dict(img=np.random.rand(224, 224, 3))
data_info = transform(data_info)
processed_img = data_info['img']
```

## Transform Wrappers

Transform wrappers is a series of special transforms. They don't manipulate
image/label data, but enhance the behavior of data transforms.

### Remap

The `Remap` wrapper is to remap the keys in the data dict. For example, usually
image processing transforms will get the value of `"img"` in the data dict. But
sometimes, we want them to process the image of other key, like `"gt_img"`.

```python
pipeline = [
    ...
    dict(type='Remap',
        input_mapping={'img': 'gt_img'},  # Map the "gt_img" key to "img" key.
        inplace=True,  # After data transforms, revert map the "img" key to "gt_img" key.
        transforms=[
            # `RandomFlip` just reads the "img" key and modify the value.
            dict(type='RandomFlip'),
        ])
    ...
]
```

With the `Remap` wrapper, we don't worry about which key to get in the
`transform` function of data transforms.

### RandomChoice

The `RandomChoice` wrapper is to apply a single transforms combination randomly
picked from a list. We can use it to implement augmentation like AutoAugment
simply.

```python
pipeline = [
    ...
    dict(type='RandomChoice',
        pipelines=[
            [
                dict(type='Posterize', bits=4),
                dict(type='Rotate', angle=30.)
            ],  # transforms combination 1
            [
                dict(type='Equalize'),
                dict(type='Rotate', angle=30)
            ],  # transforms combination 2
        ],
        pipeline_probs=[0.4, 0.6]  # The choice probability of two pipelines.
        )
    ...
]
```

### ApplyToMultiple

Usually, a data transform will only get the manipulation target by a fixed key,
although you can change the key by the `Remap` wrapper. The `ApplyToMultiple`
wrapper can help us apply data transforms on multiple values of the data dict.

It has two usages, the first is specify multiple keys to apply the data
transforms, and the second is specify a single key with a list of targets to
apply data transforms.

1. Specify multiple keys

   Assume we need to apply transformations on images of `"lq"` (low-quanlity)
   and `"gt"` (ground-truth).
   
   ```python
   pipeline = [
       dict(type='ApplyToMultiple',
           # Apply data transforms on both "lq" and "gt", and map the key to "img"
           input_mapping={'img': ['lq', 'gt']},
           # After data transforms, revert map the "img" key to the original keys.
           inplace=True,
           # Whether to share random variables.
           # More details can be found in the below section.
           share_random_param=True,
           transforms=[
               # `RandomFlip` just reads the "img" key and modify the value.
               dict(type='RandomFlip'),
           ])
   ]
   ```

2. Specify a key with multiple targets

   Assume we need to apply transformations on a list of images in the key
   `"images"`.
   
   ```python
   pipeline = [
       dict(type='ApplyToMultiple',
           # Map each image in "images" to the "img" key.
           input_mapping={'img': 'images'},
           # After data transforms, revert map the "img" key to "images".
           inplace=True,
           # Whether to share random variables.
           # More details can be found in the below section.
           share_random_param=True,
           transforms=[
               # `RandomFlip` just reads the "img" key and modify the value.
               dict(type='RandomFlip'),
           ])
   ]
   ```

## Random Varibles Sharing

Sometimes, we want to share the random status of data transforms along several
calls. For example, in super-resolution task, we want to apply random flip on
both low-quanlity and ground-truth images synchronously.

In the `ApplyToMultiple` wrapper, we provide the `share_random_param` option.
To enable this function, the corresponding data transforms need to mark the
random variables that need to be shared.

Use the `RandomFlip` above as example.

```python
from mmengine.data.utils import cacheable_method

class RandomFlip(BaseTransform):
    @cacheable_method  # Mark the output as an reusable random variable
    def do_flip(self):
        flip = True if random.random() > 0.5 else False
        return flip

    def transform(self, results: dict) -> dict:
        img = results['img']
        if self.do_flip():
            results['img'] = mmcv.imflip(img)
        return results
```

With the `cacheable_method` decorator, the random variable `flip` can be
recorded and re-used among multiple calls in the `ApplyToMultiple`.

Dive into the `ApplyToMultiple` wrapper, we can find it use the
`cache_random_params` context manager to enable the random variables reuse in a
certain scope. We can use a small demo to experience the details of this
function.

```python
>>> import random
>>> from mmengine.data import BaseTransform
>>> from mmengine.data.utils import cacheable_method, cache_random_params
>>>
>>> class RandomNumber(BaseTransform):
...     @cacheable_method  # Mark the output as an reusable random variable
...     def get_cached_random(self):
...         return random.random()
... 
...     def transform(self, results: dict) -> dict:
...         results['cache'] = self.get_cached_random()
...         results['no_cache'] = random.random()
...         return results
>>>
>>> transform = RandomNumber()
>>> # Without `cache_random_params`, the marked random variables won't be recorded.
>>> for i in range(3):
...     data_dict = transform({})
...     print(f'{data_dict["cache"]:.4f}, {data_dict["no_cache"]:.4f}')
0.7994, 0.1712
0.5317, 0.5089
0.6758, 0.0542
>>> # With `cache_random_params`, the marked random variables will be reused.
>>> with cache_random_params(transform):
...     for i in range(3):
...         data_dict = transform({})
...         print(f'{data_dict["cache"]:.4f}, {data_dict["no_cache"]:.4f}')
0.9899, 0.5399
0.9899, 0.4246
0.9899, 0.9384
```
