# Weight initialization

Usually, we'll customize our module based on [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), which is implemented by Native PyTorch. Also, [torch.nn.init](https://pytorch.org/docs/stable/nn.init.html) could help us initialize the parameters of the model easily. To simplify the process of model construction and initialization, MMEngine designed the [BaseModule](mmengine.model.BaseModule) to help us define and initialize the model from config easily.

## Initialize the model from config

The core function of `BaseModule` is that it could help us to initialize the model from config. Subclasses inherited from `BaseModule` could define the `init_cfg` in the `__init__` function, and we can choose the method of initialization by configuring `init_cfg`.

Currently, we support the following initialization methods:

<table class="docutils">
<thead>
  <tr>
    <th>Initializer</th>
    <th>Registered name</th>
    <th>Function</th>
<tbody>
<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.ConstantInit.html#mmengine.model.ConstantInit">ConstantInit</a></td>
  <td>Constant</td>
  <td>Initialize the weight and bias with a constant, commonly used for Convolution</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.XavierInit.html#mmengine.model.XavierInit">XavierInit</a></td>
  <td>Xavier</td>
  <td>Initialize the weight by Xavier initialization, and initialize the bias with a constant</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.NormalInit.html#mmengine.model.NormalInit">NormalInit</a></td>
  <td>Normal</td>
  <td>Initialize the weight by normal distribution, and initialize the bias with a constant</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.TruncNormalInit.html#mmengine.model.TruncNormalInit">TruncNormalInit</a></td>
  <td>TruncNormal</td>
  <td>Initialize the weight by truncated normal distribution, and initialize the bias with a constant, commonly used for Transformer</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.UniformInit.html#mmengine.model.UniformInit">UniformInit</a></td>
  <td>Uniform</td>
  <td>Initialize the weight by uniform distribution, and initialize the bias with a constant, commonly used for convolution</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.KaimingInit.html#mmengine.model.KaimingInit">KaimingInit</a></td>
  <td>Kaiming</td>
  <td>Initialize the weight by Kaiming initialization, and initialize the bias with a constant. Commonly used for convolution</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.Caffe2XavierInit.html#mmengine.model.Caffe2XavierInit">Caffe2XavierInit</a></td>
  <td>Caffe2Xavier</td>
  <td>Xavier initialization in Caffe2, and Kaiming initialization in PyTorh with "fan_in" and "normal" mode. Commonly used for convolution</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.PretrainedInit.html#mmengine.model.PretrainedInit">PretrainedInit</a></td>
  <td>Pretrained</td>
  <td>Initialize the model with the pretrained model</td>
</tr>

</thead>
</table>

### Initialize the model with pretrained model

Defining the `ToyNet` as below:

```python
import torch
import torch.nn as nn

from mmengine.model import BaseModule


class ToyNet(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Linear(1, 1)


# Save the checkpoint.
toy_net = ToyNet()
torch.save(toy_net.state_dict(), './pretrained.pth')
pretrained = './pretrained.pth'

toy_net = ToyNet(init_cfg=dict(type='Pretrained', checkpoint=pretrained))
```

and then we can configure the `init_cfg` to make it load the pretrained model by calling `initi_weights()` after its construction.

```python
# Initialize the model with the saved checkpoint.
toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO - load model from: ./pretrained.pth
08/19 16:50:24 - mmengine - INFO - local loads checkpoint from path: ./pretrained.pth
```

If `init_cfg` is a `dict`, `type` means a kind of initializer registered in `WEIGHT_INITIALIZERS`. The `Pretrained` means `PretrainedInit`, which could help us to load the target checkpoint.
All initializers have the same mapping relationship like `Pretrained` -> `PretrainedInit`, which strips the suffix `Init` of the class name. The `checkpoint` argument of `PretrainedInit` means the path of the checkpoint. It could be a local path or a URL.

```{note}
`PretrainedInit` has a higher priority than any other initializer. The loaded pretrained weights will overwrite the previous initialized weights.
```

### Commonly used initialization methods

Similarly, we could use the `Kaiming` initialization just like `Pretrained` initializer. For example, we could make `init_cfg=dict(type='Kaiming', layer='Conv2d')` to initialize all `Conv2d` module with `Kaiming` initialization.

Sometimes we need to initialize the model with different initialization methods for different modules. For example, we could initialize the `Conv2d` module with `Kaiming` initialization and initialize the `Linear` module with `Xavier` initialization. We could make `init_cfg=dict(type='Kaiming', layer='Conv2d')`:

```python
import torch.nn as nn

from mmengine.model import BaseModule


class ToyNet(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.linear = nn.Linear(1, 1)
        self.conv = nn.Conv2d(1, 1, 1)


# Apply `Kaiming` initialization to `Conv2d` module and `Xavier` initialization to `Linear` module.
toy_net = ToyNet(
    init_cfg=[
        dict(type='Kaiming', layer='Conv2d'),
        dict(type='Xavier', layer='Linear')
    ], )
toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO -
linear.weight - torch.Size([1, 1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
linear.bias - torch.Size([1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
```

`layer` could also be a list, each element of which means a type of applied module.

```python
# Apply Kaiming initialization to `Conv2d` and `Linear` module.
toy_net = ToyNet(init_cfg=[dict(type='Kaiming', layer=['Conv2d', 'Linear'])], )
toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO -
linear.weight - torch.Size([1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
linear.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
```

### More fine-grained initialization

Sometimes we need to initialize the same type of module with different types of initialization. For example, we've defined `conv1` and `conv2` submodules, and we want to initialize the `conv1` with `Kaiming` initialization and `conv2` with `Xavier` initialization. We could configure the init_cfg with `override`:

```python
import torch.nn as nn

from mmengine.model import BaseModule


class ToyNet(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)


# Apllly `Kaiming` initialization to `conv1` and `Xavier` initialization to `conv2`.
toy_net = ToyNet(
    init_cfg=[
        dict(
            type='Kaiming',
            layer=['Conv2d'],
            override=dict(name='conv2', type='Xavier')),
    ], )
toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO -
conv1.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv1.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.weight - torch.Size([1, 1, 1, 1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0
```

`override` could be understood as an nested `init_cfg`, which could also be a `list` or `dict`, and we should also set "`type`" for it. The difference is that we must set `name` in `override` to specify the applied scope for submodule. As the example above, we set `name='conv2'` to specify that the `Xavier` initialization is applied to all submodules of `toy_net.conv2`.

### Customize the initialization method

Although the `init_cfg` could control the initialization method for different modules, we would have to register a new initialization method to `WEIGHT_INITIALIZERS` if we want to customize initialization process. It is not convenient right? Actually, we could also override the `init_weights` method to customize the initialization process.

Assuming we've defined the following modules:

- `ToyConv` inherit from `nn.Module`, implements `init_weights`which initialize `custom_weight`(`parameter` of `ToyConv`) with 1 and  initialize `custom_bias` with 0

- `ToyNet` defines a `ToyConv` submodule.

`ToyNet.init_weights` will call `init_weights` of all submodules sequentially.

```python
import torch
import torch.nn as nn

from mmengine.model import BaseModule


class ToyConv(nn.Module):

    def __init__(self):
        super().__init__()
        self.custom_weight = nn.Parameter(torch.empty(1, 1, 1, 1))
        self.custom_bias = nn.Parameter(torch.empty(1))

    def init_weights(self):
        with torch.no_grad():
            self.custom_weight = self.custom_weight.fill_(1)
            self.custom_bias = self.custom_bias.fill_(0)


class ToyNet(BaseModule):

    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.custom_conv = ToyConv()


toy_net = ToyNet(
    init_cfg=[
        dict(
            type='Kaiming',
            layer=['Conv2d'],
            override=dict(name='conv2', type='Xavier'))
    ])

toy_net.init_weights()
```

```
08/19 16:50:24 - mmengine - INFO -
conv1.weight - torch.Size([1, 1, 1, 1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv1.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.weight - torch.Size([1, 1, 1, 1]):
XavierInit: gain=1, distribution=normal, bias=0

08/19 16:50:24 - mmengine - INFO -
conv2.bias - torch.Size([1]):
KaimingInit: a=0, mode=fan_out, nonlinearity=relu, distribution =normal, bias=0

08/19 16:50:24 - mmengine - INFO -
custom_conv.custom_weight - torch.Size([1, 1, 1, 1]):
Initialized by user-defined `init_weights` in ToyConv

08/19 16:50:24 - mmengine - INFO -
custom_conv.custom_bias - torch.Size([1]):
Initialized by user-defined `init_weights` in ToyConv
```

### Conclusion

**1. Configure `init_cfg` to initialize model**

- Commonly used for the initialization of `Conv2d`, `Linear` and other underlying module. All initialization methods should be managed by `WEIGHT_INITIALIZERS`
- Dynamic initialization controlled by `init_cfg`

**2. Customize `init_weights`**

- Compared to configuring the `init_cfg`, implementing the `init_weights` is simpler and does not require registration. However, it is not as flexible as `init_cfg`, and it is not possible to initialize the module dynamically.

```{note}
- The priorify of init_weights is higher than `init_cfg`
- Runner will call `init_weights` in Runner.train()
```

### Ininitailize module with function

As mentioned in prior [section](#customize-the-initialization-method), we could customize our initialization in `init_weights`. To make it more convenient to initialize modules, MMEngine provides a series of **module initialization functions** to initialize the whole module based on `torch.nn.init`. For example, we want to initialize the weights of the convolutional layer with normal distribution and initialize the bias of the convolutional layer with a constant. The implementation of `torch.nn.init` is as follows:

```python
from torch.nn.init import normal_, constant_
import torch.nn as nn

model = nn.Conv2d(1, 1, 1)
normal_(model.weight, mean=0, std=0.01)
constant_(model.bias, val=0)
```

```
Parameter containing:
tensor([0.], requires_grad=True)
```

The above process is actually a standard process for initializing a convolutional module with normal distribution, so MMEngine simplifies this by implementing a series of common **module** initialization functions. Compared with `torch.nn.init`, the module initialization functions could accept the convolution module directly:

```python
from mmengine.model import normal_init

normal_init(model, mean=0, std=0.01, bias=0)
```

Similarly, we could also use [Kaiming](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) initialization and  [Xavier](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) initialization:

```python
from mmengine.model import kaiming_init, xavier_init

kaiming_init(model)
xavier_init(model)
```

Currently, MMEngine provide the following initialization function:

<table class="docutils">
<thead>
  <tr>
    <th>Initialization function</th>
    <th>Function</th>
<tbody>
<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.constant_init.html#mmengine.model.constant_init">constant_init</a></td>
  <td>Initialize the weight and bias with a constant, commonly used for Convolution</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.xavier_init.html#mmengine.model.xavier_init">xavier_init</a></td>
  <td>Initialize the weight by Xavier initialization, and initialize the bias with a constant</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.normal_init.html#mmengine.model.normal_init">normal_init</a></td>
  <td>Initialize the weight by normal distribution, and initialize the bias with a constant</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.trunc_normal_init.html#mmengine.model.trunc_normal_init">trunc_normal_init</a></td>
  <td>Initialize the weight by truncated normal distribution, and initialize the bias with a constant, commonly used for Transformer</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.uniform_init.html#mmengine.model.uniform_init">uniform_init</a></td>
  <td>Initialize the weight by uniform distribution, and initialize the bias with a constant, commonly used for convolution</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.kaiming_init.html#mmengine.model.kaiming_init">kaiming_init</a></td>
  <td>Initialize the weight by Kaiming initialization, and initialize the bias with a constant. Commonly used for convolution</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.caffe2_xavier_init.html#mmengine.model.caffe2_xavier_init">caffe2_xavier_init</a></td>
  <td>Xavier initialization in Caffe2, and Kaiming initialization in PyTorh with "fan_in" and "normal" mode. Commonly used for convolution</td>
</tr>

<tr>
  <td><a class="reference internal" href="../api/generated/mmengine.model.bias_init_with_prob.html#mmengine.model.bias_init_with_prob">bias_init_with_prob</a></td>
  <td>Initialize the bias with the probability</td>
</tr>

</thead>
</table>
