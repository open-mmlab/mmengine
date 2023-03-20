# Global manager (ManagerMixin)

During the training process, it is inevitable that we need to access some variables globally. Here are some examples:

- Accessing the [logger](mmengine.logging.MMLogger) in model to print some initialization information
- Accessing the [Visualizer](mmengine.config.Config) anywhere to visualize the predictions and feature maps.
- Accessing the scope in [Registry](mmengine.registry.Registry) to get the current scope.

In order to unify the mechanism to get the global variable built from different classes, MMEngine designs the [ManagerMixin](mmengine.utils.ManagerMixin).

## Interface introduction

- get_instance(name='', \*\*kwargs): Create or get the instance by name.
- get_current_instance(): Get the currently built instance.
- instance_name: Get the name of the instance.

## How to use

1. Define a class inherited from `ManagerMixin`

```python
from mmengine.utils import ManagerMixin


class GlobalClass(ManagerMixin):
    def __init__(self, name, value):
        super().__init__(name)
        self.value = value
```

```{note}
Subclasses of `ManagerMixin` must accept `name` argument in `__init__`. The `name` argument is used to identify the instance, and you can get the instance by `get_instance(name)`.
```

2. Instantiate the instance anywhere. let's take the hook as an example:

```python
from mmengine import Hook

class CustomHook(Hook):
    def before_run(self, runner):
        GlobalClass.get_instance('mmengine', value=50)
        GlobalClass.get_instance(runner.experiment_name, value=100)
```

`GlobalClass.get_instance({name})` will first check whether the instance with the name `{name}` has been built. If not, it will build a new instance with the name `{name}`, otherwise it will return the existing instance. As the above example shows, when we call `GlobalClass.get_instance('mmengine')` at the first time, it will build a new instance with the name `mmengine`. Then we call `GlobalClass.get_instance(runner.experiment_name)`, it will also build a new instance with a different name.

Here we build two instances for the convenience of the subsequent introduction of `get_current_instance`.

3. Accessing the instance anywhere

```python
import torch.nn as nn


class CustomModule(nn.Module):
    def forward(self, x):
        value = GlobalClass.get_current_instance().value
        # Since the name of the latest built instance is
        # `runner.experiment_name`, value will be 100.

        value = GlobalClass.get_instance('mmengine').value
        # The value of instance with the name mmengine is 50.

        value = GlobalClass.get_instance('mmengine', 1000).value
        # `mmengine` instance has been built, an error will be raised
        # if `get_instance` accepts other parameters.
```

We can get the instance with the specified name by `get_instance(name)`, or get the currently built instance by `get_current_instance` anywhere.

```{warning}
If the instance with the specified name has already been built, `get_instance` will raise an error if it accepts its construct parameters.
```
