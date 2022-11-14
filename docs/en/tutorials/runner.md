# Runner

Welcome to the tutorial of runner, the core of MMEngine's user interface!

The runner, as an "integrator" in MMEngine, covers all aspects of the framework and shoulders the responsibility of organizing and scheduling nearly all modules. Therefore, the code and implementation logic in it have to take into account various situations, making it relatively hard to understand. But **don't worry**! In this tutorial, we will leave out some messy details and have a quick overview of commonly used APIs, functionalities and examples. Hopefully this should provide you with a clear and easy-to-understand user interface. After reading through this tutorial, you will be able to:

- Master the common usage and configuration of the runner
- Learn the best practice -- writing config files -- of the runner
- Know about the basic dataflow and execution order
- Feel by yourself the advantages of using runner, perhaps

## Example codes of the runner

To build your training pipeline with a runner, there are typically two ways to get started:

- Refer to runner's [API documentations](mmengine.runner.Runner) for argument-by-argument configuration
- Make your custom modifications based on some existing configurations, such as [Getting started in 15 minutes](../get_started/15_minutes.md) and downstream repos like [MMDet](https://github.com/open-mmlab/mmdetection)

Pros and cons lie in both approaches. For the former one, beginners may be lost in a vast number of configurable arguments. For the latter one, beginers may find it hard to get a good reference, since neither an over-simplified nor an over-detailed reference is conductive to them.

We argue that the key to learn runner is using it as a memo. You should master its most commonly used arguments and only focus on those less used when in need, since default values usually work fine. In the following we will provide a beginer-friendly example to illustrate most commonly used arguments of the runner, along with advanced guidelines for those less used.

### A beginer-friendly example

```{hint}
In this tutorial, we hope you can focus more on overall architecture instead of implementation details. This "top-down" way of thinking is exactly what we advocate. Don't worry, you will definitely have plenty of opportunities and guidance afterwards to focus on modules you want to improve.
```

<details>
<summary>Before running the actual example below, you should first run this piece of code for preparation of model, dataset and metric. However, these implementations are not important in this tutorial and you can simply look through</summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from mmengine.model import BaseModel
from mmengine.evaluator import BaseMetric
from mmengine.registry import MODELS, DATASETS, METRICS


@MODELS.register_module()
class MyAwesomeModel(BaseModel):
    def __init__(self, layers=4, activation='relu') -> None:
        super().__init__()
        if activation == 'relu':
            act_type = nn.ReLU
        elif activation == 'silu':
            act_type = nn.SiLU
        elif activation == 'none':
            act_type = nn.Identity
        else:
            raise NotImplementedError
        sequence = [nn.Linear(2, 64), act_type()]
        for _ in range(layers-1):
            sequence.extend([nn.Linear(64, 64), act_type()])
        self.mlp = nn.Sequential(*sequence)
        self.classifier = nn.Linear(64, 2)

    def forward(self, data, labels, mode):
        x = self.mlp(data)
        x = self.classifier(x)
        if mode == 'tensor':
            return x
        elif mode == 'predict':
            return F.softmax(x, dim=1), labels
        elif mode == 'loss':
            return {'loss': F.cross_entropy(x, labels)}


@DATASETS.register_module()
class MyDataset(Dataset):
    def __init__(self, is_train, size):
        self.is_train = is_train
        if self.is_train:
            torch.manual_seed(0)
            self.labels = torch.randint(0, 2, (size,))
        else:
            torch.manual_seed(3407)
            self.labels = torch.randint(0, 2, (size,))
        r = 3 * (self.labels+1) + torch.randn(self.labels.shape)
        theta = torch.rand(self.labels.shape) * 2 * torch.pi
        self.data = torch.vstack([r*torch.cos(theta), r*torch.sin(theta)]).T

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)


@METRICS.register_module()
class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__()

    def process(self, data_batch, data_samples):
        score, gt = data_samples
        self.results.append({
            'batch_size': len(gt),
            'correct': (score.argmax(dim=1) == gt).sum().cpu(),
        })

    def compute_metrics(self, results):
        total_correct = sum(r['correct'] for r in results)
        total_size = sum(r['batch_size'] for r in results)
        return dict(accuracy=100*total_correct/total_size)
```

</details>

<details>
<summary>Click to show a long example. Be well prepared</summary>

```python
from torch.utils.data import DataLoader, default_collate
from torch.optim import Adam
from mmengine.runner import Runner


runner = Runner(
    # your model
    model=MyAwesomeModel(
        layers=2,
        activation='relu'),
    # work directory for saving checkpoints and logs
    work_dir='exp/my_awesome_model',

    # training data
    train_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=True,
            size=10000),
        shuffle=True,
        collate_fn=default_collate,
        batch_size=64,
        pin_memory=True,
        num_workers=2),
    # training configurations
    train_cfg=dict(
        by_epoch=True,   # display in epoch number instead of iterations
        max_epochs=10,
        val_begin=2,     # start validation from the 2nd epoch
        val_interval=1), # do validation every 1 epoch

    # OptimizerWrapper, new concept in MMEngine for richer optimization options
    # Default value works fine for most cases. You may check our documentations
    # for more details, e.g. 'AmpOptimWrapper' for enabling mixed precision
    # training.
    optim_wrapper=dict(
        optimizer=dict(
            type=Adam,
            lr=0.001)),
    # ParamScheduler to adjust learning rates or momentums during training
    param_scheduler=dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1),

    # validation data
    val_dataloader=DataLoader(
        dataset=MyDataset(
            is_train=False,
            size=1000),
        shuffle=False,
        collate_fn=default_collate,
        batch_size=1000,
        pin_memory=True,
        num_workers=2),
    # validation configurations, usually leave it an empty dict
    val_cfg=dict(),
    # evaluation metrics and evaluator
    val_evaluator=dict(type=Accuracy),

    # following are advanced configurations, try to default when not in need
    # hooks are advanced usage, try to default when not in need
    default_hooks=dict(
        # the most commonly used hook for modifying checkpoint saving interval
        checkpoint=dict(type='CheckpointHook', interval=1)),

    # `luancher` and `env_cfg` responsible for distributed environment
    launcher='none',
    env_cfg=dict(
        cudnn_benchmark=False,   # whether enable cudnn_benchmark
        backend='nccl',   # distributed communication backend
        mp_cfg=dict(mp_start_method='fork')),  # multiprocessing configs
    log_level='INFO',

    # load model weights from given path. None for no loading.
    load_from=None
    # resume training from the given path
    resume=False
)

# start training your model
runner.train()
```

</details>

### Explanations on example codes

Really a long piece of code, isn't it! However, if you read through the above example, you may have already understood the training process in general even without knowing any implementation details, thanks to the compactness and readability of runner codes(probably). This is what MMEngine expects: a structured, modular and standardized training process that allows for more reliable reproductions and clearer comparisons.

The above example may lead you to the following confusions:

<details>
<summary>There are too many arguments!</summary>

Don't worry. As we mentioned before, **use runner as a memo**. The runner covers all aspects just to ensure you won't miss something important. You don't actually need to configure everything. The simple example in [15 minutes](../get_started/15_minutes.md) still works fine, and it can be even more simplified by removing `val_evaluator`, `val_dataloader` and `val_cfg` without breaking down. All configurable arguments are driven by your demands. Those not in your focus usually works fine by default.

</details>

<details>
<summary>Why are some arguments passed as dicts?</summary>

Well, this is related to MMEngine's style. In MMEngine, we provide 2 different styles of runner construction: a) manual construction and b) construction via registry and configs. If this confuses you, the following example will give a good illustration:

```python
from mmengine.model import BaseModel
from mmengine.runner import Runner
from mmengine.registry import MODELS # 模型根注册器，你的自定义模型需要注册到这个根注册器中

@MODELS.register_module() # decorator for registration
class MyAwesomeModel(BaseModel): # your custom model
    def __init__(self, layers=18, activation='silu'):
        ...

# An example of manual construction
runner = Runner(
    model=dict(
        type='MyAwesomeModel',
        layers=50,
        activation='relu'),
    ...
)

# An example of construction via registry and configs
model = MyAwesomeModel(layers=18, activation='relu')
runner = Runner(
    model=model,
    ...
)
```

Similar to the above example, most arguments in the runner accepts both 2 types of inputs. These 2 styles are conceptually equivalent. The difference is, in the former style, the module(passed in as a `dict`) will be built **in the runner when actually needed**, while in the latter style, the module has been built before passed to the runner.
