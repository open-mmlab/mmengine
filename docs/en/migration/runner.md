# Migrate Runner from MMCV to MMEngine

## Introduction

As MMCV supports more and more deep learning tasks, and users' needs become much more complicated, we have higher requirements for the flexibility and versatility of the existing `Runner` of MMCV. Therefore, MMEngine implements a more general and flexible `Runner` based on MMCV to support more complicated training processes.

The `Runner` in MMEngine expands the scope and takes on more functions. we abstracted [training loop controller (EpochBasedTrainLoop/IterBasedTrainLoop)](mmengine.runner.EpochBasedTrainLoop), [validation loop controller (ValLoop)](mmengine.runner.ValLoop) and [TestLoop](mmengine.runner.TestLoop) to make it more convenient for users to customize their training process.

Firstly, we will introduce how to migrate the entry point of training from MMCV to MMEngine, to simplify and unify the training script. Then, we'll introduce the difference in the instantiation of `Runner` between MMCV and MMEngine in detail.

## Migrate the entry point

Take MMDet as an example, the differences between training scripts in MMCV and MMEngine are as follows:

### Migrate the configuration file

<table class="docutils">
<thead>
  <tr>
    <th>Configuration file based on MMCV Runner </th>
    <th>Configuration file based on MMEngine Runner</th>
<tbody>
<tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# default_runtime.py
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]


opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# default_runtime.py
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
```

</div>
  </td>
  </tr>
<tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# scheduler.py
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# scheduler.py
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
```

</div>
  </td>
</tr>
<tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# coco_detection.py

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# coco_detection.py

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False)
test_evaluator = val_evaluator
```

</div>
  </td>

</tr>
</thead>
</table>

`Runner` in MMEngine provides more customizable components, including training/validation/testing process and DataLoader. Therefore, the configuration file is a bit longer compared to MMCV.

`MMEngine` follows the WYSIWYG principle and reorganizes the hierarchy of each component in configuration so that most of the first-level fields of configuration correspond to the core components in the `Runner`, such as DataLoader, [Evaluator](../tutorials/evaluation.md), [Hook](../tutorials/hook.md), etc. The new format configuration file could help users to read and understand the core components in `Runner`, and ignore the relatively unimportant parts.

### Migrate the training script

Compared with the `Runner` in MMCV, `Runner` in MMEngine takes on more functions, such as building DataLoader and distributed model. Therefore, we do not need to build the components like DataLoader and distributed model manually anymore. We can configure them during the instantiation of `Runner`, and then build them in the training/validation/testing process. Take the training script of MMDet as an example:

<table class="docutils">
<thead>
  <tr>
    <th>Training script based on MMCV Runner</th>
    <th>Training script based on MMEngine Runner</th>
<tbody>
<tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# tools/train.py
args = parse_args()

cfg = Config.fromfile(args.config)

# replace the ${key} with the value of cfg.key
cfg = replace_cfg_vals(cfg)

# update data root according to MMDET_DATASETS
update_data_root(cfg)

if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)

if args.auto_scale_lr:
    if 'auto_scale_lr' in cfg and \
            'enable' in cfg.auto_scale_lr and \
            'base_batch_size' in cfg.auto_scale_lr:
        cfg.auto_scale_lr.enable = True
    else:
        warnings.warn('Can not find "auto_scale_lr" or '
                        '"auto_scale_lr.enable" or '
                        '"auto_scale_lr.base_batch_size" in your'
                        ' configuration file. Please update all the '
                        'configuration files to mmdet >= 2.24.1.')

# set multi-process settings
setup_multi_processes(cfg)

# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

# work_dir is determined in this priority: CLI > segment in file > filename
if args.work_dir is not None:
    # update configs according to CLI args if args.work_dir is not None
    cfg.work_dir = args.work_dir
elif cfg.get('work_dir', None) is None:
    # use config filename as default work_dir if cfg.work_dir is None
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])

if args.resume_from is not None:
    cfg.resume_from = args.resume_from
cfg.auto_resume = args.auto_resume
if args.gpus is not None:
    cfg.gpu_ids = range(1)
    warnings.warn('`--gpus` is deprecated because we only support '
                    'single GPU mode in non-distributed training. '
                    'Use `gpus=1` now.')
if args.gpu_ids is not None:
    cfg.gpu_ids = args.gpu_ids[0:1]
    warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                    'Because we only support single GPU mode in '
                    'non-distributed training. Use the first GPU '
                    'in `gpu_ids` now.')
if args.gpus is None and args.gpu_ids is None:
    cfg.gpu_ids = [args.gpu_id]

# init distributed env first, since logger depends on the dist info.
if args.launcher == 'none':
    distributed = False
else:
    distributed = True
    init_dist(args.launcher, **cfg.dist_params)
    # re-set gpu_ids with distributed training mode
    _, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

# create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# dump config
cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
# init the logger before other steps
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

# init the meta dict to record some important information such as
# environment info and seed, which will be logged
meta = dict()
# log env info
env_info_dict = collect_env()
env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' +
            dash_line)
meta['env_info'] = env_info
meta['config'] = cfg.pretty_text
# log some basic info
logger.info(f'Distributed training: {distributed}')
logger.info(f'Config:\n{cfg.pretty_text}')

cfg.device = get_device()
# set random seeds
seed = init_random_seed(args.seed, device=cfg.device)
seed = seed + dist.get_rank() if args.diff_seed else seed
logger.info(f'Set random seed to {seed}, '
            f'deterministic: {args.deterministic}')
set_random_seed(seed, deterministic=args.deterministic)
cfg.seed = seed
meta['seed'] = seed
meta['exp_name'] = osp.basename(args.config)

model = build_detector(
    cfg.model,
    train_cfg=cfg.get('train_cfg'),
    test_cfg=cfg.get('test_cfg'))
model.init_weights()

datasets = []
train_detector(
    model,
    datasets,
    cfg,
    distributed=distributed,
    validate=(not args.no_validate),
    timestamp=timestamp,
    meta=meta)
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# tools/train.py
args = parse_args()

# register all modules in mmdet into the registries
# do not init the default scope here because it will be init in the runner
register_all_modules(init_default_scope=False)

# load config
cfg = Config.fromfile(args.config)
cfg.launcher = args.launcher
if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)

# work_dir is determined in this priority: CLI > segment in file > filename
if args.work_dir is not None:
    # update configs according to CLI args if args.work_dir is not None
    cfg.work_dir = args.work_dir
elif cfg.get('work_dir', None) is None:
    # use config filename as default work_dir if cfg.work_dir is None
    cfg.work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])

# enable automatic-mixed-precision training
if args.amp is True:
    optim_wrapper = cfg.optim_wrapper.type
    if optim_wrapper == 'AmpOptimWrapper':
        print_log(
            'AMP training is already enabled in your config.',
            logger='current',
            level=logging.WARNING)
    else:
        assert optim_wrapper == 'OptimWrapper', (
            '`--amp` is only supported when the optimizer wrapper type is '
            f'`OptimWrapper` but got {optim_wrapper}.')
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

# enable automatically scaling LR
if args.auto_scale_lr:
    if 'auto_scale_lr' in cfg and \
            'enable' in cfg.auto_scale_lr and \
            'base_batch_size' in cfg.auto_scale_lr:
        cfg.auto_scale_lr.enable = True
    else:
        raise RuntimeError('Can not find "auto_scale_lr" or '
                            '"auto_scale_lr.enable" or '
                            '"auto_scale_lr.base_batch_size" in your'
                            ' configuration file.')

cfg.resume = args.resume

# build the runner from config
if 'runner_type' not in cfg:
    # build the default runner
    runner = Runner.from_cfg(cfg)
else:
    # build customized runner from the registry
    # if 'runner_type' is set in the cfg
    runner = RUNNERS.build(cfg)

# start training
runner.train()
```

</div>
</td>
</tr>
<tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# apis/train.py
def init_random_seed(...):
    ...

def set_random_seed(...):
    ...

# define function tools.
...


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    auto_scale_lr(cfg, distributed, logger)
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
```

</div>
  </td>
  <td valign="top">

```python
# `apis/train.py` is removed in `mmengine`
```

</td>
</tr>
</thead>
</table>

Table above shows the differences between training script of MMEngine `Runner` and MMCV `Runner`. Repositories of OpenMMLab 1.x organize their own process to build `Runner`, which contributes to the large amount of redundant code. MMEngine unifies and formats the building process, such as setting random seed, initializing distributed environment, building DataLoader, building `Optimizer`, etc. This help the downstream repositories simplify the process to prepare the runner, and only need to configure the parameters of `Runner`.

For the downstream repositories, training script based on MMEngine Runner not only simplify the `tools/train.py`, but also can directly omit the `apis/train.py`. Similarly, we can also set random seed, initialize distributed environment by configuring the parameters of `Runner`, and do not need to implement the corresponding code.

## Migrate Runner

This section describes the differences in the training, validation, and testing processes between the MMCV Runner and the MMEngine Runner, as follows.

01. [Prepare logger](#prepare-logger)
02. [Set random seed](#set-random-seed)
03. [Initialize environment variables](#initialize-environment-variables)
04. [Prepare data](#prepare-data)
05. [Prepare model](#prepare-model)
06. [Prepare optimizer](#prepare-optimizer)
07. [Prepare hooks](#prepare-hooks)
08. [Prepare testing/validation components](#prepare-testingvalidation-components)
09. [Build runner](#build-runner)
10. [Load checkpoint](#load-checkpoint)
11. [Training process](#training-process), [Testing process](#testing-process)
12. [Custom training process](#customize-training-process)

The following tutorial will describe the difference above in detail.

### Prepare logger

**Prepare logger in MMCV**

MMCV needs to call the `get_logger` to get a formatted logger and use it to output and log the training information.

```python
logger = get_logger(name='custom', log_file=log_file, log_level=cfg.log_level)
env_info_dict = collect_env()
env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' +
            dash_line)
```

The instantiation of the Runner also relies on the logger:

```python
runner = Runner(
    ...
    logger=logger
    ...)
```

**Prepare logger in MMEngine**

Configure the `log_level` for `Runner`, and it will build the logger automatically.

```python
log_level = 'INFO'
```

### Set random seed

**Set random seed in MMCV**

Set random seed manually in training script:

```python
...
seed = init_random_seed(args.seed, device=cfg.device)
seed = seed + dist.get_rank() if args.diff_seed else seed
logger.info(f'Set random seed to {seed}, '
            f'deterministic: {args.deterministic}')
set_random_seed(seed, deterministic=args.deterministic)
...
```

**Set random seed in MMEngine**

Configure the `randomness` for `Runner`, see more information in [Runner.set_randomness](mmengine.runner.Runner.set_randomness)

**Configuration changes**

<table class="docutils">
<thead>
  <tr>
    <th>Configuration of MMCV</th>
    <th>Configuration of MMEngine</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
seed = 1
deterministic=False
diff_seed=False
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
randomness=dict(seed=1,
                deterministic=True,
                diff_rank_seed=False)
```

</div>
  </td>
</tr>
</thead>
</table>

### Initialize environment variables

**Initialize the environment variables**

MMCV needs to setup launcher of distributed training, set environment variables for multi-process communication, initialize the distributed environment and wrap model with the distributed wrapper like this:

```python
...
setup_multi_processes(cfg)
init_dist(cfg.launcher, **cfg.dist_params)
model = MMDistributedDataParallel(
    model,
    device_ids=[int(os.environ['LOCAL_RANK'])],
    broadcast_buffers=False,
    find_unused_parameters=find_unused_parameters)
```

As for MMEngine, you can setup launcher by configuring `launcher` of `Runner`, and configure other items mentioned above in `env_cfg`. See more information in the table below:

**Configuration changes**

<table class="docutils">
<thead>
  <tr>
    <th>MMCV configuration</th>
    <th>MMEngine configuration</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
launcher = 'pytorch'  # enable distributed training
dist_params = dict(backend='nccl')  # choose communication backend
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
launcher = 'pytorch'
env_cfg = dict(dist_cfg=dict(backend='nccl'))
```

</div>
  </td>
</tr>
</thead>
</table>

In this tutorial, we set `env_cfg` to:

```python
env_cfg = dict(dist_cfg=dict(backend='nccl'))
```

### Prepare data

Both MMEngine and MMCV `Runner` can accept built `DataLoader`

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CIFAR10(
    root='data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2)

val_dataset = CIFAR10(
    root='data', train=False, download=True, transform=transform)
val_dataloader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=2)
```

**Configuration changes**

<table class="docutils">
<thead>
  <tr>
    <th>Configuration of MMCV</th>
    <th>Configuration of MMEngine</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
data = dict(
    samples_per_gpu=2,  # batch_size of single gpu
    workers_per_gpu=2,  # num_workers of DataLoader
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    # Configurable sampler
    sampler=dict(type='DefaultSampler', shuffle=True),
    # Configurable batch_sampler
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1, # batch_size of validation process
    num_workers=2,
    persistent_workers=True,
    drop_last=False, # whether drop the last batch
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader
```

</div>
  </td>
</tr>
</thead>
</table>

### Prepare model

See [Migrate model from mmcv](./model.md) for more information

```python
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel


class Model(BaseModel):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, img, label, mode):
        feat = self.pool(F.relu(self.conv1(img)))
        feat = self.pool(F.relu(self.conv2(feat)))
        feat = feat.view(-1, 16 * 5 * 5)
        feat = F.relu(self.fc1(feat))
        feat = F.relu(self.fc2(feat))
        feat = self.fc3(feat)
        if mode == 'loss':
            loss = self.loss_fn(feat, label)
            return dict(loss=loss)
        else:
            return [feat.argmax(1)]

model = Model()
```

### Prepare optimizer

**Prepare optimizer in MMCV**

MMCV Runner can accept built optimizer

```python
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
```

For complicated configurations of optimizers, MMCV needs to build optimizers based on the optimizer constructors.

```python

optimizer_cfg = dict(
    optimizer=dict(type='SGD', lr=0.01, weight_decay=0.0001),
    paramwise_cfg=dict(norm_decay_mult=0))

def build_optimizer_constructor(cfg):
    constructor_type = cfg.get('type')
    if constructor_type in OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, OPTIMIZER_BUILDERS)
    elif constructor_type in MMCV_OPTIMIZER_BUILDERS:
        return build_from_cfg(cfg, MMCV_OPTIMIZER_BUILDERS)
    else:
        raise KeyError(f'{constructor_type} is not registered '
                       'in the optimizer builder registry.')


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer

optimizer = build_optimizer(model, optimizer_cfg)
```

**Prepare optimizer in MMEngine**

MMEngine needs to configure [optim_wrapper](mmengine.optim.OptimWrapper) for `Runner`. For more complicated cases, you can also configure the `optim_wrapper` more specifically. See more information in the API [documents](mmengine.runner.Runner.build_optim_wrapper)

**Configuration changes**

<table class="docutils">
<thead>
  <tr>
    <th>Configuration in MMCV</th>
    <th>Configuration in MMEngine</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
optimizer = dict(
    constructor='CustomConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={  # parameters of constructor
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6
    })

# MMCV needs to configure `optim_config` additionally
optimizer_config = dict(grad_clip=None)
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
optim_wrapper = dict(
    constructor='CustomConstructor',
    type='OptimWrapper',  # Specify the type of OptimWrapper
    optimizer=dict(  # optimizer configuration
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05)
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6
    })
```

</div>
  </td>
</tr>
</thead>
</table>

```{note}
For the high-level tasks like detection and classification, MMCV needs to configure `optim_config` to build `OptimizerHook`, while not necessary for MMEngine.
```

`optim_wrapper` used in this tutorial is as follows:

```python
from torch.optim import SGD

optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
optim_wrapper = dict(optimizer=optimizer)
```

### Prepare hooks

**Prepare hooks in MMCV**

The commonly used hooks configuration in MMCV is as follows:

```python
# learning rate scheduler config
lr_config = dict(policy='step', step=[2, 3])
# configuration of optimizer
optimizer_config = dict(grad_clip=None)
# configuration of saving checkpoints periodically
checkpoint_config = dict(interval=1)
# save log periodically and multiple hooks can be used simultaneously
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
# register hooks to runner and those hooks will be invoked automatically
runner.register_training_hooks(
    lr_config=lr_config,
    optimizer_config=optimizer_config,
    checkpoint_config=checkpoint_config,
    log_config=log_config)
```

Among them:

- `lr_config` is used for `LrUpdaterHook`
- `optimizer_config` is used for `OptimizerHook`
- `checkpoint_config` is used for `CheckPointHook`
- `log_config` is used for `LoggerHook`

Besides the hooks mentioned above, MMCV Runner will build `IterTimerHook` automatically. MMCV `Runner` will register the training hooks after instantiating the model, while MMEngine Runner will initialize the hooks during instantiating the model.

**Prepare hooks in MMEngine**

MMEngine `Runner` takes some commonly used hooks in MMCV as the default hooks.

- [RuntimeInfoHook](mmengine.hooks.RuntimeInfoHook)
- [IterTimerHook](mmengine.hooks.IterTimerHook)
- [DistSamplerSeedHook](mmengine.hooks.DistSamplerSeedHook)
- [LoggerHook](mmengine.hooks.LoggerHook)
- [CheckpointHook](mmengine.hooks.CheckpointHook)
- [ParamSchedulerHook](mmengine.hooks.ParamSchedulerHook)

Compared with the example of MMCV

- `LrUpdaterHook` correspond to the `ParamSchedulerHook`, find more details in [migrate scheduler](./param_scheduler.md)
- MMEngine optimize the model in [train_step](mmengine.model.BaseModel.train_step), therefore we do not need `OptimizerHook` in MMEngine anymore
- MMEngine takes `CheckPointHook` as the default hook
- MMEngine take `LoggerHook` as the default hook

Therefore, we can achieve the same effect as the MMCV example as long as we configure the [param_scheduler](../tutorials/param_scheduler.md) correctly.

We can also register custom hooks in MMEngine runner, find more details in [runner tutorial](../tutorials/runner.md) and [migrate hook](./hook.md).

<table class="docutils">
<thead>
  <tr>
    <th>Commonly used hooks in MMCV</th>
    <th>Default hooks in MMEngine</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# Configure training hooks
# Configure LrUpdaterHook
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# Configure OptimizerHook
optimizer_config = dict(grad_clip=None)

# Configure LoggerHook
log_config = dict(  # LoggerHook
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# Configure CheckPointHook
checkpoint_config = dict(interval=1)  # CheckPointHook
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# Configure parameter scheduler
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# Configure default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
```

</div>
  </td>
</tr>
</thead>
</table>

The parameter scheduler used in this tutorial is as follows:

```python
from math import gamma

param_scheduler = dict(type='MultiStepLR', milestones=[2, 3], gamma=0.1)
```

### Prepare testing/validation components

MMCV implements the validation process by `EvalHook`, and we'll not talk too much about it here. Given that validation is a common process in training, MMEngine abstracts validation as two independent modules: [Evaluator](../tutorials/evaluation.md) and [ValLoop](../tutorials/runner.md). We can customize the metric or the validation process by defining a new [loop](mmengine.runner.ValLoop) or a new [metric](mmengine.evaluator.BaseMetric).

```python
import torch
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

@METRICS.register_module(force=True)
class ToyAccuracyMetric(BaseMetric):

    def process(self, label, pred) -> None:
        self.results.append((label[1], pred, len(label[1])))

    def compute_metrics(self, results: list) -> dict:
        num_sample = 0
        acc = 0
        for label, pred, batch_size in results:
            acc += (label == torch.stack(pred)).sum()
            num_sample += batch_size
        return dict(Accuracy=acc / num_sample)
```

After defining the metric, we should also configure the evaluator and loop for `Runner`. The example used in this tutorial is as follows:

```python
val_evaluator = dict(type='ToyAccuracyMetric')
val_cfg = dict(type='ValLoop')
```

<table class="docutils">
<thead>
  <tr>
    <th>Configure validation in MMCV</th>
    <th>Configure validation in MMEngine</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
eval_cfg = cfg.get('evaluation', {})
eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
eval_hook = DistEvalHook if distributed else EvalHook
runner.register_hook(
    eval_hook(val_dataloader, **eval_cfg), priority='LOW')
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
val_dataloader = val_dataloader
val_evaluator = dict(type='ToyAccuracyMetric')
val_cfg = dict(type='ValLoop')
```

</div>
  </td>
</tr>
</thead>
</table>

### Build Runner

**Building Runner in MMCV**

```python
runner = EpochBasedRunner(
    model=model,
    optimizer=optimizer,
    work_dir=work_dir,
    logger=logger,
    max_epochs=4
)
```

**Building Runner in MMEngine**

The `EpochBasedRunner` and `max_epochs` arguments in `MMCV` are moved to `train_cfg` in MMEngine. All parameters configurable in `train_cfg` are listed below:

- by_epoch: `True` equivalent to `EpochBasedRunner`. `False` equivalent to `IterBasedRunner`
- `max_epoch/max_iter`: Equivalent to `max_epochs` and `max_iters` in MMCV
- `val_iterval`: Equivalent to `interval` in MMCV

```python
from mmengine.runner import Runner

runner = Runner(
    model=model,  # model to be optimized
    work_dir='./work_dir',  # working directory
    randomness=randomness,  # random seed
    env_cfg=env_cfg,  # environment config
    launcher='none',  # launcher for distributed training
    optim_wrapper=optim_wrapper,  # configure optimizer wrapper
    param_scheduler=param_scheduler,  # configure parameter scheduler
    train_dataloader=train_dataloader,  # configure train dataloader
    train_cfg=dict(by_epoch=True, max_epochs=4, val_interval=1),  # Configure training loop
    val_dataloader=val_dataloader,  # Configure validation dataloader
    val_evaluator=val_evaluator,  # Configure evaluator and metrics
    val_cfg=val_cfg)  # Configure validation loop
```

### Load checkpoint

**Loading checkpoint in MMCV**

```python
if cfg.resume_from:
    runner.resume(cfg.resume_from)
elif cfg.load_from:
    runner.load_checkpoint(cfg.load_from)
```

**Loading checkpoint in MMEngine**

```python
runner = Runner(
    ...
    load_from='/path/to/checkpoint',
    resume=True
)
```

<table class="docutils">
<thead>
  <tr>
    <th>Configuration of loading checkpoint in MMCV</th>
    <th>Configuration of loading checkpoint in MMEngine</th>
<tbody>
<tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
load_from = 'path/to/ckpt'
```

</td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
load_from = 'path/to/ckpt'
resume = False
```

</div>
  </td>
</tr>
<tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
resume_from = 'path/to/ckpt'
```

</td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
load_from = 'path/to/ckpt'
resume = True
```

</div>
  </td>
  </tr>
</thead>
</table>

### Training process

**Training process in MMCV**

Resume or load checkpoint firstly, and then start training.

```python
if cfg.resume_from:
    runner.resume(cfg.resume_from)
elif cfg.load_from:
    runner.load_checkpoint(cfg.load_from)
runner.run(data_loaders, cfg.workflow)
```

**Training process in MMEngine**

Complete the process mentioned above the `Runner.__init__` and `Runner.train`

```python
runner.train()
```

### Testing process

Since MMCV Runner does not integrate the test function, we need to implement the test scripts by ourselves.

For MMEngine Runner, as long as we have configured the `test_dataloader`, `test_cfg` and `test_evaluator` for the `Runner`, we can call `Runner.test` to start the testing process.

**`work_dir` is the same for training**

```python
runner = Runner(
    model=model,
    work_dir='./work_dir',
    randomness=randomness,
    env_cfg=env_cfg,
    launcher='none',
    optim_wrapper=optim_wrapper,
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_evaluator=val_evaluator,
    val_cfg=val_cfg,
    test_dataloader=val_dataloader,
    test_evaluator=val_evaluator,
    test_cfg=dict(type='TestLoop'),
)
runner.test()
```

**`work_dir` is the different for training, configure load_from manually**

```python
runner = Runner(
    model=model,
    work_dir='./test_work_dir',
    load_from='./work_dir/epoch_5.pth',  # set load_from additionally
    randomness=randomness,
    env_cfg=env_cfg,
    launcher='none',
    optim_wrapper=optim_wrapper,
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_evaluator=val_evaluator,
    val_cfg=val_cfg,
    test_dataloader=val_dataloader,
    test_evaluator=val_evaluator,
    test_cfg=dict(type='TestLoop'),
)
runner.test()
```

### Customize training process

If we want to customize a training/validation process, we need to override the `Runner.val` or `Runner.train` in a custom `Runner`. Take overriding `runner.train` as an example, suppose we need to train with the same batch twice for each iteration, we can override the `Runner.train` like this:

```python
class CustomRunner(EpochBasedRunner):
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self.data_batch = data_batch
            self._inner_iter = i
            for _ in range(2)
                self.call_hook('before_train_iter')
                self.run_iter(data_batch, train_mode=True, **kwargs)
                self.call_hook('after_train_iter')
            del self.data_batch
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
```

In MMEngine, we need to customize a train loop.

```python
from mmengine.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop


@LOOPS.register_module()
class CustomEpochBasedTrainLoop(EpochBasedTrainLoop):
    def run_iter(self, idx, data_batch) -> None:
        for _ in range(2):
            super().run_iter(idx, data_batch)
```

and then, we need to set `type` as `CustomEpochBasedTrainLoop` in `train_cfg`. Note that `by_epoch` and `type` cannot be configured at the same time. Once `by_epoch` is configured, the type of the training loop will be inferred as `EpochBasedTrainLoop`.

```python
runner = Runner(
    model=model,
    work_dir='./test_work_dir',
    randomness=randomness,
    env_cfg=env_cfg,
    launcher='none',
    optim_wrapper=dict(optimizer=dict(type='SGD', lr=0.001, momentum=0.9)),
    train_dataloader=train_dataloader,
    train_cfg=dict(
        type='CustomEpochBasedTrainLoop',
        max_epochs=5,
        val_interval=1),
    val_dataloader=val_dataloader,
    val_evaluator=val_evaluator,
    val_cfg=val_cfg,
    test_dataloader=val_dataloader,
    test_evaluator=val_evaluator,
    test_cfg=dict(type='TestLoop'),
)
runner.train()
```

For more complicated migration needs of `Runner`, you can refer to the [runner tutorials](../tutorials/runner.md) and [runner design](../design/runner.md).
