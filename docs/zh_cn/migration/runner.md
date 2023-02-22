# 迁移 MMCV 执行器到 MMEngine

## 简介

随着支持的深度学习任务越来越多，用户的需求不断增加，我们对 MMCV 已有的执行器（Runner）的灵活性和通用性有了更高的要求。
因此，MMEngine 在 MMCV 的基础上，实现了一个更加通用灵活的执行器以支持更多复杂的模型训练流程。
MMEngine 中的执行器扩大了作用域，也承担了更多的功能；我们抽象出了[训练循环控制器（EpochBasedTrainLoop/IterBasedTrainLoop）](mmengine.runner.EpochBasedTrainLoop)、[验证循环控制器（ValLoop）](mmengine.runner.ValLoop)和[测试循环控制器（TestLoop）](mmengine.runner.TestLoop)来方便用户灵活拓展模型的执行流程。

我们将首先介绍算法库的执行入口该如何从 MMCV 迁移到 MMEngine， 以最大程度地简化和统一执行入口的代码。
然后我们将详细介绍在 MMCV 和 MMEngine 中构造执行器及其内部组件进行训练的差异。
在开始迁移前，我们建议用户先阅读[执行器教程](../tutorials/runner.md)。

## 执行入口

以 MMDet 为例，我们首先展示基于 MMEngine 重构前后，配置文件和训练启动脚本的区别：

### 配置文件的迁移

<table class="docutils">
<thead>
  <tr>
    <th>基于 MMCV 执行器的配置文件概览 </th>
    <th>基于 MMEngine 执行器的配置文件概览</th>
<tbody>
<tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# default_runtime.py
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
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
# schedule.py

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

MMEngine 中的执行器提供了更多可自定义的部分，包括训练、验证、测试过程和数据加载器的配置，因此配置文件和之前相比会长一些。
为了方便用户的理解和阅读，我们遵循所见即所得的原则，重新调整了各个组件配置的层次，使得大部分一级字段都对应着执行器中关键属性的配置，例如数据加载器、评测器、钩子配置等。
这些配置在 OpenMMLab 2.0 算法库中都有默认配置，因此用户很多时候无需关心其中的大部分参数。

### 启动脚本的迁移

相比于 MMCV 的执行器，MMEngine 的执行器可以承担更多的功能，例如构建 `DataLoader`，构建分布式模型等。因此我们需要在配置文件中指定更多的参数，例如 `DataLoader` 的 `sampler` 和 `batch_sampler`，而无需在训练的启动脚本里实现构建 `DataLoader` 相关的代码。以 MMDet 的训练启动脚本为例:

<table class="docutils">
<thead>
  <tr>
    <th>基于 MMCV 执行器的训练启动脚本 </th>
    <th>基于 MMEngine 执行器的训练启动脚本</th>
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

上表对比了基于 MMCV 执行器和 MMEngine 执行器 MMDet 启动脚本的区别。
OpenMMLab 1.x 中的算法库都实现了一套 runner 的构建和训练流程，其中存在着大量的冗余代码。因此，MMEngine 的执行器在内部实现了很多流程化的代码以统一各个算法库的执行流程，例如初始化随机种子、初始化分布式环境、构建 `DataLoader` 等，使得下游算法库从此无需在训练启动脚本里实现相关代码，只需配置执行器的构造参数，就能够执行相应的流程。
基于 MMEngine 执行器的启动脚本不仅简化了 `tools/train.py` 的代码，甚至可以直接删除 `apis/train.py`，极大程度的简化了训练启动脚本。同样的，我们在基于 MMEngine 开发自己的代码仓库时，可以通过配置执行器参数来设置随机种子、初始化分布式环境，无需自行实现相关代码。

## 迁移执行器（Runner）

本节主要介绍 MMCV 执行器和 MMEngine 执行器在训练、验证、测试流程上的区别。
在使用 MMCV 执行器和 MMEngine 执行器训练、测试模型时，以下流程有着明显的不同：

01. [准备logger](#准备logger)
02. [设置随机种子](#设置随机种子)
03. [初始化环境变量](#初始化训练环境)
04. [准备数据](#准备数据)
05. [准备模型](#准备模型)
06. [准备优化器](#准备优化器)
07. [准备钩子](#准备训练钩子)
08. [准备验证/测试模块](#准备验证模块)
09. [构建执行器](#构建执行器)
10. [执行器加载检查点](#执行器加载检查点)
11. [开始训练](#执行器训练流程)、[开始测试](#执行器测试流程)
12. [迁移自定义训练流程](#迁移自定义执行流程)

后续的教程中，我们会对每个流程的差异进行详细介绍。

### 准备logger

**MMCV 准备 logger**

MMCV 需要在训练脚本里调用 `get_logger` 接口获得 logger，并用它输出、记录训练环境。

```python
logger = get_logger(name='custom', log_file=log_file, log_level=cfg.log_level)
env_info_dict = collect_env()
env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
dash_line = '-' * 60 + '\n'
logger.info('Environment info:\n' + dash_line + env_info + '\n' +
            dash_line)
```

执行器构造时，也需要传入 logger。

```python
runner = Runner(
    ...
    logger=logger
    ...)
```

**MMEngine 准备 logger**

在执行器构建时传入 logger 的日志等级，执行器构建时会自动创建 logger，并输出、记录训练环境。

```python
log_level = 'INFO'
```

### 设置随机种子

**MMCV 设置随机种子**

在训练脚本中手动设置随机种子：

```python
...
seed = init_random_seed(args.seed, device=cfg.device)
seed = seed + dist.get_rank() if args.diff_seed else seed
logger.info(f'Set random seed to {seed}, '
            f'deterministic: {args.deterministic}')
set_random_seed(seed, deterministic=args.deterministic)
...
```

**MMEngine 设计随机种子**

配置执行器的 `randomness` 参数，配置规则详见[执行器 api 文档](mmengine.runner.Runner.set_randomness)

**OpenMMLab 系列算法库配置变更**

<table class="docutils">
<thead>
  <tr>
    <th>MMCV 配置</th>
    <th>MMEngine 配置</th>
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

在本教程中，我们将 `randomness` 配置为：

```python
randomness = dict(seed=5)
```

### 初始化训练环境

**MMCV 初始化训练环境**

MMCV 需要在训练脚本中配置多进程启动方式、多进程通信后端等环境变量，并在执行器构建之前初始化分布式环境，对模型进行分布式封装：

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

**MMEngine 初始化训练环境**

MMEngine 通过配置 `env_cfg` 来选择多进程启动方式和多进程通信后端, 其默认值为 `dict(dist_cfg=dict(backend='nccl'))`，配置方式详见[执行器 api 文档](mmengine.runner.Runner.setup_env)。

执行器构建时接受 `launcher` 参数，如果其值不为 `'none'`，执行器构建时会自动执行分布式初始化，模型分布式封装。换句话说，使用 `MMEngine` 的执行器时，我们无需在执行器外做分布式相关的操作，只需配置 `launcher` 参数，选择训练的启动方式即可。

**OpenMMLab 系列算法库配置变更**

<table class="docutils">
<thead>
  <tr>
    <th>MMCV 配置</th>
    <th>MMEngine 配置</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
launcher = 'pytorch'  # 开启分布式训练
dist_params = dict(backend='nccl')  # 选择多进程通信后端
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

在本教程中，我们将 `env_cfg` 配置为：

```python
env_cfg = dict(dist_cfg=dict(backend='nccl'))
```

### 准备数据

MMCV 和 MMEngine 的执行器均可接受构建好的 `DataLoader` 实例。因此准备数据的流程没有差异：

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

**OpenMMLab 系列算法库配置变更**

<table class="docutils">
<thead>
  <tr>
    <th>MMCV 配置</th>
    <th>MMEngine 配置</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
data = dict(
    samples_per_gpu=2,  # 单卡 batch_size
    workers_per_gpu=2,  # Dataloader 采样进程数
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
    batch_size=2, # samples_per_gpu -> batch_size,
    num_workers=2,
    # 遍历完 DataLoader 后，是否重启多进程采样
    persistent_workers=True,
    # 可配置的 sampler
    sampler=dict(type='DefaultSampler', shuffle=True),
    # 可配置的 batch_sampler
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1, # 验证阶段的 batch_size
    num_workers=2,
    persistent_workers=True,
    drop_last=False, # 是否丢弃最后一个 batch
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

相比于 MMCV 的算法库配置，MMEngine 的配置更加复杂，但是也更加灵活。`DataLoader` 的配置过程由 `Runner` 负责，无需各个算法库实现。

### 准备模型

详见[迁移 MMCV 模型至 MMEngine](../migration/model.md)

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

需要注意的是，分布式训练时，MMCV 的执行器需要接受分布式封装后的模型，而 `MMEngine` 接受分布式封装前的模型，在执行器实例化阶段对其段进行分布式封装。

### 准备优化器

**MMCV 准备优化器**

MMCV 执行器构造时，可以直接接受 Pytorch 优化器，如

```python
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
```

对于复杂配置的优化器，MMCV 需要基于优化器构造器来构建优化器：

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

**MMEngine 准备优化器**

构建 MMEngine 执行器时，需要接受 `optim_wrapper` 参数，即[优化器封装](mmengine.optim.OptimWrapper)实例或者优化器封装配置，对于复杂配置的优化器封装，`MMEngine` 同样只需要配置 `optim_wrapper`。`optim_wrapper` 的详细介绍见[执行器 api 文档](mmengine.runner.Runner.build_optim_wrapper)。

**OpenMMLab 系列算法库配置变更**

<table class="docutils">
<thead>
  <tr>
    <th>MMCV 配置</th>
    <th>MMEngine 配置</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
optimizer = dict(
    constructor='CustomConstructor',
    type='AdamW',  # 优化器配置为一级字段
    lr=0.0001,  # 优化器配置为一级字段
    betas=(0.9, 0.999),  # 优化器配置为一级字段
    weight_decay=0.05,  # 优化器配置为一级字段
    paramwise_cfg={  # constructor 的参数
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6
    })
# MMCV 还需要配置 `optim_config`
# 来构建优化器钩子，而 MMEngine 不需要
optimizer_config = dict(grad_clip=None)
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
optim_wrapper = dict(
    constructor='CustomConstructor',
    type='OptimWrapper',  # 指定优化器封装类型
    optimizer=dict(  # 将优化器配置集中在 optimizer 内
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
对于检测、分类一类的上层任务（High level）MMCV 需要配置 `optim_config` 来构建优化器钩子，而 MMEngine 不需要。
```

本教程使用的 `optim_wrapper` 如下：

```python
from torch.optim import SGD

optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
optim_wrapper = dict(optimizer=optimizer)
```

### 准备训练钩子

**MMCV 准备训练钩子：**

MMCV 常用训练钩子的配置如下：

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

其中：

- `lr_config` 用于配置 `LrUpdaterHook`
- `optimizer_config` 用于配置 `OptimizerHook`
- `checkpoint_config` 用于配置 `CheckPointHook`
- `log_config` 用于配置 `LoggerHook`

除了上面提到的 4 个 Hook，MMCV 执行器自带 `IterTimerHook`。MMCV 需要先实例化执行器，再注册训练钩子，而 `MMEngine` 则在实例化阶段配置钩子。

**MMEngine 准备训练钩子**

MMEngine 执行器将 MMCV 常用的训练钩子配置成默认钩子：

- [RuntimeInfoHook](mmengine.hooks.RuntimeInfoHook)
- [IterTimerHook](mmengine.hooks.IterTimerHook)
- [DistSamplerSeedHook](mmengine.hooks.DistSamplerSeedHook)
- [LoggerHook](mmengine.hooks.LoggerHook)
- [CheckpointHook](mmengine.hooks.CheckpointHook)
- [ParamSchedulerHook](mmengine.hooks.ParamSchedulerHook)

对比上例中 MMCV 配置的训练钩子：

- `LrUpdaterHook` 对应 MMEngine 中的 `ParamSchedulerHook`，二者对应关系详见[迁移 `scheduler` 文档](./param_scheduler.md)
- MMEngine 在模型的 [train_step](mmengine.model.BaseModel.train_step) 时更新参数，因此不需要配置优化器钩子（`OptimizerHook`）
- MMEngine 自带 `CheckPointHook`，可以使用默认配置
- MMEngine 自带 `LoggerHook`，可以使用默认配置

因此我们只需要配置执行器[优化器参数调整策略（param_scheduler）](../tutorials/param_scheduler.md)，就能达到和 MMCV 示例一样的效果。
MMEngine 也支持注册自定义钩子，具体教程详见[钩子教程](../tutorials/hook.md#自定义钩子) 和[迁移 `hook` 文档](../migration/hook.md)。

<table class="docutils">
<thead>
  <tr>
    <th>MMCV 常用训练钩子</th>
    <th>MMEngine 默认钩子</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# MMCV 零散的配置训练钩子
# 配置 LrUpdaterHook，相当于 MMEngine 的参数调度器
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])

# 配置 OptimizerHook，MMEngine 不需要
optimizer_config = dict(grad_clip=None)

# 配置 LoggerHook
log_config = dict(  # LoggerHook
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# 配置 CheckPointHook
checkpoint_config = dict(interval=1)  # CheckPointHook
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
# 配置参数调度器
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

# MMEngine 集中配置默认钩子
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

```{note}
MMEngine 移除了 `OptimizerHook`，优化步骤在 model 中执行。
```

本教程使用的 param_scheduler 如下：

```python
from math import gamma

param_scheduler = dict(type='MultiStepLR', milestones=[2, 3], gamma=0.1)
```

### 准备验证模块

MMCV 借助 `EvalHook` 实现验证流程，受限于篇幅，这里不做进一步展开。MMEngine 通过[验证循环控制器（ValLoop）](mmengine.runner.ValLoop) 和[评测器（Evaluator）](../tutorials/evaluation.md)实现执行流程，如果我们想基于自定义的评价指标完成验证流程，则需要定义一个 `Metric`，并将其注册至 `METRICS` 注册器：

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

实现自定义 `Metric` 后，我们还需在执行器的构造参数中配置评测器和[验证循环控制器](mmengine.runner.ValLoop)，本教程中示例配置如下：

```python
val_evaluator = dict(type='ToyAccuracyMetric')
val_cfg = dict(type='ValLoop')
```

<table class="docutils">
<thead>
  <tr>
    <th>MMCV 配置验证流程</th>
    <th>MMEngine 配置验证流程</th>
<tbody>
  <tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
eval_cfg = cfg.get('evaluation', {})
eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
eval_hook = DistEvalHook if distributed else EvalHook  # 配置 EvalHook
runner.register_hook(
    eval_hook(val_dataloader, **eval_cfg), priority='LOW')  # 注册 EvalHook
```

</div>
  </td>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
val_dataloader = val_dataloader  # 配置验证数据
val_evaluator = dict(type='ToyAccuracyMetric')  # 配置评测器
val_cfg = dict(type='ValLoop')  # 配置验证循环控制器
```

</div>
  </td>
</tr>
</thead>
</table>

### 构建执行器

**MMCV 构建执行器**

```python
runner = EpochBasedRunner(
    model=model,
    optimizer=optimizer,
    work_dir=work_dir,
    logger=logger,
    max_epochs=4
)
```

**MMEngine 构建执行器**

`MMEngine` 执行器的作用域比 MMCV 更广，将设置随机种子、启动分布式训练等流程参数化。除了前几节提到的参数，上例中出现的`EpochBasedRunner`，`max_epochs`，`val_iterval` 现在由 `train_cfg` 配置：

- `by_epoch`: `True` 时相当于 MMCV 的 ``` EpochBasedRunner``，False ``` 时相当于 `IterBasedRunner`。
- `max_epoch`/`max_iters`: 同 MMCV 执行器的配置
- `val_iterval`: 同 `EvalHook` 的 `interval` 参数

`train_cfg` 实际上是训练循环控制器的构造参数，当 `by_epoch=True` 时，使用 `EpochBasedTrainLoop`。

```python
from mmengine.runner import Runner

runner = Runner(
    model=model,  # 待优化的模型
    work_dir='./work_dir',  # 配置工作目录
    randomness=randomness,  # 配置随机种子
    env_cfg=env_cfg,  # 配置环境变量
    launcher='none',  # 分布式训练启动方式
    optim_wrapper=optim_wrapper,  # 配置优化器
    param_scheduler=param_scheduler,  # 配置学习率调度器
    train_dataloader=train_dataloader,  # 配置训练数据
    train_cfg=dict(by_epoch=True, max_epochs=4, val_interval=1),  # 配置训练循环控制器
    val_dataloader=val_dataloader,  # 配置验证数据
    val_evaluator=val_evaluator,  # 配置评测器
    val_cfg=val_cfg)  # 配置验证循环控制器
```

### 执行器加载检查点

**MMCV 加载检查点**：

在训练之前执行加载权重、恢复训练的流程。

```python
if cfg.resume_from:
    runner.resume(cfg.resume_from)
elif cfg.load_from:
    runner.load_checkpoint(cfg.load_from)
```

**MMEngine 加载检查点**

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
    <th>MMCV 加载检查点配置</th>
    <th>MMEngine 加载检查点配置</th>
<tbody>
<tr>
  <td valign="top" class='two-column-table-wrapper' width="50%"><div style="overflow-x: auto">

```python
load_from = 'path/to/ckpt'
```

</div>
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

</div>
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

### 执行器训练流程

**MMCV 执行器训练流程**：

在训练之前执行加载权重、恢复训练的流程。然后再执行 `runner.run`，并传入训练数据。

```python
if cfg.resume_from:
    runner.resume(cfg.resume_from)
elif cfg.load_from:
    runner.load_checkpoint(cfg.load_from)
runner.run(data_loaders, cfg.workflow)
```

**MMEngine** 执行器训练流程

在执行器构建时配置加载权重、恢复训练参数

由于 MMEngine 的执行器在构造阶段就传入了训练数据，因此在调用 runner.train() 无需传入参数。

```python
runner.train()
```

### 执行器测试流程

MMCV 的执行器没有测试功能，因此需要自行实现测试脚本。MMEngine 的执行器只需要在构建时配置 `test_dataloader`、`test_cfg` 和 `test_evaluator`，然后再调用 `runner.test()` 就能完成测试流程。

**`work_dir` 和训练时一致，无需手动加载 checkpoint:**

```python
runner = Runner(
    model=model,
    work_dir='./work_dir',
    randomness=randomness,
    env_cfg=env_cfg,
    launcher='none',  # 不开启分布式训练
    optim_wrapper=optim_wrapper,
    train_dataloader=train_dataloader,
    train_cfg=dict(by_epoch=True, max_epochs=5, val_interval=1),
    val_dataloader=val_dataloader,
    val_evaluator=val_evaluator,
    val_cfg=val_cfg,
    test_dataloader=val_dataloader,  # 假设测试和验证使用相同的数据和评测器
    test_evaluator=val_evaluator,
    test_cfg=dict(type='TestLoop'),
)
runner.test()
```

**`work_dir` 和训练时不一致，需要额外指定 load_from:**

```python
runner = Runner(
    model=model,
    work_dir='./test_work_dir',
    load_from='./work_dir/epoch_5.pth',  # work_dir 不一致，指定 load_from，以加载指定的模型
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

### 迁移自定义执行流程

使用 MMCV 执行器时，我们可能会重载 `runner.train/runner.val` 或者 `runner.run_iter` 实现自定义的训练、测试流程。以重载 `runner.train` 为例，假设我们想对每个批次的图片训练两遍，我们可以这样重载 MMCV 的执行器：

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

在 MMEngine 中，要实现上述功能，我们需要重载一个新的循环控制器

```python
from mmengine.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop


@LOOPS.register_module()
class CustomEpochBasedTrainLoop(EpochBasedTrainLoop):
    def run_iter(self, idx, data_batch) -> None:
        for _ in range(2):
            super().run_iter(idx, data_batch)
```

在构建执行器时，指定 `train_cfg` 的 `type` 为 `CustomEpochBasedTrainLoop`。需要注意的是，`by_epoch` 和 `type` 不能同时配置，当配置 `by_epoch` 时，会推断训练循环控制器的类型为 `EpochBasedTrainLoop`。

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

如果有更加复杂的执行器迁移需求，可以参考[执行器教程](../tutorials/runner.md) 和[执行器设计文档](../design/runner.md)。
