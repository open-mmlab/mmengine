# Use modules from other libraries

Based on MMEngine's [Registry](registry.md) and [Config](config.md), users can build modules across libraries.
For example, use [MMPretrain](https://github.com/open-mmlab/mmpretrain)'s backbones in [MMDetection](https://github.com/open-mmlab/mmdetection), or [MMDetection](https://github.com/open-mmlab/mmdetection)'s data transforms in [MMRotate](https://github.com/open-mmlab/mmrotate), or using [MMDetection](https://github.com/open-mmlab/mmdetection)'s detectors in [MMTracking](https://github.com/open-mmlab/mmtracking).

Modules registered in the same registry tree can be called across libraries by adding the **package name prefix** before the module's type in the config. Here are some common examples:

## Use backbone across libraries

Taking the example of using MMPretrain's ConvNeXt in MMDetection:

Firstly, adding the `custom_imports` field to the config to register the backbones of MMPretrain to the registry.

Secondly, adding the package name of MMPretrain `mmpretrain` to the `type` of the backbone as a prefix: `mmpretrain.ConvNeXt`

```python
# Use custom_imports to register mmpretrain models to the registry
custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

model = dict(
  type='MaskRCNN',
  data_preprocessor=dict(...),
  backbone=dict(
      type='mmpretrain.ConvNeXt', # Add mmpretrain prefix to enable cross-library mechanism
      arch='tiny',
      out_indices=[0, 1, 2, 3],
      drop_path_rate=0.4,
      layer_scale_init_value=1.0,
      gap_before_final_norm=False,
      init_cfg=dict(
          type='Pretrained',
          checkpoint=
          'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth',
          prefix='backbone.')),
  neck=dict(...),
  rpn_head=dict(...))
```

## Use data transform across libraries

As with the example of backbone above, cross-library calls can be simply achieved by adding custom_imports and prefix in the config:

```python
# Use custom_imports to register mmdet transforms to the registry
custom_imports = dict(imports=['mmdet.datasets.transforms'], allow_failed_imports=False)

# Add mmdet prefix to enable cross-library mechanism
train_pipeline=[
    dict(type='mmdet.LoadImageFromFile'),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 2014), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.PackDetInputs')
]
```

## Use detector across libraries

Using an algorithm from another library is a little bit complex.

An algorithm contains multiple submodules. Each submodule needs to add a prefix to its `type`. Take  using MMDetection's YOLOX in MMTracking as an example:

```python
# Use custom_imports to register mmdet models to the registry
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)

model = dict(
    type='mmdet.YOLOX',
    backbone=dict(type='mmdet.CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        type='mmdet.YOLOXPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4),
    bbox_head=dict(
        type='mmdet.YOLOXHead', num_classes=1, in_channels=320, feat_channels=320),
    train_cfg=dict(assigner=dict(type='mmdet.SimOTAAssigner', center_radius=2.5)))
```

To prevent adding prefix to all of the submodules manually, the `_scope_` keyword is introduced. When the `_scope_` keyword is added to the config of a module, all submodules' scope will be changed by the `_scope_` keyword. Here is an example config:

```python
# Use custom_imports to register mmdet models to the registry
custom_imports = dict(imports=['mmdet.models'], allow_failed_imports=False)

model = dict(
    _scope_='mmdet', # use the _scope_ keyword to avoid adding prefix to all submodules
    type='YOLOX',
    backbone=dict(type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4),
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=320, feat_channels=320),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)))
```

These two examples are equivalent to each other.

If you want to know more about the registry and config, please refer to [Config Tutorial](config.md) and [Registry Tutorial](registry.md)
