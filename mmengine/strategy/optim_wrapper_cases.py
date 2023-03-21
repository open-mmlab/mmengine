# -------------------------------------------------------
# OptimWrapper configs
#
# NOTE: We should try to split 3 levels of build process:
#     - Optimizer
#     - OptimWrapper
#     - OptimWrapperDict
#
# Concerns:
#     - Single or Multiple(OptimWrapperDict): `type` or keywords
#     - Instance or Config: optimizer=Optimizer, encoder=OptimWrapper, ...
#     - constructor: None or assigned
#     - `type`: given or not
#     - paramwise_cfg: given or not
#     - arguments: in `wrapper` or `strategy`
# -------------------------------------------------------

# 1. Common case: from mmcls swin-v2-large
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=5e-6,
        weight_decay=0.0005,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
    # accumulative_counts=2,  # self-sup use this param very often
)
# Instance counter part, should build optim_wrapper only
optim_wrapper = dict(
    optimizer=AdamW(
        lr=5e-6,
        weight_decay=0.0005,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
    clip_grad=dict(max_norm=5.0),
    # accumulative_counts=2,
)
# Instance counter part 2, will not illustrate below
optim_wrapper = OptimWrapper(optimizer=...)


# 2. Type assigned to enable FP16, from mmcls resnet50-fp16
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001), type='AmpOptimWrapper',
    loss_scale='dynamic')
# Instance counter part
optim_wrapper = dict(
    optimizer=SGD(lr=0.1, momentum=0.9, weight_decay=0.0001),
    type='AmpOptimWrapper',
    loss_scale='dynamic')


# 3. Type assigned as OptimWrapper, from mmdet detr-r50
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# 4. Constructor given, single optimizer, from mmdet mask-rcnn-convnext
optim_wrapper = dict(
    type='AmpOptimWrapper',
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'layer_wise',
        'num_layers': 6
    },
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ))

# 5. Constructor given as defulat, single optimizer, from mmediting
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# 6. User-defined constructor given, multiple optimizer, from mmediting
optim_wrapper = dict(
    constructor='PGGANOptimWrapperConstructor',
    generator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
    discriminator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.001, betas=(0., 0.99))),
    lr_schedule=dict(
        generator={
            '128': 0.0015,
            '256': 0.002,
            '512': 0.003,
            '1024': 0.003
        },
        discriminator={
            '128': 0.0015,
            '256': 0.002,
            '512': 0.003,
            '1024': 0.003
        }))
# Instance counterpart. Q: AmpOptimWrapper or user-defined OptimWraper?
# Error!!! This partially built style is not supported
optim_wrapper = dict(
    generator=dict(optimizer=Optimizer(), ...),
    discriminator=dict(optimizer=Optimizer(), ...)
)
# Instance counterpart 2
optim_wrapper = dict(generator=OptimWrapper(), discriminator=OptimWraper())
# Instance counterpart 3
optim_wrapper = OptimWrapperDict(generator=OptimWrapper(), discriminator=OptimWrapper())
