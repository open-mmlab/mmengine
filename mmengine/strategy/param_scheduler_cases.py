# -------------------------------------------------------
# ParamScheduler configs
#
# Concerns:
#     - Single or Multiple: dict or list
#     - Instance or Config: _ParamScheduler
#     - Global: `type` is not None
#     - Local: `type` is None, keys corresponds to OptimWrapperDict
# -------------------------------------------------------

# 1. A single ParamScheduler on a single/multiple OptimWrapper(s)
# This will build N times for each OptimWrapper
param_scheduler = dict(
    type='MultiStepLR',
    milestones=[52000, 67600],
    gamma=0.1,
    by_epoch=False,
)
# instance counterpart, no build process is required
# Q: how to check it's valid?
param_scheduler = MultiStepLR(
    milestones=[52000, 67600],
    gamma=0.1,
    by_epoch=False,
)

# 2. Multiple ParamSchedulers on a single/multiple OptimWrapper(s)
# Each scheduler in list will be built N times for each OptimWrapper
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        begin=0,
        end=5000,
        by_epoch=False,
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=200_000,  # TODO, need more check
        eta_min=0,
        begin=0,
        end=200_000,
        by_epoch=False,
    )
]
# instance counterpart, no build process is required
param_scheduler = [
    LinearLR(
        start_factor=0.001,
        begin=0,
        end=5000,
        by_epoch=False,
    ),
    CosineAnnealingLR(
        T_max=200_000,  # TODO, need more check
        eta_min=0,
        begin=0,
        end=200_000,
        by_epoch=False,
    )
]

# 3. ParamSchedulers corresponds to OptimWrappers, 1:1
# This example is from `mmrazor`, also paste optim_wrapper here
optim_wrapper = dict(
    _delete_=True,
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(
        optimizer=dict(type='SGD', lr=0.1, weight_decay=5e-4, momentum=0.9)),
    generator=dict(optimizer=dict(type='AdamW', lr=1e-3)))
param_scheduler = dict(
    _delete_=True,
    architecture=dict(
        type='MultiStepLR',
        milestones=[100 * iter_size, 200 * iter_size],
        by_epoch=False),
    generator=dict(
        type='MultiStepLR',
        milestones=[100 * iter_size, 200 * iter_size],
        by_epoch=False))
# instance counterpart, no build process is required
param_scheduler = dict(
    architecture=MultiStepLR(
        milestones=[100 * iter_size, 200 * iter_size],
        by_epoch=False),
    generator=MultiStepLR(
        milestones=[100 * iter_size, 200 * iter_size],
        by_epoch=False))

# 4. ParamSchedulers corresponds to OptimWrappers, N:1
# This example is also from `mmrazor`
optim_wrapper = dict(
    _delete_=True,
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(optimizer=dict(type='AdamW', lr=1e-1)),
    generator=dict(optimizer=dict(type='AdamW', lr=1e-3)))
param_scheduler = dict(
    _delete_=True,
    architecture=[
        dict(type='LinearLR', end=500, by_epoch=False, start_factor=0.0001),
        dict(
            type='MultiStepLR',
            begin=500,
            milestones=[100 * 120, 200 * 120],
            by_epoch=False)
    ],
    generator=dict(
        type='LinearLR', end=500, by_epoch=False, start_factor=0.0001))
# instance counterpart, no build process is required
param_scheduler = dict(
    architecture=[
        LinearLR(end=500, by_epoch=False, start_factor=0.0001),
        MultiStepLR(
            begin=500,
            milestones=[100 * 120, 200 * 120],
            by_epoch=False)
    ],
    generator=LinearLR(end=500, by_epoch=False, start_factor=0.0001))
