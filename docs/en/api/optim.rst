.. role:: hidden
    :class: hidden-section

mmengine.optim
===================================

.. contents:: mmengine.optim
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmengine.optim

Optimizer
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

    AmpOptimWrapper
    ApexOptimWrapper
    OptimWrapper
    OptimWrapperDict
    DefaultOptimWrapperConstructor
    ZeroRedundancyOptimizer

.. autosummary::
   :toctree: generated
   :nosignatures:

    build_optim_wrapper

Scheduler
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   _ParamScheduler
   ConstantLR
   ConstantMomentum
   ConstantParamScheduler
   CosineAnnealingLR
   CosineAnnealingMomentum
   CosineAnnealingParamScheduler
   ExponentialLR
   ExponentialMomentum
   ExponentialParamScheduler
   LinearLR
   LinearMomentum
   LinearParamScheduler
   MultiStepLR
   MultiStepMomentum
   MultiStepParamScheduler
   OneCycleLR
   OneCycleParamScheduler
   PolyLR
   PolyMomentum
   PolyParamScheduler
   StepLR
   StepMomentum
   StepParamScheduler
   ReduceOnPlateauLR
   ReduceOnPlateauMomentum
   ReduceOnPlateauParamScheduler
