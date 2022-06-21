.. role:: hidden
    :class: hidden-section

mmengine.model
===================================

.. contents:: mmengine.model
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmengine.model

Module
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseModule
   ModuleDict
   ModuleList
   Sequential

Model
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseModel
   BaseDataPreprocessor
   ImgDataPreprocessor

EMA
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ExponentialMovingAverage
   MomentumAnnealingEMA
   StochasticWeightAverage

Wrapper
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   MMDistributedDataParallel
   MMSeparateDistributedDataParallel
   is_model_wrapper

Utils
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   detect_anomalous_params
   merge_dict
   stack_batch
