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
   BaseTTAModel

EMA
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseAveragedModel
   ExponentialMovingAverage
   MomentumAnnealingEMA
   StochasticWeightAverage

Model Wrapper
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   MMDistributedDataParallel
   MMSeparateDistributedDataParallel
   MMFullyShardedDataParallel

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   is_model_wrapper

Weight Initialization
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseInit
   Caffe2XavierInit
   ConstantInit
   KaimingInit
   NormalInit
   PretrainedInit
   TruncNormalInit
   UniformInit
   XavierInit

.. autosummary::
   :toctree: generated
   :nosignatures:

   bias_init_with_prob
   caffe2_xavier_init
   constant_init
   initialize
   kaiming_init
   normal_init
   trunc_normal_init
   uniform_init
   update_init_info
   xavier_init

Utils
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   detect_anomalous_params
   merge_dict
   stack_batch
   revert_sync_batchnorm
   convert_sync_batchnorm
