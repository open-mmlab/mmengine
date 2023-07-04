.. role:: hidden
    :class: hidden-section

mmengine.runner
===================================

.. contents:: mmengine.runner
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmengine.runner

Runner
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Runner
   FlexibleRunner

Loop
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseLoop
   EpochBasedTrainLoop
   IterBasedTrainLoop
   ValLoop
   TestLoop

Checkpoints
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   CheckpointLoader

.. autosummary::
   :toctree: generated
   :nosignatures:

   find_latest_checkpoint
   get_deprecated_model_names
   get_external_models
   get_mmcls_models
   get_state_dict
   get_torchvision_models
   load_checkpoint
   load_state_dict
   save_checkpoint
   weights_to_cpu

AMP
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   autocast

Miscellaneous
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   LogProcessor
   Priority

.. autosummary::
   :toctree: generated
   :nosignatures:

   get_priority
   set_random_seed
