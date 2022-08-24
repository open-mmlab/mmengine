.. role:: hidden
    :class: hidden-section

mmengine.utils
===================================

.. contents:: mmengine.utils
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmengine.utils

Manager
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ManagerMeta
   ManagerMixin

Path
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   check_file_exist
   fopen
   is_abs
   is_filepath
   mkdir_or_exist
   scandir
   symlink

Package
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   call_command
   check_install_package
   get_installed_path
   is_installed

Version
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   digit_version
   get_git_hash

Progress Bar
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ProgressBar
   track_iter_progress
   track_parallel_progress
   track_progress


Miscellaneous
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   is_list_of
   is_tuple_of
   is_seq_of
   is_str
   iter_cast
   list_cast
   tuple_cast
   concat_list
   slice_list
   to_1tuple
   to_2tuple
   to_3tuple
   to_4tuple
   to_ntuple
   check_prerequisites
   deprecated_api_warning
   has_method
   is_method_overridden
   import_modules_from_strings
   requires_executable
   requires_package
   Timer
   TimerError
   check_time

mmengine.utils.dl_utils
---------------------------

.. contents:: mmengine.utils
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmengine.utils.dl_utils

   collect_env
   load_url
   has_batch_norm
   is_norm
   mmcv_full_available
   tensor2imgs
   TORCH_VERSION
   set_multi_processing
   TimeCounter
   torch_meshgrid
   is_jit_tracing
