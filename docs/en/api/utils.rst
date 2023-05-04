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

   call_command
   install_package
   get_installed_path
   is_installed

Version
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   digit_version
   get_git_hash

Progress Bar
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   ProgressBar

.. autosummary::
   :toctree: generated
   :nosignatures:

   track_iter_progress
   track_parallel_progress
   track_progress


Miscellaneous
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   Timer
   TimerError

.. autosummary::
   :toctree: generated
   :nosignatures:

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
   deprecated_function
   has_method
   is_method_overridden
   import_modules_from_strings
   requires_executable
   requires_package
   check_time
   apply_to
