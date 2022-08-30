.. role:: hidden
    :class: hidden-section

mmengine.fileio
===================================

.. contents:: mmengine.fileio
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmengine.fileio

File Client
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseStorageBackend
   FileClient
   HardDiskBackend
   HTTPBackend
   LmdbBackend
   MemcachedBackend
   PetrelBackend

File Handler
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseFileHandler
   JsonHandler
   PickleHandler
   YamlHandler

File IO
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   dump
   load
   register_handler

Parse File
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   dict_from_file
   list_from_file
