.. role:: hidden
    :class: hidden-section

mmengine.fileio
===================================

.. contents:: mmengine.fileio
   :depth: 2
   :local:
   :backlinks: top

.. currentmodule:: mmengine.fileio

File Backend
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: classtemplate.rst

   BaseStorageBackend
   FileClient
   HardDiskBackend
   LocalBackend
   HTTPBackend
   LmdbBackend
   MemcachedBackend
   PetrelBackend

.. autosummary::
   :toctree: generated
   :nosignatures:

   register_backend

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

.. autosummary::
   :toctree: generated
   :nosignatures:

   register_handler

File IO
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dump
   load
   copy_if_symlink_fails
   copyfile
   copyfile_from_local
   copyfile_to_local
   copytree
   copytree_from_local
   copytree_to_local
   exists
   generate_presigned_url
   get
   get_file_backend
   get_local_path
   get_text
   isdir
   isfile
   join_path
   list_dir_or_file
   put
   put_text
   remove
   rmtree

Parse File
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   dict_from_file
   list_from_file
