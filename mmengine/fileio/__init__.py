# Copyright (c) OpenMMLab. All rights reserved.
from .backends import register_backend
from .file_client import (BaseStorageBackend, FileClient, HardDiskBackend,
                          HTTPBackend, LmdbBackend, MemcachedBackend,
                          PetrelBackend)
from .handlers import BaseFileHandler, JsonHandler, PickleHandler, YamlHandler
from .io import (copy_if_symlink_fails, copyfile, copyfile_from_local,
                 copyfile_to_local, copytree, copytree_from_local,
                 copytree_to_local, dump, exists, generate_presigned_url,
                 get_bytes, get_file_backend, get_local_path, get_text, isdir,
                 isfile, join_path, list_dir_or_file, load, put_bytes,
                 put_text, register_handler, rmfile, rmtree)
from .parse import dict_from_file, list_from_file

__all__ = [
    'BaseStorageBackend', 'FileClient', 'PetrelBackend', 'MemcachedBackend',
    'LmdbBackend', 'HardDiskBackend', 'HTTPBackend', 'copy_if_symlink_fails',
    'copyfile', 'copyfile_from_local', 'copyfile_to_local', 'copytree',
    'copytree_from_local', 'copytree_to_local', 'exists',
    'generate_presigned_url', 'get_bytes', 'get_file_backend',
    'get_local_path', 'get_text', 'isdir', 'isfile', 'join_path',
    'list_dir_or_file', 'put_bytes', 'put_text', 'rmfile', 'rmtree', 'load',
    'dump', 'register_handler', 'BaseFileHandler', 'JsonHandler',
    'PickleHandler', 'YamlHandler', 'list_from_file', 'dict_from_file',
    'register_backend'
]
