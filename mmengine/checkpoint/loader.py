# Copyright (c) OpenMMLab. All rights reserved.
import io
import logging
import os
import os.path as osp
import re
from collections import OrderedDict
from tempfile import TemporaryDirectory
from typing import Callable, Dict

import torch

from mmengine.dist import get_dist_info
from mmengine.fileio import get_file_backend
from mmengine.logging import print_log
from mmengine.utils.dl_utils import load_url
from .utils import (_get_mmengine_home, get_deprecated_model_names,
                    get_external_models, get_mmcls_models,
                    get_torchvision_models)


class CheckpointLoader:
    """A general checkpoint loader to manage all schemes."""

    _schemes: Dict[str, Callable] = {}

    @classmethod
    def _register_scheme(cls, prefixes, loader, force=False):
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        else:
            assert isinstance(prefixes, (list, tuple))
        for prefix in prefixes:
            if (prefix not in cls._schemes) or force:
                cls._schemes[prefix] = loader
            else:
                raise KeyError(
                    f'{prefix} is already registered as a loader backend, '
                    'add "force=True" if you want to override it')
        # sort, longer prefixes take priority
        cls._schemes = OrderedDict(
            sorted(cls._schemes.items(), key=lambda t: t[0], reverse=True))

    @classmethod
    def register_scheme(cls, prefixes, loader=None, force=False):
        """Register a loader to CheckpointLoader.

        This method can be used as a normal class method or a decorator.

        Args:
            prefixes (str or list[str] or tuple[str]):
            The prefix of the registered loader.
            loader (function, optional): The loader function to be registered.
                When this method is used as a decorator, loader is None.
                Defaults to None.
            force (bool, optional): Whether to override the loader
                if the prefix has already been registered. Defaults to False.
        """

        if loader is not None:
            cls._register_scheme(prefixes, loader, force=force)
            return

        def _register(loader_cls):
            cls._register_scheme(prefixes, loader_cls, force=force)
            return loader_cls

        return _register

    @classmethod
    def _get_checkpoint_loader(cls, path):
        """Finds a loader that supports the given path. Falls back to the local
        loader if no other loader is found.

        Args:
            path (str): checkpoint path

        Returns:
            callable: checkpoint loader
        """
        for p in cls._schemes:
            # use regular match to handle some cases that where the prefix of
            # loader has a prefix. For example, both 's3://path' and
            # 'open-mmlab:s3://path' should return `load_from_ceph`
            if re.match(p, path) is not None:
                return cls._schemes[p]

    @classmethod
    def load_checkpoint(cls, filename, map_location=None, logger='current'):
        """load checkpoint through URL scheme path.

        Args:
            filename (str): checkpoint file name with given prefix
            map_location (str, optional): Same as :func:`torch.load`.
                Defaults to None
            logger (str): The logger for message. Defaults to 'current'.

        Returns:
            dict or OrderedDict: The loaded checkpoint.
        """

        checkpoint_loader = cls._get_checkpoint_loader(filename)
        class_name = checkpoint_loader.__name__
        print_log(
            f'Loads checkpoint by {class_name[10:]} backend from path: '
            f'{filename}',
            logger=logger)
        return checkpoint_loader(filename, map_location)


@CheckpointLoader.register_scheme(prefixes='')
def load_from_local(filename, map_location):
    """load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f'{filename} can not be found.')
    checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('http://', 'https://'))
def load_from_http(filename,
                   map_location=None,
                   model_dir=None,
                   progress=os.isatty(0)):
    """load checkpoint through HTTP or HTTPS scheme path. In distributed
    setting, this function only download checkpoint at local rank 0.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.
        model_dir (string, optional): directory in which to save the object,
            Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    rank, world_size = get_dist_info()
    if rank == 0:
        checkpoint = load_url(
            filename,
            model_dir=model_dir,
            map_location=map_location,
            progress=progress)
    if world_size > 1:
        torch.distributed.barrier()
        if rank > 0:
            checkpoint = load_url(
                filename,
                model_dir=model_dir,
                map_location=map_location,
                progress=progress)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes='pavi://')
def load_from_pavi(filename, map_location=None):
    """load checkpoint through the file path prefixed with pavi. In distributed
    setting, this function download ckpt at all ranks to different temporary
    directories.

    Args:
        filename (str): checkpoint file path with pavi prefix
        map_location (str, optional): Same as :func:`torch.load`.
          Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    assert filename.startswith('pavi://'), \
        f'Expected filename startswith `pavi://`, but get {filename}'
    model_path = filename[7:]

    try:
        from pavi import modelcloud
    except ImportError:
        raise ImportError(
            'Please install pavi to load checkpoint from modelcloud.')

    model = modelcloud.get(model_path)
    with TemporaryDirectory() as tmp_dir:
        downloaded_file = osp.join(tmp_dir, model.name)
        model.download(downloaded_file)
        checkpoint = torch.load(downloaded_file, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(
    prefixes=[r'(\S+\:)?s3://', r'(\S+\:)?petrel://'])
def load_from_ceph(filename, map_location=None, backend='petrel'):
    """load checkpoint through the file path prefixed with s3.  In distributed
    setting, this function download ckpt at all ranks to different temporary
    directories.

    Args:
        filename (str): checkpoint file path with s3 prefix
        map_location (str, optional): Same as :func:`torch.load`.
        backend (str, optional): The storage backend type.
            Defaults to 'petrel'.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    file_backend = get_file_backend(
        filename, backend_args={'backend': backend})
    with io.BytesIO(file_backend.get(filename)) as buffer:
        checkpoint = torch.load(buffer, map_location=map_location)
    return checkpoint


@CheckpointLoader.register_scheme(prefixes=('modelzoo://', 'torchvision://'))
def load_from_torchvision(filename, map_location=None):
    """load checkpoint through the file path prefixed with modelzoo or
    torchvision.

    Args:
        filename (str): checkpoint file path with modelzoo or
            torchvision prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    model_urls = get_torchvision_models()
    if filename.startswith('modelzoo://'):
        print_log(
            'The URL scheme of "modelzoo://" is deprecated, please '
            'use "torchvision://" instead',
            logger='current',
            level=logging.WARNING)
        model_name = filename[11:]
    else:
        model_name = filename[14:]
    return load_from_http(model_urls[model_name], map_location=map_location)


@CheckpointLoader.register_scheme(prefixes=('open-mmlab://', 'openmmlab://'))
def load_from_openmmlab(filename, map_location=None):
    """load checkpoint through the file path prefixed with open-mmlab or
    openmmlab.

    Args:
        filename (str): checkpoint file path with open-mmlab or
        openmmlab prefix
        map_location (str, optional): Same as :func:`torch.load`.
          Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_external_models()
    prefix_str = 'open-mmlab://'
    if filename.startswith(prefix_str):
        model_name = filename[13:]
    else:
        model_name = filename[12:]
        prefix_str = 'openmmlab://'

    deprecated_urls = get_deprecated_model_names()
    if model_name in deprecated_urls:
        print_log(
            f'{prefix_str}{model_name} is deprecated in favor '
            f'of {prefix_str}{deprecated_urls[model_name]}',
            logger='current',
            level=logging.WARNING)
        model_name = deprecated_urls[model_name]
    model_url = model_urls[model_name]
    # check if is url
    if model_url.startswith(('http://', 'https://')):
        checkpoint = load_from_http(model_url, map_location=map_location)
    else:
        filename = osp.join(_get_mmengine_home(), model_url)
        if not osp.isfile(filename):
            raise FileNotFoundError(f'{filename} can not be found.')
        checkpoint = torch.load(filename, map_location=map_location)
    return checkpoint


def _process_mmcls_checkpoint(checkpoint):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Some checkpoints converted from 3rd-party repo don't
        # have the "state_dict" key.
        state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k[9:]] = v
    new_checkpoint = dict(state_dict=new_state_dict)

    return new_checkpoint


@CheckpointLoader.register_scheme(prefixes='mmcls://')
def load_from_mmcls(filename, map_location=None):
    """load checkpoint through the file path prefixed with mmcls.

    Args:
        filename (str): checkpoint file path with mmcls prefix
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    model_urls = get_mmcls_models()
    model_name = filename[8:]
    checkpoint = load_from_http(
        model_urls[model_name], map_location=map_location)
    checkpoint = _process_mmcls_checkpoint(checkpoint)
    return checkpoint
