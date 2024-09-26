# Copyright (c) OpenMMLab. All rights reserved.
import io
import logging
import os.path as osp
import re
from collections import OrderedDict, namedtuple
from tempfile import TemporaryDirectory

import torch

from mmengine.dist import get_dist_info
from mmengine.fileio import FileClient, get_file_backend
from mmengine.logging import print_log
from mmengine.utils import apply_to, deprecated_function
from .loader import CheckpointLoader


def _load_checkpoint(filename, map_location=None, logger=None):
    """Load checkpoint from somewhere (modelzoo, file, url).

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str, optional): Same as :func:`torch.load`.
           Defaults to None.
        logger (:mod:`logging.Logger`, optional): The logger for error message.
           Defaults to None

    Returns:
        dict or OrderedDict: The loaded checkpoint. It can be either an
        OrderedDict storing model weights or a dict containing other
        information, which depends on the checkpoint.
    """
    return CheckpointLoader.load_checkpoint(filename, map_location, logger)


def _load_checkpoint_with_prefix(prefix, filename, map_location=None):
    """Load partial pretrained model with specific prefix.

    Args:
        prefix (str): The prefix of sub-module.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str | None): Same as :func:`torch.load`.
            Defaults to None.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """

    checkpoint = _load_checkpoint(filename, map_location=map_location)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    assert state_dict, f'{prefix} is not in the pretrained model'
    return state_dict


def _load_checkpoint_to_model(model,
                              checkpoint,
                              strict=False,
                              logger=None,
                              revise_keys=[(r'^module\.', '')]):

    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class _IncompatibleKeys(
        namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):

    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super().__repr__()

    __str__ = __repr__


def load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Defaults to False.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    from mmengine.model import BaseTTAModel, is_model_wrapper

    unexpected_keys = []
    missing_keys = []
    err_msg = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, local_state_dict, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_model_wrapper(module) or isinstance(module, BaseTTAModel):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(local_state_dict, prefix, local_metadata,
                                     True, missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + '.'
                child_state_dict = {
                    k: v
                    for k, v in local_state_dict.items()
                    if k.startswith(child_prefix)
                }
                load(child, child_state_dict, child_prefix)

        # Note that the hook can modify missing_keys and unexpected_keys.
        incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
        if hasattr(module, '_load_state_dict_post_hooks'):
            for hook in module._load_state_dict_post_hooks.values():
                out = hook(module, incompatible_keys)
                assert out is None, (
                    'Hooks registered with '
                    '``register_load_state_dict_post_hook`` are not expected '
                    'to return new values, if incompatible_keys need to be '
                    'modified, it should be done inplace.')

    load(module, state_dict)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print_log(err_msg, logger=logger, level=logging.WARNING)


def load_checkpoint(model,
                    filename,
                    map_location=None,
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Defaults to strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')

    return _load_checkpoint_to_model(model, checkpoint, strict, logger,
                                     revise_keys)


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    # stash metadata to put in state_dict later
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    state_dict = apply_to(state_dict, lambda x: hasattr(x, 'cpu'),
                          lambda x: x.cpu())
    state_dict._metadata = metadata
    return state_dict


@deprecated_function(
    since='0.3.0',
    removed_in='0.5.0',
    instructions='`_save_to_state_dict` will be deprecated in the future, '
    'please use `nn.Module._save_to_state_dict` directly.')
def _save_to_state_dict(module, destination, prefix, keep_vars):
    """Saves module state to `destination` dictionary.

    This method is modified from :meth:`torch.nn.Module._save_to_state_dict`.

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (dict): A dict where state will be stored.
        prefix (str): The prefix for parameters and buffers used in this
            module.
        keep_vars (bool): Whether to keep the variable property of the
            parameters.
    """
    for name, param in module._parameters.items():
        if param is not None:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        if buf is not None and name not in module._non_persistent_buffers_set:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module, destination=None, prefix='', keep_vars=False):
    """Returns a dictionary containing a whole state of the module.

    Both parameters and persistent buffers (e.g. running averages) are
    included. Keys are corresponding parameter and buffer names.
    This method is modified from :meth:`torch.nn.Module.state_dict` to
    recursively check parallel module in case that the model has a complicated
    structure, e.g., nn.Module(nn.Module(DDP)).

    Args:
        module (nn.Module): The module to generate state_dict.
        destination (OrderedDict): Returned dict for the state of the
            module.
        prefix (str): Prefix of the key.
        keep_vars (bool): Whether to keep the variable property of the
            parameters. Defaults to False.

    Returns:
        dict: A dictionary containing a whole state of the module.
    """
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    from mmengine.model import is_model_wrapper
    if is_model_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
    destination._metadata[prefix[:-1]] = local_metadata = dict(
        version=module._version)
    module._save_to_state_dict(destination, prefix, keep_vars)
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination


def save_checkpoint(checkpoint,
                    filename,
                    file_client_args=None,
                    backend_args=None):
    """Save checkpoint to file.

    Args:
        checkpoint (dict): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            `backend_args` instead.
        backend_args (dict, optional): Arguments to instantiate the
            prefix of uri corresponding backend. Defaults to None.
            New in v0.2.0.
    """
    if file_client_args is not None:
        print_log(
            '"file_client_args" will be deprecated in future. '
            'Please use "backend_args" instead',
            logger='current',
            level=logging.WARNING)
        if backend_args is not None:
            raise ValueError(
                '"file_client_args" and "backend_args" cannot be set '
                'at the same time.')

    if filename.startswith('pavi://'):
        if file_client_args is not None or backend_args is not None:
            raise ValueError(
                '"file_client_args" or "backend_args" should be "None" if '
                'filename starts with "pavi://"')
        try:
            from pavi import exception, modelcloud
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except exception.NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        file_client = FileClient.infer_client(file_client_args, filename)
        if file_client_args is None:
            file_backend = get_file_backend(
                filename, backend_args=backend_args)
        else:
            file_backend = file_client

        with io.BytesIO() as f:
            torch.save(checkpoint, f)
            file_backend.put(f.getvalue(), filename)
