# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
from typing import List

from ..registry import HOOKS
from ..utils import is_list_of
from .hook import Hook


@HOOKS.register_module()
class RewriteCheckPointHook(Hook):
    """A hook to rewrite key in checkpoint.

    You can set ``applied_key`` to rewrite dictionary like instance saved in
    checkpoint.

    ``RewriteCheckPointHook`` has three mode to rewrite original checkpoint:

    - remove: Removes specified keys in target dictionary saved in checkpoint.
    - merge: Merge another state dictionary into the target dictionary.
    - name_mapping: Maps the original key to the target one, and overwrites it.


    Args:
        applied_key (str): Target state dictionary saved in checkpoints, which
            needs to be overwritten. Defaults to "state_dict".
        removed_keys (List[str]): Keys need to be removed. Defaults to [].
        name_mappings (List[dict]): A list of dictionary. Each dictionary has
            two keys: ``src`` and ``dst``. ``src`` means the original key
            prefix and ``src`` means the target key prefix, see more
            information in examples. Defaults to [].
        merged_state_dicts (List[str]): A list of checkpoint paths need to be
            merged. Defaults to [].

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>>
        >>> from mmengine.hooks import RewriteCheckPointHook
        >>>
        >>> class SubModule(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.layer1 = nn.Linear(1, 1)
        >>>         self.layer2 = nn.Linear(1, 1)

        >>> class Model(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.layer1 = nn.Linear(1, 1)
        >>>         self.layer2 = nn.Linear(1, 1)
        >>>         self.submodule = SubModule()
        >>>
        >>> # original `state_dict`.
        >>> model = Model()
        >>> model.state_dict().keys()
        >>> # ['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias',
        >>> #  'submodule.layer1.weight', 'submodule.layer1.bias',
        >>> #  'submodule.layer2.weight', 'submodule.layer2.bias']
        >>>
        >>> # remove `layer1` in `state_dict`.
        >>> checkpoint = dict(state_dict=model.state_dict())
        >>> hook = RewriteCheckPointHook(removed_keys='layer1')
        >>> hook.after_load_checkpoint(None, checkpoint)
        >>> checkpoint['state_dict'].keys()
        >>> # ['layer2.weight', 'layer2.bias', 'submodule.layer1.weight',
        >>> #  'submodule.layer1.bias', 'submodule.layer2.weight',
        >>> #  'submodule.layer2.bias']
        >>>
        >>> # remove key with prefix `submodule`.
        >>> checkpoint = dict(state_dict=model.state_dict())
        >>> hook = RewriteCheckPointHook(removed_keys='submodule')
        >>> hook.after_load_checkpoint(None, checkpoint)
        >>> checkpoint['state_dict'].keys()
        >>> # ['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias']
        >>>
        >>> # remapping prefix `module` to `submodule`.
        >>> checkpoint = dict(state_dict=model.state_dict())
        >>> hook = RewriteCheckPointHook(name_mappings=[dict(src='submodule', dst='module')])  # noqa: E501
        >>> hook.after_load_checkpoint(None, checkpoint)
        >>> checkpoint['state_dict'].keys()
        >>> # ['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias',
        >>> #  'module.layer1.weight', 'module.layer1.bias',
        >>> #  'module.layer2.weight', 'module.layer2.bias']
        >>>
        >>> # remapping prefix `module` to `submodule`, `layer1` to `linear1`.
        >>> checkpoint = dict(state_dict=model.state_dict())
        >>> hook = RewriteCheckPointHook(
        >>>     name_mappings=[dict(src='submodule', dst='module'),
        >>>                 dict(src='layer1', dst='linear1')])
        >>> hook.after_load_checkpoint(None, checkpoint)
        >>> checkpoint['state_dict'].keys()
        >>> # ['linear1.weight', 'linear1.bias', 'layer2.weight',
        >>> #  'layer2.bias', 'module.layer1.weight', 'module.layer1.bias',
        >>> #  'module.layer2.weight', 'module.layer2.bias']
        >>>
        >>> # merge other `state_dict`.
        >>> checkpoint = dict(state_dict=model.state_dict())
        >>> merged_ckpt = dict(state_dict=nn.Conv2d(1, 1, 1).state_dict())
        >>> torch.save(merged_ckpt, 'docs_demo.pth')
        >>> hook = RewriteCheckPointHook(
        >>>     merged_state_dicts=['docs_demo.pth'])
        >>> hook.after_load_checkpoint(None, checkpoint)
        >>> checkpoint['state_dict'].keys()
        >>> # ['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias',
        >>> #  'submodule.layer1.weight', 'submodule.layer1.bias',
        >>> #  'submodule.layer2.weight', 'submodule.layer2.bias',
        >>> #  'weight', 'bias'
    """

    priority = 'HIGH'

    def __init__(
        self,
        applied_key: str = 'state_dict',
        removed_keys: List[str] = [],
        name_mappings: List[dict] = [],
        merged_state_dicts: List[str] = [],
    ):
        assert isinstance(applied_key, str), (
            f'applied_key should be a string, but got {type(applied_key)}: '
            f'{applied_key}')
        if not isinstance(removed_keys, list):
            removed_keys = [removed_keys]  # type: ignore

        if not isinstance(name_mappings, list):
            name_mappings = [name_mappings]  # type: ignore

        if not isinstance(merged_state_dicts, list):
            merged_state_dicts = [merged_state_dicts]  # type: ignore

        assert is_list_of(removed_keys, str), (
            'removed_keys should be a list instance or a single string, '
            f'but got a {type(removed_keys)}: {removed_keys}')

        assert is_list_of(merged_state_dicts, str), (
            'merged_state_dicts should be a list or a single string, but got '
            f'{type(merged_state_dicts)}: {merged_state_dicts}')

        assert is_list_of(
            name_mappings,
            dict), ('name_mappings should be a list of dict a dict, but got '
                    f'{type(name_mappings)}: {name_mappings}')

        self.applied_key = applied_key
        self.removed_keys = removed_keys
        self.merged_state_dicts = merged_state_dicts
        self.name_mappings = name_mappings

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        """Overwrites the key of corresponding status dictionary in checkpoint.

        Args:
            runner (Runner): Runner before training.
            checkpoint (dict): loaded checkpoint.
        """
        new_state_dict: dict = dict()
        state_dict: dict = checkpoint[self.applied_key]

        for key in state_dict:
            if self._should_remove(key):
                continue
            self._remapping_key(key, state_dict, new_state_dict)
        self._merge_keys(new_state_dict)
        checkpoint[self.applied_key] = new_state_dict

    def _should_remove(self, key) -> bool:
        """Returns whether to remove the key.

        Args:
            key (str): Key from original state dictionary.

        Returns:
            bool: Whether to remove the key.
        """
        matched_removed_keys = []
        for removed_key in self.removed_keys:
            if re.match(rf'{removed_key}(.*)', key) is not None:
                matched_removed_keys.append(key)

        # Each key in `state_dict` should only match one removed_keys at most.
        if len(matched_removed_keys) == 0:
            return False
        elif len(matched_removed_keys) == 1:
            return True
        else:
            raise ValueError(
                f'removed_keys have a vague meaning, key: {key} '
                f'matched with {matched_removed_keys} at the same time')

    def _remapping_key(self, key: str, state_dict: dict,
                       new_state_dict: dict) -> None:
        """Maps the source key to the target key, and overwrites it.

        Args:
            key (str): Key from original state dictionary.
            state_dict (dict): Original state dictionary.
            new_state_dict (dict): New dictionary, of which key is remapped
                from original dictionary.
        """
        matched_remapping_keys = []
        for name_mapping in self.name_mappings:
            src, dst = name_mapping['src'], name_mapping['dst']
            if re.match(rf'{src}(.*)', key) is not None:
                matched_remapping_keys.append((src, dst))

        # Finds no mapped key,
        if len(matched_remapping_keys) == 0:
            src, dst = '', ''
        # Each key should only match one `name_mapping`.
        elif len(matched_remapping_keys) == 1:
            src, dst = matched_remapping_keys[0]
        else:
            raise ValueError(
                f'name_mappings have a vague meaning, key: {key} '
                f'matched with {matched_remapping_keys} at the same time')
        new_key = key.replace(src, dst)
        new_state_dict[new_key] = copy.deepcopy(state_dict[key])

    def _merge_keys(self, new_state_dict: dict) -> None:
        """Merges state dictionary from other checkpoints.

        Args:
            new_state_dict (dict): New dictionary, of which key is remapped
                from original dictionary.
        """
        from mmengine.runner.checkpoint import _load_checkpoint
        for checkpoint_path in self.merged_state_dicts:
            new_state_dict.update(
                _load_checkpoint(checkpoint_path)['state_dict'])
