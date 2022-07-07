# Copyright (c) OpenMMLab. All rights reserved.
import re
import copy

from ..utils import is_list_of
from .hook import Hook


class ModifyStateDictHook(Hook):
    priority = 'NORMAL'

    def __init__(self,
                 remove_keys: list = [],
                 merged_state_dicts: list = [],
                 name_mappings: list = []):
        assert is_list_of(
            remove_keys,
            str), ('remove_keys should be a list instance, but got a '
                   f'{type(remove_keys)} instance')
        assert is_list_of(
            merged_state_dicts,
            str), ('merged_state_dicts should be a list instance, but got a '
                   f'{type(merged_state_dicts)} instance')
        self.remove_keys = remove_keys
        self.merged_state_dicts = merged_state_dicts
        self.name_mappings = name_mappings

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        new_state_dict = dict()
        state_dict: dict = checkpoint['state_dict']

        for key in state_dict:
            if self._should_remove(key, state_dict, new_state_dict):
                continue
            self._remapping_key(key, state_dict, new_state_dict)
        self._merge_keys(new_state_dict)
        checkpoint['state_dict'] = new_state_dict

    def _should_remove(self, key, state_dict, new_state_dict):
        matched_removed_keys = []
        for removed_key in self.remove_keys:
            if re.match(rf'{removed_key}(.*)', key) is not None:
                matched_removed_keys.append(key)
        if len(matched_removed_keys) == 0:
            return False
        elif len(matched_removed_keys) == 1:
            return True
        elif len(matched_removed_keys) > 1:
            raise ValueError(
                f'removed_keys have a vague meaning, key: {key} '
                f'matched with {matched_removed_keys} at the same time')

    def _remapping_key(self, key, state_dict, new_state_dict):
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
        new_state_dict[new_key] = state_dict[key]

    def _merge_keys(self, new_state_dict):
        from mmengine.runner.checkpoint import _load_checkpoint
        for checkpoint_path in self.merged_state_dicts:
            new_state_dict.update(_load_checkpoint(checkpoint_path))