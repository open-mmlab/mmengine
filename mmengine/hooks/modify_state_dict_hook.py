# Copyright (c) OpenMMLab. All rights reserved.
import re

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
        self.name_mapping = name_mappings

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        from mmengine.runner.checkpoint import _load_checkpoint
        new_state_dict = dict()
        for checkpoint_path in self.merged_state_dicts:
            new_state_dict.update(_load_checkpoint(checkpoint_path))
        state_dict: dict = checkpoint['state_dict']

        for key, value in state_dict.items():
            if key in self.remove_keys:
                continue
            max_match_length = 0
            ori_string = ''
            new_string = ''
            for name_mapping in self.name_mapping:
                _ori_string, _new_string = name_mapping['ori'], name_mapping[
                    'new']
                match_length = len(
                    re.match(rf'(.*).{_ori_string}.(.*)', key).string)
                if match_length > max_match_length:
                    max_match_length = match_length
                    ori_string = _ori_string
                    new_string = _new_string
            new_key = key.replace(ori_string, new_string)
            new_state_dict[new_key] = value
        checkpoint['state_dict'] = new_state_dict
