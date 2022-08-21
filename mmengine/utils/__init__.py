# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
from .manager import ManagerMeta, ManagerMixin
from .misc import (check_prerequisites, concat_list, deprecated_api_warning,
                   has_method, import_modules_from_strings, is_list_of,
                   is_method_overridden, is_seq_of, is_str, is_tuple_of,
                   iter_cast, list_cast, requires_executable, requires_package,
                   slice_list, to_1tuple, to_2tuple, to_3tuple, to_4tuple,
                   to_ntuple, tuple_cast)
from .package_utils import (call_command, check_install_package,
                            get_installed_path, is_installed)
from .parrots_wrapper import TORCH_VERSION
from .path import (check_file_exist, fopen, is_abs, is_filepath,
                   mkdir_or_exist, scandir, symlink)
from .version_utils import digit_version, get_git_hash

# TODO: creates intractable circular import issues
# from .time_counter import TimeCounter

__all__ = [
    'is_str', 'iter_cast', 'list_cast', 'tuple_cast', 'is_seq_of',
    'is_list_of', 'is_tuple_of', 'slice_list', 'concat_list',
    'check_prerequisites', 'requires_package', 'requires_executable',
    'is_filepath', 'fopen', 'check_file_exist', 'mkdir_or_exist', 'symlink',
    'scandir', 'deprecated_api_warning', 'import_modules_from_strings',
    'to_1tuple', 'to_2tuple', 'to_3tuple', 'to_4tuple', 'to_ntuple',
    'is_installed', 'call_command', 'get_installed_path',
    'check_install_package', 'is_abs', 'collect_env', 'is_method_overridden',
    'has_method', 'digit_version', 'get_git_hash', 'ManagerMeta',
    'ManagerMixin'
]

try:
    import torch
except ImportError:
    pass
else:
    from .collect_env import collect_env
    from .hub import load_url
    from .parrots_wrapper import TORCH_VERSION
    from .setup_env import set_multi_processing
    from .torch_misc import (has_batch_norm, is_norm, mmcv_full_available,
                             tensor2imgs)

    __all__.extend([
        'load_url', 'TORCH_VERSION', 'set_multi_processing', 'has_batch_norm',
        'is_norm', 'tensor2imgs', 'mmcv_full_available', 'collect_env'
    ])
