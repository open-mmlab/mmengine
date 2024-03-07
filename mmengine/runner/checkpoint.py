# Copyright (c) OpenMMLab. All rights reserved.
# All functions and classes in this file have been moved to mmengine.checkpoint
# Import them here to avoid BC
# flake8: noqa
from mmengine.checkpoint.io import (_IncompatibleKeys, _load_checkpoint,
                                    _load_checkpoint_to_model,
                                    _load_checkpoint_with_prefix,
                                    _save_to_state_dict, get_state_dict,
                                    load_checkpoint, load_state_dict,
                                    save_checkpoint, weights_to_cpu)
from mmengine.checkpoint.loader import (CheckpointLoader,
                                        _process_mmcls_checkpoint,
                                        load_from_ceph, load_from_http,
                                        load_from_local, load_from_mmcls,
                                        load_from_openmmlab, load_from_pavi,
                                        load_from_torchvision)
from mmengine.checkpoint.utils import (DEFAULT_CACHE_DIR, ENV_MMENGINE_HOME,
                                       ENV_XDG_CACHE_HOME, _get_mmengine_home,
                                       find_latest_checkpoint,
                                       get_deprecated_model_names,
                                       get_external_models, get_mmcls_models,
                                       get_torchvision_models)
