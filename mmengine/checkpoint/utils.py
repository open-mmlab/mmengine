# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pkgutil
from importlib import import_module
from typing import Optional

import mmengine
from mmengine.fileio import load as load_file
from mmengine.logging import print_log
from mmengine.utils import digit_version, mkdir_or_exist

# `MMENGINE_HOME` is the highest priority directory to save checkpoints
# downloaded from Internet. If it is not set, as a workaround, using
# `XDG_CACHE_HOME`` or `~/.cache` instead.
# Note that `XDG_CACHE_HOME` defines the base directory relative to which
# user-specific non-essential data files should be stored. If `XDG_CACHE_HOME`
# is either not set or empty, a default equal to `~/.cache` should be used.
ENV_MMENGINE_HOME = 'MMENGINE_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


def _get_mmengine_home():
    mmengine_home = os.path.expanduser(
        os.getenv(
            ENV_MMENGINE_HOME,
            os.path.join(
                os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'mmengine')))

    mkdir_or_exist(mmengine_home)
    return mmengine_home


def get_torchvision_models():
    import torchvision
    if digit_version(torchvision.__version__) < digit_version('0.13.0a0'):
        model_urls = dict()
        # When the version of torchvision is lower than 0.13, the model url is
        # not declared in `torchvision.model.__init__.py`, so we need to
        # iterate through `torchvision.models.__path__` to get the url for each
        # model.
        for _, name, ispkg in pkgutil.walk_packages(
                torchvision.models.__path__):
            if ispkg:
                continue
            _zoo = import_module(f'torchvision.models.{name}')
            if hasattr(_zoo, 'model_urls'):
                _urls = getattr(_zoo, 'model_urls')
                model_urls.update(_urls)
    else:
        # Since torchvision bumps to v0.13, the weight loading logic,
        # model keys and model urls have been changed. Here the URLs of old
        # version is loaded to avoid breaking back compatibility. If the
        # torchvision version>=0.13.0, new URLs will be added. Users can get
        # the resnet50 checkpoint by setting 'resnet50.imagent1k_v1',
        # 'resnet50' or 'ResNet50_Weights.IMAGENET1K_V1' in the config.
        json_path = osp.join(mmengine.__path__[0], 'hub/torchvision_0.12.json')
        model_urls = mmengine.load(json_path)
        if digit_version(torchvision.__version__) < digit_version('0.14.0a0'):
            weights_list = [
                cls for cls_name, cls in torchvision.models.__dict__.items()
                if cls_name.endswith('_Weights')
            ]
        else:
            weights_list = [
                torchvision.models.get_model_weights(model)
                for model in torchvision.models.list_models(torchvision.models)
            ]

        for cls in weights_list:
            # The name of torchvision model weights classes ends with
            # `_Weights` such as `ResNet18_Weights`. However, some model weight
            # classes, such as `MNASNet0_75_Weights` does not have any urls in
            # torchvision 0.13.0 and cannot be iterated. Here we simply check
            # `DEFAULT` attribute to ensure the class is not empty.
            if not hasattr(cls, 'DEFAULT'):
                continue
            # Since `cls.DEFAULT` can not be accessed by iterating cls, we set
            # default urls explicitly.
            cls_name = cls.__name__
            cls_key = cls_name.replace('_Weights', '').lower()
            model_urls[f'{cls_key}.default'] = cls.DEFAULT.url
            for weight_enum in cls:
                cls_key = cls_name.replace('_Weights', '').lower()
                cls_key = f'{cls_key}.{weight_enum.name.lower()}'
                model_urls[cls_key] = weight_enum.url

    return model_urls


def get_external_models():
    mmengine_home = _get_mmengine_home()
    default_json_path = osp.join(mmengine.__path__[0], 'hub/openmmlab.json')
    default_urls = load_file(default_json_path)
    assert isinstance(default_urls, dict)
    external_json_path = osp.join(mmengine_home, 'open_mmlab.json')
    if osp.exists(external_json_path):
        external_urls = load_file(external_json_path)
        assert isinstance(external_urls, dict)
        default_urls.update(external_urls)

    return default_urls


def get_mmcls_models():
    mmcls_json_path = osp.join(mmengine.__path__[0], 'hub/mmcls.json')
    mmcls_urls = load_file(mmcls_json_path)

    return mmcls_urls


def get_deprecated_model_names():
    deprecate_json_path = osp.join(mmengine.__path__[0], 'hub/deprecated.json')
    deprecate_urls = load_file(deprecate_json_path)
    assert isinstance(deprecate_urls, dict)

    return deprecate_urls


def find_latest_checkpoint(path: str) -> Optional[str]:
    """Find the latest checkpoint from the given path.

    Refer to https://github.com/facebookresearch/fvcore/blob/main/fvcore/common/checkpoint.py  # noqa: E501

    Args:
        path(str): The path to find checkpoints.

    Returns:
        str or None: File path of the latest checkpoint.
    """
    save_file = osp.join(path, 'last_checkpoint')
    last_saved: Optional[str]
    if os.path.exists(save_file):
        with open(save_file) as f:
            last_saved = f.read().strip()
    else:
        print_log('Did not find last_checkpoint to be resumed.')
        last_saved = None
    return last_saved
