# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys

import pkg_resources
import pytest

from mmengine.utils import get_installed_path, is_installed


def test_is_installed():
    # TODO: Windows CI may failed in unknown reason. Skip check the value
    is_installed('mmengine')

    # If there is `__init__.py` in the directory which is added into
    # `sys.path`, the directory will be recognized as a package.
    PYTHONPATH = osp.abspath(
        osp.join(osp.dirname(__file__), '..', '..', 'mmengine'))
    sys.path.append(PYTHONPATH)
    assert is_installed('optim')
    sys.path.pop()


def test_get_install_path():
    # TODO: Windows CI may failed in unknown reason. Skip check the value
    get_installed_path('mmengine')

    # get path for package "installed" by setting PYTHONPATH
    PYTHONPATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    PYTHONPATH = osp.abspath(
        osp.join(osp.dirname(__file__), '..', '..', 'mmengine'))
    sys.path.append(PYTHONPATH)
    assert get_installed_path('optim') == osp.join(PYTHONPATH, 'optim')
    sys.path.pop()

    with pytest.raises(pkg_resources.DistributionNotFound):
        get_installed_path('unknown')
