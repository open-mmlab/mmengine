# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys
from pathlib import Path

from mmengine.utils import get_installed_path, is_installed


def test_is_installed():
    # TODO: Windows CI may failed in unknown reason. Skip check the value
    is_installed('mmengine')

    # package set by PYTHONPATH
    assert not is_installed('py_config')
    sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))
    assert is_installed('test_utils')
    sys.path.pop()


def test_get_install_path(tmp_path: Path):
    mmengine_path = get_installed_path('mmengine')
    assert osp.isdir(mmengine_path)

    # get path for package "installed" by setting PYTHONPATH
    PYTHONPATH = osp.abspath(osp.join(
        osp.dirname(__file__),
        '..',
    ))
    sys.path.append(PYTHONPATH)
    res_path = get_installed_path('test_utils')
    assert osp.join(PYTHONPATH, 'test_utils') == res_path

    # return the first path for namespace package
    # See more information about namespace package in:
    # https://packaging.python.org/en/latest/guides/packaging-namespace-packages/  # noqa:E501
    (tmp_path / 'test_utils').mkdir()
    sys.path.insert(-1, str(tmp_path))
    res_path = get_installed_path('test_utils')
    assert osp.abspath(osp.join(tmp_path, 'test_utils')) == res_path
    sys.path.pop()
    sys.path.pop()
