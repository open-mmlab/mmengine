# Copyright (c) OpenMMLab. All rights reserved.
import os.path

import pytest

from mmengine.config.collect_meta import (_get_external_cfg_base_path,
                                          _parse_external_cfg_path,
                                          _parse_rel_cfg_path)


def test_get_external_cfg_base_path(tmp_path):
    package_path = tmp_path
    rel_cfg_path = 'cfg_dir/cfg_file'
    with pytest.raises(FileNotFoundError):
        _get_external_cfg_base_path(str(package_path), rel_cfg_path)
    cfg_dir = tmp_path / '.mmengine' / 'configs' / 'cfg_dir'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    f = open(cfg_dir / 'cfg_file', 'w')
    f.close()
    cfg_path = _get_external_cfg_base_path(str(package_path), rel_cfg_path)
    assert cfg_path == f'{os.path.join(str(cfg_dir), "cfg_file")}'


def test_parse_external_cfg_path():
    external_cfg_path = 'package::path/cfg'
    package, rel_cfg_path = _parse_external_cfg_path(external_cfg_path)
    assert package == 'package'
    assert rel_cfg_path == 'path/cfg'
    # external config must contain `::`.
    external_cfg_path = 'path/cfg'
    with pytest.raises(ValueError):
        _parse_external_cfg_path(external_cfg_path)
    # Use `:::` as operator will raise an error.
    external_cfg_path = 'package:::path/cfg'
    with pytest.raises(ValueError):
        _parse_external_cfg_path(external_cfg_path)
    # Use `:` as operator will raise an error.
    external_cfg_path = 'package:path/cfg'
    with pytest.raises(ValueError):
        _parse_external_cfg_path(external_cfg_path)
    # Too much `::`
    external_cfg_path = 'mmdet::path/cfg::error'
    with pytest.raises(ValueError):
        _parse_external_cfg_path(external_cfg_path)


def test_parse_rel_cfg_path():
    rel_cfg_path = 'cfg_dir/cfg_file'
    rel_cfg_dir, rel_cfg_file = _parse_rel_cfg_path(rel_cfg_path)
    assert rel_cfg_dir == 'cfg_dir'
    assert rel_cfg_file == 'cfg_file'
    with pytest.raises(AssertionError):
        _parse_rel_cfg_path('error/cfg_dir/cfg_file')
