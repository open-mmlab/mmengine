# Copyright (c) OpenMMLab. All rights reserved.
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from mmengine.checkpoint.io import save_checkpoint
from mmengine.checkpoint.loader import (CheckpointLoader, load_from_local,
                                        load_from_pavi)
from mmengine.testing import assert_tensor_equal


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.norm = nn.BatchNorm2d(3)


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.block = Block()
        self.conv = nn.Conv2d(3, 3, 1)


class Mockpavimodel:

    def __init__(self, name='fakename'):
        self.name = name

    def download(self, file):
        pass


@patch.dict(sys.modules, {'pavi': MagicMock()})
def test_load_pavimodel_dist():
    pavimodel = Mockpavimodel()
    import pavi
    pavi.modelcloud.get = MagicMock(return_value=pavimodel)
    with pytest.raises(AssertionError):
        # test pavi prefix
        _ = load_from_pavi('MyPaviFolder/checkpoint.pth')

    with pytest.raises(FileNotFoundError):
        # there is not such checkpoint for us to load
        _ = load_from_pavi('pavi://checkpoint.pth')


def test_load_from_local():
    import os
    home_path = os.path.expanduser('~')
    checkpoint_path = os.path.join(
        home_path, 'dummy_checkpoint_used_to_test_load_from_local.pth')
    model = Model()
    save_checkpoint(model.state_dict(), checkpoint_path)
    checkpoint = load_from_local(
        '~/dummy_checkpoint_used_to_test_load_from_local.pth',
        map_location=None)
    assert_tensor_equal(checkpoint['block.conv.weight'],
                        model.block.conv.weight)
    os.remove(checkpoint_path)


@patch.dict(sys.modules, {'petrel_client': MagicMock()})
def test_checkpoint_loader():
    filenames = [
        'http://xx.xx/xx.pth', 'https://xx.xx/xx.pth',
        'modelzoo://xx.xx/xx.pth', 'torchvision://xx.xx/xx.pth',
        'open-mmlab://xx.xx/xx.pth', 'openmmlab://xx.xx/xx.pth',
        'mmcls://xx.xx/xx.pth', 'pavi://xx.xx/xx.pth', 's3://xx.xx/xx.pth',
        'ss3://xx.xx/xx.pth', ' s3://xx.xx/xx.pth',
        'open-mmlab:s3://xx.xx/xx.pth', 'openmmlab:s3://xx.xx/xx.pth',
        'openmmlabs3://xx.xx/xx.pth', ':s3://xx.xx/xx.path'
    ]
    fn_names = [
        'load_from_http', 'load_from_http', 'load_from_torchvision',
        'load_from_torchvision', 'load_from_openmmlab', 'load_from_openmmlab',
        'load_from_mmcls', 'load_from_pavi', 'load_from_ceph',
        'load_from_local', 'load_from_local', 'load_from_ceph',
        'load_from_ceph', 'load_from_local', 'load_from_local'
    ]

    for filename, fn_name in zip(filenames, fn_names):
        loader = CheckpointLoader._get_checkpoint_loader(filename)
        assert loader.__name__ == fn_name

    @CheckpointLoader.register_scheme(prefixes='ftp://')
    def load_from_ftp(filename, map_location):
        return dict(filename=filename)

    # test register_loader
    filename = 'ftp://xx.xx/xx.pth'
    loader = CheckpointLoader._get_checkpoint_loader(filename)
    assert loader.__name__ == 'load_from_ftp'

    def load_from_ftp1(filename, map_location):
        return dict(filename=filename)

    # test duplicate registered error
    with pytest.raises(KeyError):
        CheckpointLoader.register_scheme('ftp://', load_from_ftp1)

    # test force param
    CheckpointLoader.register_scheme('ftp://', load_from_ftp1, force=True)
    checkpoint = CheckpointLoader.load_checkpoint(filename)
    assert checkpoint['filename'] == filename

    # test print function name
    loader = CheckpointLoader._get_checkpoint_loader(filename)
    assert loader.__name__ == 'load_from_ftp1'

    # test sort
    @CheckpointLoader.register_scheme(prefixes='a/b')
    def load_from_ab(filename, map_location):
        return dict(filename=filename)

    @CheckpointLoader.register_scheme(prefixes='a/b/c')
    def load_from_abc(filename, map_location):
        return dict(filename=filename)

    filename = 'a/b/c/d'
    loader = CheckpointLoader._get_checkpoint_loader(filename)
    assert loader.__name__ == 'load_from_abc'
