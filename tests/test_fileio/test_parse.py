# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys
from unittest.mock import MagicMock, patch

from mmengine.fileio import (HTTPBackend, PetrelBackend, dict_from_file,
                             list_from_file)

sys.modules['petrel_client'] = MagicMock()
sys.modules['petrel_client.client'] = MagicMock()


def test_list_from_file():
    # get list from disk
    filename = osp.join(
        osp.dirname(osp.dirname(__file__)), 'data/filelist.txt')
    filelist = list_from_file(filename)
    assert filelist == ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']
    filelist = list_from_file(filename, prefix='a/')
    assert filelist == ['a/1.jpg', 'a/2.jpg', 'a/3.jpg', 'a/4.jpg', 'a/5.jpg']
    filelist = list_from_file(filename, offset=2)
    assert filelist == ['3.jpg', '4.jpg', '5.jpg']
    filelist = list_from_file(filename, max_num=2)
    assert filelist == ['1.jpg', '2.jpg']
    filelist = list_from_file(filename, offset=3, max_num=3)
    assert filelist == ['4.jpg', '5.jpg']

    # get list from http
    with patch.object(
            HTTPBackend, 'get_text', return_value='1.jpg\n2.jpg\n3.jpg'):
        filename = 'http://path/of/your/file'
        filelist = list_from_file(filename, backend_args={'backend': 'http'})
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']
        filelist = list_from_file(filename)
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']

    # get list from petrel
    with patch.object(
            PetrelBackend, 'get_text', return_value='1.jpg\n2.jpg\n3.jpg'):
        filename = 'petrel://path/of/your/file'
        filelist = list_from_file(filename, backend_args={'backend': 'petrel'})
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']
        filelist = list_from_file(filename)
        assert filelist == ['1.jpg', '2.jpg', '3.jpg']


def test_dict_from_file():
    # get dict from disk
    filename = osp.join(osp.dirname(osp.dirname(__file__)), 'data/mapping.txt')
    mapping = dict_from_file(filename)
    assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
    mapping = dict_from_file(filename, key_type=int)
    assert mapping == {1: 'cat', 2: ['dog', 'cow'], 3: 'panda'}

    # get dict from http
    with patch.object(
            HTTPBackend, 'get_text', return_value='1 cat\n2 dog cow\n3 panda'):
        filename = 'http://path/of/your/file'
        mapping = dict_from_file(filename, backend_args={'backend': 'http'})
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
        mapping = dict_from_file(filename)
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}

    # get dict from petrel
    with patch.object(
            PetrelBackend, 'get_text',
            return_value='1 cat\n2 dog cow\n3 panda'):
        filename = 'petrel://path/of/your/file'
        mapping = dict_from_file(filename, backend_args={'backend': 'petrel'})
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
        mapping = dict_from_file(filename)
        assert mapping == {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
