# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import pytest
import torch

import mmengine.dist as dist
from mmengine.dist.dist import sync_random_seed


def test_all_reduce():
    data = torch.arange(2, dtype=torch.int64)
    expected = torch.arange(2, dtype=torch.int64)
    dist.all_reduce(data)
    torch.allclose(data, expected)


def test_all_gather():
    data = torch.arange(2, dtype=torch.int64)
    expected = torch.arange(2, dtype=torch.int64)
    dist.all_gather(data)
    torch.allclose(data[0], expected)


def test_gather():
    data = torch.arange(2, dtype=torch.int64)
    expected = torch.arange(2, dtype=torch.int64)
    dist.gather(data)
    torch.allclose(data[0], expected)


def test_broadcast():
    data = torch.arange(2, dtype=torch.int64)
    expected = torch.arange(2, dtype=torch.int64)
    dist.broadcast(data)
    torch.allclose(data, expected)


@patch('numpy.random.randint', return_value=10)
def test_sync_random_seed(mock):
    assert sync_random_seed() == 10


def test_broadcast_object_list(self):
    with pytest.raises(AssertionError):
        # input should be list of object
        dist.broadcast_object_list('foo')

    data = ['foo', 12, {1: 2}]
    expected = ['foo', 12, {1: 2}]
    dist.broadcast_object_list(data)
    assert data == expected


def test_all_reduce_dict():
    with pytest.raises(AssertionError):
        # input should be dict
        dist.all_reduce_dict('foo')

    data = {
        'key1': torch.arange(2, dtype=torch.int64),
        'key2': torch.arange(3, dtype=torch.int64)
    }
    expected = {
        'key1': torch.arange(2, dtype=torch.int64),
        'key2': torch.arange(3, dtype=torch.int64)
    }
    dist.all_reduce_dict(data)
    for key in data:
        torch.allclose(data[key], expected[key])


def test_all_gather_object():
    data = 'foo'
    expected = 'foo'
    gather_objects = dist.all_gather_object(data)
    assert gather_objects[0] == expected


def test_gather_object():
    data = 'foo'
    expected = 'foo'
    gather_objects = dist.gather_object(data)
    assert gather_objects[0] == expected
