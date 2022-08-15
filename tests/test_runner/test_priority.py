# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine.runner import Priority, get_priority


def test_get_priority():
    # test `priority` parameter which can be int, str or Priority
    # `priority` is an integer
    assert get_priority(10) == 10
    # `priority` is an integer but it exceeds the valid ranges
    with pytest.raises(ValueError, match='priority must be between 0 and 100'):
        get_priority(-1)
    with pytest.raises(ValueError, match='priority must be between 0 and 100'):
        get_priority(101)

    # `priority` is a Priority enum value
    assert get_priority(Priority.HIGHEST) == 0
    assert get_priority(Priority.LOWEST) == 100

    # `priority` is a string
    assert get_priority('HIGHEST') == 0
    assert get_priority('LOWEST') == 100

    # `priority` is an invalid type
    with pytest.raises(
            TypeError,
            match='priority must be an integer or Priority enum value'):
        get_priority([10])
