# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine.fileio.backends import BaseStorageBackend


def test_base_storage_backend():
    # test inheritance
    class ExampleBackend(BaseStorageBackend):
        pass

    with pytest.raises(
            TypeError,
            match="Can't instantiate abstract class ExampleBackend"):
        ExampleBackend()

    class ExampleBackend(BaseStorageBackend):

        def get(self, filepath):
            return filepath

        def get_text(self, filepath):
            return filepath

    backend = ExampleBackend()
    assert backend.get('test') == 'test'
    assert backend.get_text('test') == 'test'
