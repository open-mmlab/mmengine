# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmengine.fileio.backends import (BaseStorageBackend, backends,
                                      prefix_to_backends, register_backend)


def test_register_backend():
    # 1. two ways to register backend
    # 1.1 use it as a decorator
    @register_backend('example')
    class ExampleBackend(BaseStorageBackend):

        def get(self, filepath):
            return filepath

        def get_text(self, filepath):
            return filepath

    assert 'example' in backends

    # 1.2 use it as a normal function
    class ExampleBackend1(BaseStorageBackend):

        def get(self, filepath):
            return filepath

        def get_text(self, filepath):
            return filepath

    register_backend('example1', ExampleBackend1)
    assert 'example1' in backends

    # 2. test `name` parameter
    # 2. name should a string
    with pytest.raises(TypeError, match='name should be a string'):
        register_backend(1, ExampleBackend)

    register_backend('example2', ExampleBackend)
    assert 'example2' in backends

    # 3. test `backend` parameter
    # If backend is not None, it should be a class and a subclass of
    # BaseStorageBackend.
    with pytest.raises(TypeError, match='backend should be a class'):

        def test_backend():
            pass

        register_backend('example3', test_backend)

    class ExampleBackend2:

        def get(self, filepath):
            return filepath

        def get_text(self, filepath):
            return filepath

    with pytest.raises(
            TypeError, match='not a subclass of BaseStorageBackend'):
        register_backend('example3', ExampleBackend2)

    # 4. test `force` parameter
    # 4.1 force=False
    with pytest.raises(ValueError, match='example is already registered'):
        register_backend('example', ExampleBackend)

    # 4.2 force=True
    register_backend('example', ExampleBackend, force=True)
    assert 'example' in backends

    # 5. test `prefixes` parameter
    class ExampleBackend3(BaseStorageBackend):

        def get(self, filepath):
            return filepath

        def get_text(self, filepath):
            return filepath

    # 5.1 prefixes is a string
    register_backend('example3', ExampleBackend3, prefixes='prefix1')
    assert 'example3' in backends
    assert 'prefix1' in prefix_to_backends

    # 5.2 prefixes is a list (tuple) of strings
    register_backend(
        'example4', ExampleBackend3, prefixes=['prefix2', 'prefix3'])
    assert 'example4' in backends
    assert 'prefix2' in prefix_to_backends
    assert 'prefix3' in prefix_to_backends
    assert prefix_to_backends['prefix2'] == prefix_to_backends['prefix3']

    # 5.3 prefixes is an invalid type
    with pytest.raises(AssertionError):
        register_backend('example5', ExampleBackend3, prefixes=1)

    # 5.4 prefixes is already registered
    with pytest.raises(ValueError, match='prefix2 is already registered'):
        register_backend('example6', ExampleBackend3, prefixes='prefix2')

    class ExampleBackend4(BaseStorageBackend):

        def get(self, filepath):
            return filepath

        def get_text(self, filepath):
            return filepath

    register_backend(
        'example6', ExampleBackend4, prefixes='prefix2', force=True)
    assert 'example6' in backends
    assert 'prefix2' in prefix_to_backends
