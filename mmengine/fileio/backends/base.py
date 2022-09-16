# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: :meth:`get()` and
    :meth:`get_text()`.

    - :meth:`get()` reads the file as a byte stream.
    - :meth:`get_text()` reads the file as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    # This attribute will be deprecated in future.
    _allow_symlink = False

    @property
    def allow_symlink(self):
        warnings.warn('allow_symlink will be deprecated in future',
                      DeprecationWarning)
        return self._allow_symlink

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass
