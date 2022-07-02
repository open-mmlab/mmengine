# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: :meth:`get_bytes()` and
    :meth:`get_text()`.

    - :meth:`get_bytes()` reads the file as a byte stream.
    - :meth:`get_text()` reads the file as texts.
    """

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def get_bytes(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass
