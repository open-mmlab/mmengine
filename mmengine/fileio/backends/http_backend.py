# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union
from urllib.request import urlopen

from .base import BaseStorageBackend


class HTTPBackend(BaseStorageBackend):
    """HTTP and HTTPS storage bachend."""

    def get_bytes(self, filepath: str) -> bytes:
        """ead bytes from a given ``filepath``.

        Args:
            filepath (str): Path to read data.

        Returns:
            bytes: Expected bytes object.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get_bytes('http://path/of/file')
            b'hello world'
        """
        return urlopen(filepath).read()

    def get_text(self, filepath, encoding='utf-8') -> str:
        """Read text from a given ``filepath``.

        Args:
            filepath (str): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = HTTPBackend()
            >>> backend.get_text('http://path/of/file')
            'hello world'
        """
        return urlopen(filepath).read().decode(encoding)

    @contextmanager
    def get_local_path(
            self, filepath: str) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath``.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str): Download a file from ``filepath``.

        Examples:
            >>> backend = HTTPBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> with backend.get_local_path('http://path/of/file') as path:
            ...     # do something here
        """
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get_bytes(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)
