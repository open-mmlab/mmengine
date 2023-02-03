# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from shutil import SameFileError
from typing import Generator, Iterator, Optional, Tuple, Union

import mmengine
from .base import BaseStorageBackend


class OSSBackend(BaseStorageBackend):
    """OSS storage backend for reading (writing) from (to) OSS.

    OSSBackend supports reading and writing data to multiple buckets
    from Endpoint. If the file path contains the bucket name,
    OSSBackend will read data from the specified buckets
    or write data to them. Otherwise, OSSBackend will occur an error.

    Please refer to https://help.aliyun.com/document_detail/85288.html
    to install the required packages of OSSBackend.

    Args:
        access_key_id (str, optional): AccessKey ID of your OSS.
        access_key_secret (str, optional): AccessKey Secret of your OSS.
        path_mapping (dict, optional): Path mapping dict from local path to
            ``filepath`` will be replaced by ``dst``. Defaults to None.

    Examples:
        >>> backend = OSSBackend()
        >>> filepath1 = 'oss://endpoint/bucket/file'
        >>> backend.get(filepath1)  # get data from default cluster
    """

    def __init__(self,
                 access_key_id: Optional[str] = None,
                 access_key_secret: Optional[str] = None,
                 path_mapping: Optional[dict] = None):
        try:
            import oss2
            self.oss2 = oss2
        except ImportError:
            raise ImportError('Please install OSS_client to enable '
                              'OSSBackend.')
        self._client = self.oss2.Auth(
            access_key_id=access_key_id, access_key_secret=access_key_secret)

        assert isinstance(path_mapping, dict) or path_mapping is None
        self.path_mapping = path_mapping
        # Use to parse bucket and obj_name
        self.parse_bucket = re.compile('oss://(.+?)/(.+?)/(.*)')

    def _map_path(self, filepath: Union[str, Path]) -> str:
        """Map ``filepath`` to a string path whose prefix will be replaced by
        :attr:`self.path_mapping`.

        Args:
            filepath (str or Path): Path to be mapped.
        """
        filepath = str(filepath)
        if self.path_mapping is not None:
            for k, v in self.path_mapping.items():
                filepath = filepath.replace(k, v, 1)
        return filepath

    def _format_path(self, filepath: str) -> str:
        """Convert a ``filepath`` to standard format of OSS oss.

        If the ``filepath`` is concatenated by ``os.path.join``, in a Windows
        environment, the ``filepath`` will be the format of
        'oss://bucket_name\\image.jpg'. By invoking :meth:`_format_path`, the
        above ``filepath`` will be converted to 'oss://bucket_name/image.jpg'.

        Args:
            filepath (str): Path to be formatted.
        """
        return re.sub(r'\\+', '/', filepath)

    def _parse_path(self, filepath: Union[str, Path]) -> Tuple[str, str, str]:
        """Parse endpoint, bucket, and object name from a given ``filepath``.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            tuple[str, str, str]: The first item is the endpoint meaning
            access domain or CNAME. The second is the bucket name
            of oss. The last is obj_name which is the object's relative
            path to the bucket.
        """
        filepath = self._map_path(filepath)
        filepath = self._format_path(filepath)
        parse_res = self.parse_bucket.findall(filepath)
        if not parse_res:
            raise ValueError(
                f"The input path '{filepath}' format is incorrect. "
                'An example: oss://oss-cn-hangzhou.aliyuncs.com/mmengine/.*')
        endpoint, bucket, obj_name = parse_res[0]
        return endpoint, bucket, obj_name

    def _bucket_instance(self, endpoint: str, bucket_name: str):
        """get bucket instance from a given ``endpoint`` and ``bucket_name``.

        Args:
            endpoint (str): aliyun Endpoint.
            bucket_name (str): bucket name of OSS

        Returns:
            Bucket: An instance of OSS.
        """

        bucket = self.oss2.Bucket(self._client, endpoint, bucket_name)
        return bucket

    def get(self, filepath: Union[str, Path]) -> bytes:
        """Read bytes from a given ``filepath`` with 'rb' mode.

        Args:
            filepath (str or Path): Path to read data.

        Returns:
            bytes: Return bytes read from filepath.

        Examples:
            >>> backend = OSSBackend()
            >>> filepath = 'oss://endpoint/bucket/file'
            >>> backend.get(filepath)
            b'hello world'
        """
        endpoint, bucket_name, obj_name = self._parse_path(filepath)
        bucket = self._bucket_instance(endpoint, bucket_name)
        object_stream = bucket.get_object(obj_name)
        return object_stream.read()

    def get_text(
        self,
        filepath: Union[str, Path],
        encoding: str = 'utf-8',
    ) -> str:
        """Read text from a given ``filepath`` with 'r' mode.

        Args:
            filepath (str or Path): Path to read data.
            encoding (str): The encoding format used to open the ``filepath``.
                Defaults to 'utf-8'.

        Returns:
            str: Expected text reading from ``filepath``.

        Examples:
            >>> backend = OSSBackend()
            >>> filepath = 'oss://endpoint/bucket/file'
            >>> backend.get_text(filepath)
            'hello world'
        """
        return str(self.get(filepath), encoding=encoding)

    def put(self, obj: bytes, filepath: Union[str, Path]) -> None:
        """Write bytes to a given ``filepath``.

        Args:
            obj (bytes): Data to be saved.
            filepath (str or Path): Path to write data.

        Examples:
            >>> backend = OSSBackend()
            >>> filepath = 'oss://endpoint/bucket/file'
            >>> backend.put(b'hello world', filepath)
        """
        endpoint, bucket_name, obj_name = self._parse_path(filepath)
        bucket = self._bucket_instance(endpoint, bucket_name)
        bucket.put_object(obj_name, obj)

    def put_text(
        self,
        obj: str,
        filepath: Union[str, Path],
        encoding: str = 'utf-8',
    ) -> None:
        """Write text to a given ``filepath``.

        Args:
            obj (str): Data to be written.
            filepath (str or Path): Path to write data.
            encoding (str): The encoding format used to encode the ``obj``.
                Defaults to 'utf-8'.

        Examples:
            >>> backend = OSSBackend()
            >>> filepath = 'oss://endpoint/bucket/file'
            >>> backend.put_text('hello world', filepath)
        """
        self.put(bytes(obj, encoding=encoding), filepath)

    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.

        Args:
            filepath (str or Path): Path to be checked whether exists.

        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.

        Examples:
            >>> backend = OSSBackend()
            >>> filepath = 'oss://endpoint/bucket/file'
            >>> backend.exists(filepath)
            True
        """

        endpoint, bucket_name, obj_name = self._parse_path(filepath)
        bucket = self._bucket_instance(endpoint, bucket_name)
        return bucket.object_exists(obj_name)

    def isdir(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a directory.
        note: Current version of oss2 Python SDK has not supported
        to judge whether a directory exists, latter version may support.


        Args:
            filepath (str or Path): Path to be checked
            whether it is a directory.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a directory,
            ``False`` otherwise.

        Examples:
            >>> backend = OSSBackend()
            >>> filepath = 'oss://endpoint/bucket/file/
            >>> backend.isdir(filepath)
            True
        """

        raise NotImplementedError(
            'Current version of oss2 Python SDK has not supported'
            'the `isdir` method, it may add latter.')
        return self.exists(filepath)

    def isfile(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path is a file.

        Args:
            filepath (str or Path): Path to be checked whether it is a file.

        Returns:
            bool: Return ``True`` if ``filepath`` points to a file, ``False``
            otherwise.

        Examples:
            >>> backend = OSSBackend()
            >>> filepath = 'oss://endpoint/bucket/file'
            >>> backend.isfile(filepath)
            True
        """

        return self.exists(filepath)

    def join_path(
        self,
        filepath: Union[str, Path],
        *filepaths: Union[str, Path],
    ) -> str:
        """Concatenate all file paths.

        Join one or more filepath components intelligently. The return value
        is the concatenation of filepath and any members of *filepaths.

        Args:
            filepath (str or Path): Path to be concatenated.

        Returns:
            str: The result after concatenation.

        Examples:
            >>> backend = OSSBackend()
            >>> filepath = 'oss://endpoint/bucket/dir'
            >>> backend.join_path(filepath, 'another/path')
            'oss://endpoint/bucket/dir/another/path'
        """
        filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith('/'):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_path = self._format_path(self._map_path(path))
            formatted_paths.append(formatted_path.lstrip('/'))

        return '/'.join(formatted_paths)

    @contextmanager
    def get_loacl_path(
        self,
        filepath: Union[str, Path],
    ) -> Generator[Union[str, Path], None, None]:
        """Download a file from ``filepath`` to a local temporary directory,
        and return the temporary path.

        ``get_local_path`` is decorated by :meth:`contxtlib.contextmanager`. It
        can be called with ``with`` statement, and when exists from the
        ``with`` statement, the temporary path will be released.

        Args:
            filepath (str or Path): Download a file from ``filepath``.

        Yields:
            Iterable[str]: Only yield one temporary path.

        Examples:
            >>> backend = OSSBackend()
            >>> # After existing from the ``with`` clause,
            >>> # the path will be removed
            >>> filepath = 'oss://endpoint/bucket/file'
            >>> with backend.get_local_path(filepath) as path:
            ...     # do something here
        """
        assert self.isfile(filepath)
        try:
            f = tempfile.NamedTemporaryFile(delete=False)
            f.write(self.get(filepath))
            f.close()
            yield f.name
        finally:
            os.remove(f.name)

    def copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy a file src to dst and return the destination file.

        Note: Currently, only file copy is supported.
        src and dst must be have the same endpoint.

        The files (src) in the source buckets are copied to the same or
        different target (dst) buckets in the same region.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = OSSBackend()
            >>> # copy in the same bucket.
            >>> src = 'oss://endpoint/bucket/file'
            >>> dst = 'oss://endpoint/bucket/file1'
            >>> backend.copyfile(src, dst)
            'oss://endpoint/bucket/file1'

            >>> backend = OSSBackend()
            >>> # copy in different bucket.
            >>> src = 'oss://endpoint/bucket/file'
            >>> dst = 'oss://endpoint/bucket1/file1'
            >>> backend.copyfile(src, dst)
            'oss://endpoint/bucket1/file1'
        """
        src_endpoint, src_bucket_name, src_obj_name = self._parse_path(src)
        dst_endpoint, dst_bucket_name, dst_obj_name = self._parse_path(dst)

        assert (dst_endpoint == src_endpoint
                ), 'src and dst must be have the same endpoint.'
        if src == dst:
            raise SameFileError('src and dst should not be same')
        bucket = self._bucket_instance(src_endpoint, dst_bucket_name)
        result = bucket.copy_object(src_bucket_name, src_obj_name,
                                    dst_obj_name)

        assert (result.status == 200
                ), 'copy failed, please check your network status.'
        return str(dst)

    def party_copyfile(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Copy files larger than 1GB from src to dst shard.

        Note: Currently, only file copy is supported.
        src and dst must be have the same endpoint.

        Copy the files (src) in the source Bucket to the same
        or different target (dst) buckets in the same region.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: The destination file.

        Raises:
            SameFileError: If src and dst are the same file, a SameFileError
                will be raised.

        Examples:
            >>> backend = OSSBackend()
            >>> # copy in the same bucket.
            >>> src = 'oss://endpoint/bucket/file'
            >>> dst = 'oss://endpoint/bucket/file1'
            >>> backend.copyfile(src, dst)
            'oss://endpoint/bucket/file1'

            >>> backend = OSSBackend()
            >>> # copy in different bucket.
            >>> src = 'oss://endpoint/bucket/file'
            >>> dst = 'oss://endpoint/bucket1/file1'
            >>> backend.copyfile(src, dst)
            'oss://endpoint/bucket1/file1'
        """
        src_endpoint, src_bucket_name, src_obj_name = self._parse_path(src)
        dst_endpoint, dst_bucket_name, dst_obj_name = self._parse_path(dst)

        assert (dst_endpoint == src_endpoint
                ), 'src and dst must be have the same endpoint.'
        if src == dst:
            raise SameFileError('src and dst should not be same.')

        src_bucket = self._bucket_instance(src_endpoint, src_bucket_name)
        # get src file size
        total_size = src_bucket.head_object(src_obj_name).content_length
        assert (
            total_size >= 100 * 1024
        ), f'{total_size} >= 100*1024 small file can use copyfile operator.'

        # init shard, get shard size
        from oss2 import determine_part_size
        from oss2.models import PartInfo
        part_size = determine_part_size(total_size, preferred_size=100 * 1024)

        if src_bucket_name == dst_bucket_name:
            dst_bucket = src_bucket
        else:
            dst_bucket = self._bucket_instance(dst_endpoint, dst_bucket_name)

        upload_id = dst_bucket.init_multipart_upload(dst_obj_name).upload_id
        parts = []
        part_number = 1
        offset = 0
        while offset < total_size:
            num_to_upload = min(part_size, total_size - offset)
            end = offset + num_to_upload - 1
            result = dst_bucket.upload_part_copy(src_bucket_name, src_obj_name,
                                                 (offset, end), dst_obj_name,
                                                 upload_id, part_number)
            # save part info
            parts.append(PartInfo(part_number, result.etag))
            offset += num_to_upload
            part_number += 1
        # part copy success
        result = dst_bucket.complete_multipart_upload(dst_obj_name, upload_id,
                                                      parts)

        assert (result.status == 200
                ), f'copy failed, status with {result.status}, '
        'please check your network status.'
        # get dst file meta information
        head_info = dst_bucket.head_object(dst_obj_name)
        assert (head_info.content_length == total_size
                ), 'copying file occurs error, '
        f'dst file size {head_info.content_length}'
        f'does not equals src file size {total_size}, please recopy.'
        return str(dst)

    def list_dir_or_file(self,
                         dir_path: Union[str, Path],
                         list_dir: bool = True,
                         list_file: bool = True,
                         suffix: Optional[Union[str, Tuple[str]]] = None,
                         recursive: bool = False) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.

        Note:
            OSS has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.

        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will contains the
            suffix '/' which is inconsistent with other backends.

        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Defaults to True.
            list_file (bool): List the path of files. Defaults to True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in, such as .txt, .jpg etc.
                Defaults to None.
            recursive (bool): If set to True, recursively scan the
                directory. Defaults to False.

        Yields:
            Iterable[str]: A relative path to ``dir_path``.

        Examples:
            >>> backend = OSSBackend()
            >>> dir_path = 'oss://endpoint/bucket/file/'
            >>> # list those files and directories in current directory
            >>> for file_path in backend.list_dir_or_file(dir_path):
            ...     print(file_path)
            >>> # only list files
            >>> for file_path in backend.list_dir_or_file(dir_path,
                list_dir=False):
            ...     print(file_path)
            >>> # only list directories
            >>> for file_path in backend.list_dir_or_file(dir_path,
                list_file=False):
            ...     print(file_path)
            >>> # only list files ending with specified suffixes
            >>> for file_path in backend.list_dir_or_file(dir_path,
                suffix='.txt'):
            ...     print(file_path)
        """

        endpoint, bucket_name, obj_name = self._parse_path(dir_path)
        bucket = self._bucket_instance(endpoint, bucket_name)
        if list_dir and suffix is not None:
            raise TypeError(
                '`list_dir` should be False when `suffix` is not None')

        root_path = obj_name
        if root_path and not root_path.endswith('/'):
            raise TypeError('`dir_path` must endswith "/" ')

        def __list_dir_or_file(bucket, obj_name, list_dir, list_file, suffix,
                               recursive):
            for obj in self.oss2.ObjectIterator(
                    bucket, prefix=obj_name, delimiter='/'):
                # judge if directory or not by function
                # is_prefix and drop prefix
                full_path = str(obj.key)
                dir_or_file_path = full_path[len(root_path):]
                # drop prefix
                if not dir_or_file_path:
                    continue

                if obj.is_prefix():  # is dir
                    if list_dir:
                        yield dir_or_file_path
                    if recursive:
                        yield from __list_dir_or_file(bucket, full_path,
                                                      list_dir, list_file,
                                                      suffix, recursive)
                else:
                    if list_file and not dir_or_file_path.endswith('/'):
                        if suffix:
                            if isinstance(
                                    suffix,
                                    str) and dir_or_file_path.endswith(suffix):
                                yield dir_or_file_path
                            elif isinstance(suffix, tuple):
                                _suffix = '.' + dir_or_file_path.split('.')[-1]
                                if _suffix in suffix:
                                    yield dir_or_file_path
                        else:
                            yield dir_or_file_path

        return __list_dir_or_file(bucket, root_path, list_dir, list_file,
                                  suffix, recursive)

    def copyfile_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Upload a local file src to dst and return the destination file.

        Args:
            src (str or Path): A local file to be copied.
            dst (str or Path): Copy file to dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = OSSBackend()
            >>> # dst is a file
            >>> src = 'path/of/your/file'
            >>> dst = 'oss://endpoint/bucket/file1'
            >>> backend.copyfile_from_local(src, dst)
            'oss://endpoint/bucket/file1'

            >>> # dst is a directory
            >>> dst = 'oss://endpoint/bucket/dir/'
            >>> backend.copyfile_from_local(src, dst)
            'oss://endpoint/bucket/dir/file'
        """
        dst = self._format_path(self._map_path(dst))
        src = self._format_path(self._map_path(src))
        if dst.endswith('/'):
            dst = os.path.join(dst, src.split('/')[-1])
        with open(src, 'rb') as f:
            self.put(f.read(), dst)
        return dst

    def copytree_from_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> str:
        """Recursively copy an entire directory tree rooted at src to a
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A local directory to be copied.
            dst (str or Path): Copy directory to dst.

        Returns:
            str: The destination directory.

        Raises:
            FileExistsError: If dst had already existed, a FileExistsError will
                be raised.

        Examples:
            >>> backend = OSSBackend()
            >>> src = 'path/of/your/dir'
            >>> dst = 'oss://endpoint/bucket/dir1'
            >>> backend.copytree_from_local(src, dst)
            'oss://endpoint/bucket/dir1'
        """
        dst = self._format_path(self._map_path(dst))
        if self.exists(dst):
            raise FileExistsError('dst should not exist')

        src = str(src)

        for cur_dir, _, files in os.walk(src):
            for f in files:
                src_path = osp.join(cur_dir, f)
                dst_path = self.join_path(dst, src_path.replace(src, ''))
                self.copyfile_from_local(src_path, dst_path)

        return dst

    def copyfile_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> Union[str, Path]:
        """Copy the file src to local dst and return the destination file.

        If dst specifies a directory, the file will be copied into dst using
        the base filename from src. If dst specifies a file that already
        exists, it will be replaced.

        Args:
            src (str or Path): A file to be copied.
            dst (str or Path): Copy file to to local dst.

        Returns:
            str: If dst specifies a directory, the file will be copied into dst
            using the base filename from src.

        Examples:
            >>> backend = OSSBackend()
            >>> # dst is a file
            >>> src = 'oss://endpoint/bucket/file'
            >>> dst = 'path/of/your/file'
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/file'

            >>> # dst is a directory
            >>> dst = 'path/of/your/dir'
            >>> backend.copyfile_to_local(src, dst)
            'path/of/your/dir/file'
        """
        if osp.isdir(dst):
            basename = osp.basename(src)
            if isinstance(dst, str):
                dst = osp.join(dst, basename)
            else:
                assert isinstance(dst, Path)
                dst = dst / basename

        with open(dst, 'wb') as f:
            f.write(self.get(src))

        return dst

    def copytree_to_local(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> Union[str, Path]:
        """Recursively copy an entire directory tree rooted at src to a local
        directory named dst and return the destination directory.

        Args:
            src (str or Path): A directory to be copied.Specifically,
            src is endswith "/"
            dst (str or Path): Copy directory to local dst.

        Returns:
            str: The destination directory.

        Examples:
            >>> backend = OSSBackend()
            >>> src = 'oss://endpoint/bucket/file/'
            >>> dst = 'path/of/your/dir'
            >>> backend.copytree_to_local(src, dst)
            'path/of/your/dir'
        """
        for path in self.list_dir_or_file(src, list_dir=False, recursive=True):
            dst_path = osp.join(dst, path)
            mmengine.mkdir_or_exist(osp.dirname(dst_path))
            with open(dst_path, 'wb') as f:
                f.write(self.get(self.join_path(src, path)))
        return dst

    def remove(self, filepath: Union[str, Path]) -> None:
        """Remove a file.

        Args:
            filepath (str or Path): Path to be removed.

        Raises:
            FileNotFoundError: If filepath does not exist, an FileNotFoundError
                will be raised.
            IsADirectoryError: If filepath is a directory, an IsADirectoryError
                will be raised.

        Examples:
            >>> backend = OSSBackend()
            >>> filepath = 'oss://endpoint/bucket/file'
            >>> backend.remove(filepath)
        """
        if not self.exists(filepath):
            raise FileNotFoundError(f'filepath {filepath} does not exist')
        assert (str(filepath)[-1] !=
                '/'), 'only delete a file by remove function.'
        endpoint, bucket_name, obj_name = self._parse_path(filepath)
        bucket = self._bucket_instance(endpoint, bucket_name)
        bucket.delete_object(obj_name)

    def rmtree(self, dir_path: Union[str, Path]) -> None:
        """Recursively delete a directory tree.

        Note:
            If dir_path ends with '/', this operator will
            delete all files and directories under dir_path. otherwise, delete
            all files and directories including dir_path itself.

        Args:
            dir_path (str or Path): A directory to be removed.

        Examples:
            >>> backend = OSSBackend()
            >>> dir_path = 'oss://endpoint/bucket/src'
            >>> backend.rmtree(dir_path)
        """
        endpoint, bucket_name, obj_name = self._parse_path(dir_path)
        bucket = self._bucket_instance(endpoint, bucket_name)
        for obj in self.oss2.ObjectIterator(bucket, prefix=obj_name):
            bucket.delete_object(obj.key)

    def symlink(
        self,
        src: Union[str, Path],
        dst: Union[str, Path],
    ) -> None:
        """Create a symbolic link pointing to src file named dst file.
        Note: src and dst must be have the same endpoint and bucket.

        Args:
            src (str or Path): A file to be linked.
            dst (str or Path): address linked.

        Returns:
            bool: Return True if link successfully. otherwise False

        Examples:
            >>> backend = OSSBackend()
            >>> src = 'oss://endpoint/bucket/file'
            >>> dst = 'oss://endpoint/bucket/file2'
            >>> backend.symlink(src, dst)
            True
        """
        if not self.exists(src):
            raise FileNotFoundError(f'filepath {src} does not exist')
        endpoint, bucket_name, obj_name = self._parse_path(src)
        dst_endpoint, dst_bucket_name, dst_obj_name = self._parse_path(dst)
        assert (
            dst_endpoint == endpoint
        ), f'dst_endpoint {dst_endpoint} is not equal src_endpoint {endpoint},'
        'illegal operator.'
        assert (dst_bucket_name == bucket_name
                ), f'dst_bucket_name {dst_bucket_name} '
        f'is not equal src_bucket_name {bucket_name},'
        ' illegal operator.'

        bucket = self._bucket_instance(endpoint, bucket_name)
        bucket.put_symlink(obj_name, dst_obj_name)
