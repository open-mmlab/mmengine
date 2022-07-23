# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO, StringIO
from pathlib import Path

from mmengine.utils import is_str
from .handlers import file_handlers
from .io import get_bytes, get_text, put_bytes, put_text


def load(file, file_format=None, backend_args=None, **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    ``load`` supports loading data from serialized files those can be storaged
    in different backends.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml" and
            "pickle/pkl".
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Examples:
        >>> load('/path/of/your/file')  # file is storaged in disk
        >>> load('https://path/of/your/file')  # file is storaged in Internet
        >>> load('s3://path/of/your/file')  # file is storaged in petrel

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and is_str(file):
        file_format = file.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    handler = file_handlers[file_format]
    if is_str(file):
        if handler.str_like:
            content = get_text(file, backend_args=backend_args)
            with StringIO(content) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
        else:
            content = get_bytes(file, backend_args=backend_args)
            with BytesIO(content) as f:
                obj = handler.load_from_fileobj(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load_from_fileobj(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, backend_args=None, **kwargs):
    """Dump data to json/yaml/pickle strings or files.

    This method provides a unified api for dumping data as strings or to files,
    and also supports custom arguments for each file format.

    ``dump`` supports dumping data as strings or to files which is saved to
    different backends.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk
        >>> dump('hello world', 's3://path/of/your/file')  # ceph or petrel

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if is_str(file):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if file_format not in file_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    handler = file_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif is_str(file):
        if handler.str_like:
            with StringIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                put_text(f.getvalue(), file, backend_args=backend_args)
        else:
            with BytesIO() as f:
                handler.dump_to_fileobj(obj, f, **kwargs)
                put_bytes(f.getvalue(), file, backend_args=backend_args)
    elif hasattr(file, 'write'):
        handler.dump_to_fileobj(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def list_from_file(filename,
                   prefix='',
                   offset=0,
                   max_num=0,
                   encoding='utf-8',
                   backend_args=None):
    """Load a text file and parse the content as a list of strings.

    ``list_from_file`` supports loading a text file which can be storaged in
    different backends and parsing the content as a list for strings.

    Args:
        filename (str): Filename.
        prefix (str): The prefix to be inserted to the beginning of each item.
        offset (int): The offset of lines.
        max_num (int): The maximum number of lines to be read,
            zeros and negatives mean no limitation.
        encoding (str): Encoding used to open the file. Default utf-8.
        backend_args (dict, optional): Arguments to instantiate the
            preifx of uri corresponding backend. Defaults to None.

    Examples:
        >>> list_from_file('/path/of/your/file')  # disk
        ['hello', 'world']
        >>> list_from_file('s3://path/of/your/file')  # ceph or petrel
        ['hello', 'world']

    Returns:
        list[str]: A list of strings.
    """
    cnt = 0
    item_list = []
    content = get_text(filename, encoding, backend_args=backend_args)
    with StringIO(content) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list


def dict_from_file(filename,
                   key_type=str,
                   encoding='utf-8',
                   backend_args=None):
    """Load a text file and parse the content as a dict.

    Each line of the text file will be two or more columns split by
    whitespaces or tabs. The first column will be parsed as dict keys, and
    the following columns will be parsed as dict values.

    ``dict_from_file`` supports loading a text file which can be storaged in
    different backends and parsing the content as a dict.

    Args:
        filename(str): Filename.
        key_type(type): Type of the dict keys. str is user by default and
            type conversion will be performed if specified.
        encoding (str): Encoding used to open the file. Default utf-8.
        backend_args (dict, optional): Arguments to instantiate the preifx of
            uri corresponding backend. Defaults to None.

    Examples:
        >>> dict_from_file('/path/of/your/file')  # disk
        {'key1': 'value1', 'key2': 'value2'}
        >>> dict_from_file('s3://path/of/your/file')  # ceph or petrel
        {'key1': 'value1', 'key2': 'value2'}

    Returns:
        dict: The parsed contents.
    """
    mapping = {}
    content = get_text(filename, encoding, backend_args=backend_args)
    with StringIO(content) as f:
        for line in f:
            items = line.rstrip('\n').split()
            assert len(items) >= 2
            key = key_type(items[0])
            val = items[1:] if len(items) > 2 else items[1]
            mapping[key] = val
    return mapping
