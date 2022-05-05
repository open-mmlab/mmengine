# Copyright (c) OpenMMLab. All rights reserved.
# config now can have imported modules and defined functions for convenience
import os.path as osp


def func():
    return 'string with \tescape\\ characters\n'


test_item1 = [1, 2]
bool_item2 = True
str_item3 = 'test'
dict_item4 = dict(
    a={
        'c/d': 'path/d',
        'f': 's3//f',
        6: '2333',
        '2333': 'number'
    },
    b={'8': 543},
    c={9: 678},
    d={'a': 0},
    f=dict(a='69'))
dict_item5 = {'x/x': {'a.0': 233}}
dict_list_item6 = {'x/x': [{'a.0': 1., 'b.0': 2.}, {'c/3': 3.}]}
# Test windows path and escape.
str_item_7 = osp.join(osp.expanduser('~'), 'folder') # with backslash in
str_item_8 = func()
