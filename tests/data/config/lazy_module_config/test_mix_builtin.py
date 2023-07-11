# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from functools import partial
from itertools import chain
from os.path import basename
from os.path import exists as ex
from os.path import splitext

import numpy as np

path = osp.join('a', 'b')
name, suffix = splitext('a/b.py')
chained = list(chain([1, 2], [3, 4]))
existed = ex(__file__)
cfgname = partial(basename, __file__)()

