# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from itertools import chain
from os.path import splitext

import numpy as np
from torch.nn import Linear

path = osp.join('a', 'b')
name, suffix = splitext('a/b.py')
chained = list(chain([1, 2], [3, 4]))
