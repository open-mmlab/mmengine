# Copyright (c) OpenMMLab. All rights reserved.
import os
from importlib.util import find_spec as find_module

import numpy
import numpy.compat
import numpy.linalg as linalg

from mmengine.config import Config
from mmengine.fileio import LocalBackend as local
from mmengine.fileio import PetrelBackend
from ._base_.default_runtime import default_scope as scope
from ._base_.scheduler import val_cfg
from rich.progress import Progress
start = Progress.start
