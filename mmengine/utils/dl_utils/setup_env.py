# Copyright (c) OpenMMLab. All rights reserved.
import os
import platform
import warnings
from typing import Optional

import torch.multiprocessing as mp


def set_multi_processing(mp_start_method: str = 'fork',
                         opencv_num_threads: int = 0,
                         omp_num_threads: Optional[int] = None,
                         mkl_num_threads: Optional[int] = None,
                         distributed: bool = False) -> None:
    """Set multi-processing related environment.

    Args:
        mp_start_method (str): Set the method which should be used to start
            child processes. Defaults to 'fork'.
        opencv_num_threads (int): Number of threads for opencv.
            Defaults to 0.
        omp_num_threads (int, optional): Number of threads for OMP.
            Defaults to 1 when distributed is True, otherwise no change.
        mkl_num_threads (int, optional): Number of threads for MKL.
            Defaults to 1 when distributed is True, otherwise no change.
        distributed (bool): True if distributed environment.
            Defaults to False.
    """
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        current_method = mp.get_start_method(allow_none=True)
        if (current_method is not None and current_method != mp_start_method):
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`. You can '
                'change this behavior by changing `mp_start_method` in '
                'your config.')
        mp.set_start_method(mp_start_method, force=True)

    try:
        import cv2

        # disable opencv multithreading to avoid system being overloaded
        cv2.setNumThreads(opencv_num_threads)
    except ImportError:
        pass

    if distributed:
        # setup OMP threads
        # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
        if omp_num_threads is None and 'OMP_NUM_THREADS' not in os.environ:
            omp_num_threads = 1
            warnings.warn(
                'Setting OMP_NUM_THREADS environment variable for each process'
                f' to be {omp_num_threads} in default, to avoid your system '
                'being overloaded, please further tune the variable for '
                'optimal performance in your application as needed.')
        if omp_num_threads is not None:
            os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

        # setup MKL threads
        if mkl_num_threads is None and 'MKL_NUM_THREADS' not in os.environ:
            mkl_num_threads = 1
            warnings.warn(
                'Setting MKL_NUM_THREADS environment variable for each process'
                f' to be {mkl_num_threads} in default, to avoid your system '
                'being overloaded, please further tune the variable for '
                'optimal performance in your application as needed.')
        if mkl_num_threads is not None:
            os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)
