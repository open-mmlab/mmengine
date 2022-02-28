# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, Optional, Union

from torch.testing import assert_allclose as _assert_allclose

from mmengine.utils import TORCH_VERSION, digit_version


def assert_allclose(
    actual: Any,
    expected: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = True,
    msg: Optional[Union[str, Callable]] = '',
) -> None:
    """Asserts that ``actual`` and ``expected`` are close. A wrapper function
    of ``torch.testing.assert_allclose``.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must
            also be specified. If omitted, default values based on the
            :attr:`~torch.Tensor.dtype` are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified :attr:`rtol`
            must also be specified. If omitted, default values based on the
            :attr:`~torch.Tensor.dtype` are selected with the below table.
        equal_nan (bool): If ``True``, two ``NaN`` values will be considered
            equal.
        msg (Optional[Union[str, Callable]]): Optional error message to use if
            the values of corresponding tensors mismatch. Unused when PyTorch
            < 1.6.
    """
    if 'parrots' not in TORCH_VERSION and \
            digit_version(TORCH_VERSION) >= digit_version('1.6'):
        _assert_allclose(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            msg=msg)
    else:
        # torch.testing.assert_allclose has no ``msg`` argument
        # when PyTorch < 1.6
        _assert_allclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
