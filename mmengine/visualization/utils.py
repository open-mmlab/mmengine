# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, List, Optional, Tuple, Type, Union

import cv2
import matplotlib
import numpy as np
import torch


def tensor2ndarray(value: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """If the type of value is torch.Tensor, convert the value to np.ndarray.

    Args:
        value (np.ndarray, torch.Tensor): value.

    Returns:
        Any: value.
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return value


def value2list(value: Any, valid_type: Union[Type, Tuple[Type, ...]],
               expand_dim: int) -> List[Any]:
    """If the type of ``value`` is ``valid_type``, convert the value to list
    and expand to ``expand_dim``.

    Args:
        value (Any): value.
        valid_type (Union[Type, Tuple[Type, ...]): valid type.
        expand_dim (int): expand dim.

    Returns:
        List[Any]: value.
    """
    if isinstance(value, valid_type):
        value = [value] * expand_dim
    return value


def check_type(name: str, value: Any,
               valid_type: Union[Type, Tuple[Type, ...]]) -> None:
    """Check whether the type of value is in ``valid_type``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_type (Type, Tuple[Type, ...]): expected type.
    """
    if not isinstance(value, valid_type):
        raise TypeError(f'`{name}` should be {valid_type} '
                        f' but got {type(value)}')


def check_length(name: str, value: Any, valid_length: int) -> None:
    """If type of the ``value`` is list, check whether its length is equal with
    or greater than ``valid_length``.

    Args:
        name (str): value name.
        value (Any): value.
        valid_length (int): expected length.
    """
    if isinstance(value, list):
        if len(value) < valid_length:
            raise AssertionError(
                f'The length of {name} must equal with or '
                f'greater than {valid_length}, but got {len(value)}')


def check_type_and_length(name: str, value: Any,
                          valid_type: Union[Type, Tuple[Type, ...]],
                          valid_length: int) -> None:
    """Check whether the type of value is in ``valid_type``. If type of the
    ``value`` is list, check whether its length is equal with or greater than
    ``valid_length``.

    Args:
        value (Any): value.
        legal_type (Type, Tuple[Type, ...]): legal type.
        valid_length (int): expected length.

    Returns:
        List[Any]: value.
    """
    check_type(name, value, valid_type)
    check_length(name, value, valid_length)


def color_val_matplotlib(colors):
    """Convert various input in RGB order to normalized RGB matplotlib color
    tuples,
    Args:
        color (:obj:`mmcv.Color`/str/tuple/int/ndarray): Color inputs
    Returns:
        tuple[float]: A tuple of 3 normalized floats indicating RGB channels.
    """
    if isinstance(colors, str):
        return colors
    elif isinstance(colors, tuple):
        assert len(colors) == 3
        for channel in colors:
            assert 0 <= channel <= 255
        colors = [channel / 255 for channel in colors]
        return tuple(colors)
    elif isinstance(colors, list):
        colors = [color_val_matplotlib(color) for color in colors]
        return colors
    else:
        raise TypeError(f'Invalid type for color: {type(colors)}')


def str_color_to_rgb(color):
    color = matplotlib.colors.to_rgb(color)
    color = tuple([int(c * 255) for c in color])
    return color


def convert_overlay_heatmap(feat_map: Union[np.ndarray, torch.Tensor],
                            img: Optional[np.ndarray] = None,
                            alpha: float = 0.5) -> np.ndarray:
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of origin image. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()
    norm_img = np.zeros(feat_map.shape)
    norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, alpha, heat_img, 1 - alpha, 0)
    return heat_img
