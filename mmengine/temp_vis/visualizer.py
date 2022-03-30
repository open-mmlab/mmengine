# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Callable, List, Optional, Tuple, Union
from mmengine.logging import BaseGlobalAccessible
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import (LineCollection, PatchCollection,
                                    PolyCollection)
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from mmengine.data import BaseDataSample
from mmengine.registry import VISUALIZERS
from mmengine.registry import WRITERS
from .writer import BaseWriter
from mmengine.visualization.utils import *


class Visualizer(BaseGlobalAccessible):
    def __init__(self,
                 name='visualizer',
                 writers=None,
                 image: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None,
                 ) -> None:
        super().__init__(name)
        self._metadata = metadata
        if image is not None:
            self._setup_fig(image)

        self._writers = []
        if writers is not None:
            assert isinstance(writers, list)
            for writer in writers:
                if isinstance(writer, dict):
                    self._writers.append(WRITERS.build(writer))
                else:
                    assert isinstance(writer, BaseWriter), \
                        f'writer should be an instance of a subclass of ' \
                        f'BaseWriter, but got {type(writer)}'
                    self._writers.append(writer)

    def set_image(self, image: np.ndarray) -> None:
        assert image is not None
        self._setup_fig(image)

    def get_image(self) -> np.ndarray:
        assert self._image is not None, 'Please set image using `set_image`'
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype('uint8')

    def show(self, wait_time: int = 0) -> None:
        """Show the drawn image.

        Args:
            wait_time (int, optional): Delay in milliseconds. 0 is the special
                value that means "forever". Defaults to 0.
        """
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)

    def get_writer(self, name) -> 'BaseWriter':
        return self._writers[name]

    def _setup_fig(self, image: np.ndarray) -> None:
        image = image.astype('uint8')
        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)
        fig = plt.figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's
        # truncation (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches((self.width + 1e-2) / self.dpi,
                            (self.height + 1e-2) / self.dpi)
        self.canvas = fig.canvas
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        ax = plt.gca()
        self.fig = fig
        self.ax = ax
        self.ax.imshow(
            image,
            extent=(0, self.width, self.height, 0),
            interpolation='none')

    def _is_posion_valid(self, position: np.ndarray) -> bool:
        """Judge whether the position is in image.

        Args:
            position (np.ndarray): The position to judge which last dim must
                be two and the format is [x, y].

        Returns:
            bool: Whether the position is in image.
        """
        flag = (position[..., 0] < self.width).all() and \
               (position[..., 0] >= 0).all() and \
               (position[..., 1] < self.height).all() and \
               (position[..., 1] >= 0).all()
        return flag

    def draw_bboxes(self,
                    bboxes: Union[np.ndarray, torch.Tensor],
                    alpha: Union[int, float] = 0.8,
                    edgecolors: Union[str, List[str]] = 'r',
                    linestyles: Union[str, List[str]] = '-',
                    linewidths: Union[Union[int, float],
                                      List[Union[int, float]]] = 2,
                    is_filling: bool = False) -> 'Visualizer':
        check_type('bboxes', bboxes, (np.ndarray, torch.Tensor))
        bboxes = tensor2ndarray(bboxes)

        if len(bboxes.shape) == 1:
            bboxes = bboxes[None]
        assert bboxes.shape[-1] == 4, (
            f'The shape of `bboxes` should be (N, 4), but got {bboxes.shape}')

        assert (bboxes[:, 0] <= bboxes[:, 2]).all() and (bboxes[:, 1] <=
                                                         bboxes[:, 3]).all()
        if not self._is_posion_valid(bboxes.reshape((-1, 2, 2))):
            warnings.warn(
                'Warning: The bbox is out of bounds,'
                ' the drawn bbox may not be in the image', UserWarning)
        poly = np.stack(
            (bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 1],
             bboxes[:, 2], bboxes[:, 3], bboxes[:, 0], bboxes[:, 3]),
            axis=-1).reshape(-1, 4, 2)
        poly = [p for p in poly]
        return self.draw_polygons(
            poly,
            alpha=alpha,
            edgecolors=edgecolors,
            linestyles=linestyles,
            linewidths=linewidths,
            is_filling=is_filling)

    def draw_texts(
            self,
            texts: Union[str, List[str]],
            positions: Union[np.ndarray, torch.Tensor],
            font_sizes: Optional[Union[int, List[int]]] = None,
            colors: Union[str, List[str]] = 'g',
            verticalalignments: Union[str, List[str]] = 'top',
            horizontalalignments: Union[str, List[str]] = 'left',
            font_families: Union[str, List[str]] = 'sans-serif',
            rotations: Union[int, str, List[Union[int, str]]] = 0,
            bboxes: Optional[Union[dict, List[dict]]] = None) -> 'Visualizer':
        """Draw single or multiple text boxes.

        Args:
            texts (Union[str, List[str]]): Texts to draw.
            positions (Union[np.ndarray, torch.Tensor]): The position to draw
                the texts, which should have the same length with texts and
                each dim contain x and y.
            font_sizes (Union[int, List[int]], optional): The font size of
                texts. ``font_sizes`` can have the same length with texts or
                just single value. If ``font_sizes`` is single value, all the
                texts will have the same font size. Defaults to None.
            colors (Union[str, List[str]]): The colors of texts. ``colors``
                can have the same length with texts or just single value.
                If ``colors`` is single value, all the texts will have the same
                colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g.
            verticalalignments (Union[str, List[str]]): The verticalalignment
                of texts. verticalalignment controls whether the y positional
                argument for the text indicates the bottom, center or top side
                of the text bounding box.
                ``verticalalignments`` can have the same length with
                texts or just single value. If ``verticalalignments`` is single
                value, all the texts will have the same verticalalignment.
                verticalalignment can be 'center' or 'top', 'bottom' or
                'baseline'. Defaults to 'top'.
            horizontalalignments (Union[str, List[str]]): The
                horizontalalignment of texts. Horizontalalignment controls
                whether the x positional argument for the text indicates the
                left, center or right side of the text bounding box.
                ``horizontalalignments`` can have
                the same length with texts or just single value.
                If ``horizontalalignments`` is single value, all the texts will
                have the same horizontalalignment. Horizontalalignment
                can be 'center','right' or 'left'. Defaults to 'left'.
            font_families (Union[str, List[str]]): The font family of
                texts. ``font_families`` can have the same length with texts or
                just single value. If ``font_families`` is single value, all
                the texts will have the same font family.
                font_familiy can be 'serif', 'sans-serif', 'cursive', 'fantasy'
                 or 'monospace'.  Defaults to 'sans-serif'.
            rotations (Union[int, List[int]]): The rotation degrees of
                texts. ``rotations`` can have the same length with texts or
                just single value. If ``rotations`` is single value, all the
                texts will have the same rotation. rotation can be angle
                in degrees, 'vertical' or 'horizontal'. Defaults to 0.
            bboxes (Union[dict, List[dict]], optional): The bounding box of the
                texts. If bboxes is None, there are no bounding box around
                texts. ``bboxes`` can have the same length with texts or
                just single value. If ``bboxes`` is single value, all
                the texts will have the same bbox. Reference to
                https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch
                for more details. Defaults to None.
        """
        check_type('texts', texts, (str, list))
        if isinstance(texts, str):
            texts = [texts]
        num_text = len(texts)
        check_type('positions', positions, (np.ndarray, torch.Tensor))
        positions = tensor2ndarray(positions)
        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape == (num_text, 2), (
            '`positions` should have the shape of '
            f'({num_text}, 2), but got {positions.shape}')
        if not self._is_posion_valid(positions):
            warnings.warn(
                'Warning: The text is out of bounds,'
                ' the drawn text may not be in the image', UserWarning)
        positions = positions.tolist()

        if font_sizes is None:
            font_sizes = self._default_font_size
        check_type_and_length('font_sizes', font_sizes, (int, list), num_text)
        font_sizes = value2list(font_sizes, int, num_text)

        check_type_and_length('colors', colors, (str, list), num_text)
        colors = value2list(colors, str, num_text)

        check_type_and_length('verticalalignments', verticalalignments,
                              (str, list), num_text)
        verticalalignments = value2list(verticalalignments, str, num_text)

        check_type_and_length('horizontalalignments', horizontalalignments,
                              (str, list), num_text)
        horizontalalignments = value2list(horizontalalignments, str, num_text)

        check_type_and_length('rotations', rotations, (int, list), num_text)
        rotations = value2list(rotations, int, num_text)

        check_type_and_length('font_families', font_families, (str, list),
                              num_text)
        font_families = value2list(font_families, str, num_text)

        if bboxes is None:
            bboxes = [None for _ in range(num_text)]  # type: ignore
        else:
            check_type_and_length('bboxes', bboxes, (dict, list), num_text)
            bboxes = value2list(bboxes, dict, num_text)

        for i in range(num_text):
            self.ax.text(
                positions[i][0],
                positions[i][1],
                texts[i],
                size=font_sizes[i],  # type: ignore
                bbox=bboxes[i],  # type: ignore
                verticalalignment=verticalalignments[i],
                horizontalalignment=horizontalalignments[i],
                family=font_families[i],
                color=colors[i])
        return self


    def draw_polygons(self,
                      polygons: Union[Union[np.ndarray, torch.Tensor],
                                      List[Union[np.ndarray, torch.Tensor]]],
                      alpha: Union[int, float] = 0.8,
                      edgecolors: Union[str, List[str]] = 'g',
                      linestyles: Union[str, List[str]] = '-',
                      linewidths: Union[Union[int, float],
                                        List[Union[int, float]]] = 1.0,
                      is_filling: bool = False) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            polygons (Union[Union[np.ndarray, torch.Tensor],
                List[Union[np.ndarray, torch.Tensor]]]): The polygons to draw
                with the format of (x1,y1,x2,y2,...,xn,yn).
            edgecolors (Union[str, List[str]]): The colors of polygons.
                ``colors`` can have the same length with lines or just single
                value. If ``colors`` is single value, all the lines will have
                the same colors. Refer to `matplotlib.colors` for full list of
                formats that are accepted.. Defaults to 'g.
            linestyles (Union[str, List[str]]): The linestyle
                of lines. ``linestyles`` can have the same length with
                texts or just single value. If ``linestyles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            linewidths (Union[Union[int, float], List[Union[int, float]]]): The
                linewidth of lines. ``linewidths`` can have
                the same length with lines or just single value.
                If ``linewidths`` is single value, all the lines will
                have the same linewidth. Defaults to 1.
            is_filling (bool): Whether to fill all the polygons. Defaults to
                False.
        """
        check_type('polygons', polygons, (list, np.ndarray, torch.Tensor))

        if isinstance(polygons, (np.ndarray, torch.Tensor)):
            polygons = [polygons]
        if isinstance(polygons, list):
            for polygon in polygons:
                assert polygon.shape[1] == 2, (
                    'The shape of each polygon in `polygons` should be (M, 2),'
                    f' but got {polygon.shape}')
        polygons = [tensor2ndarray(polygon) for polygon in polygons]
        for polygon in polygons:
            if not self._is_posion_valid(polygon):
                warnings.warn(
                    'Warning: The polygon is out of bounds,'
                    ' the drawn polygon may not be in the image', UserWarning)
        if is_filling:
            polygon_collection = PolyCollection(
                polygons, alpha=alpha, facecolor=edgecolors)
        else:
            if isinstance(linewidths, (int, float)):
                linewidths = [linewidths] * len(polygons)
            linewidths = [
                min(max(linewidth, 1), self._default_font_size / 4)
                for linewidth in linewidths
            ]
            polygon_collection = PolyCollection(
                polygons,
                alpha=alpha,
                facecolor='none',
                linestyles=linestyles,
                edgecolors=edgecolors,
                linewidths=linewidths)

        self.ax.add_collection(polygon_collection)
        return self

    def draw_binary_masks(
            self,
            binary_masks: Union[np.ndarray, torch.Tensor],
            colors: np.ndarray = np.array([0, 255, 0]),
            alphas: Union[float, List[float]] = 0.2) -> 'Visualizer':
        """Draw single or multiple binary masks.
        Args:
            binary_masks (np.ndarray, torch.Tensor): The binary_masks to draw
                with of shape (N, H, W), where H is the image height and W is
                the image width. Each value in the array is either a 0 or 1
                value of uint8 type.
            colors (np.ndarray): The colors which binary_masks will convert to.
                ``colors`` can have the same length with binary_masks or just
                single value. If ``colors`` is single value, all the
                binary_masks will convert to the same colors. The colors format
                is RGB. Defaults to np.array([0, 255, 0]).
            alphas (Union[int, List[int]]): The transparency of origin image.
                Defaults to 0.5.
        """
        check_type('binary_masks', binary_masks, (np.ndarray, torch.Tensor))
        binary_masks = tensor2ndarray(binary_masks)
        assert binary_masks.dtype == np.bool_, (
            'The dtype of binary_masks should be np.bool_, '
            f'but got {binary_masks.dtype}')
        binary_masks = binary_masks.astype('uint8') * 255
        img = self.get_image()
        if binary_masks.ndim == 2:
            binary_masks = binary_masks[None]
        assert img.shape[:2] == binary_masks.shape[
                                1:], '`binary_marks` must have the same shpe with image'
        assert isinstance(colors, np.ndarray)
        if colors.ndim == 1:
            colors = np.tile(colors, (binary_masks.shape[0], 1))
        assert colors.shape == (binary_masks.shape[0], 3)
        if isinstance(alphas, float):
            alphas = [alphas] * binary_masks.shape[0]

        for binary_mask, color, alpha in zip(binary_masks, colors, alphas):
            binary_mask_complement = cv2.bitwise_not(binary_mask)
            rgb = np.zeros_like(img)
            rgb[...] = color
            rgb = cv2.bitwise_and(rgb, rgb, mask=binary_mask)
            img_complement = cv2.bitwise_and(
                img, img, mask=binary_mask_complement)
            rgb = rgb + img_complement
            img = cv2.addWeighted(img, alpha, rgb, 1 - alpha, 0)
        self.ax.imshow(
            img,
            extent=(0, self.width, self.height, 0),
            interpolation='nearest')
        return self

    def add_datasample(self,
                       name,
                       image: Optional[np.ndarray] = None,
                       gt_sample: Optional['BaseDataSample'] = None,
                       pred_sample: Optional['BaseDataSample'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       step=0) -> None:
        pass

    def add_image(self,
                  name: str,
                  image,
                  step: int = 0) -> None:
        for writer in self._writers:
            writer.add_image(name, image, step)

    def close(self) -> None:
        # plt.close(self.fig)
        for writer in self._writers:
            writer.close()
