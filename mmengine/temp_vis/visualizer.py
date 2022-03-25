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
        print('Visualizer')

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
                                      List[Union[int, float]]] = 1,
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
        plt.close(self.fig)
        for writer in self._writers:
            writer.close()

# class WandbVisualizer(Visualizer):
#
#     def __init__(self,
#                  name='visualizer',
#                  writers=None,
#                  image: Optional[np.ndarray] = None,
#                  metadata: Optional[dict] = None,
#                  ):
#         super().__init__(name, writers, image, metadata)
#         self._bbox_data = []
#
#     def get_image(self):
#         image = self._image
#         _wandb = self.get_writer(1).experiment
#         boxes = {
#             'predictions': {
#                 'box_data':  self._bbox_data
#             }
#         }
#         # boxes = {'box_data': self._bbox_data}
#         return _wandb.Image(image, boxes=boxes)
#
#     def add_image(self,
#                   name: str,
#                   image=None,
#                   step: int = 0) -> None:
#         if image is None:
#             # 如果不是图片对象，则只能支持 wandb存储
#             # 此时打印警告，说明只有 wandb 存储才有效，其余存储忽略
#             image_data = self.get_image()
#             writer = self.get_writer(1)
#             writer.add_image(name, image_data, step)
#         else:
#             # 如果是图片则照旧
#             super(WandbVisualizer, self).add_image()
#
#     def draw_bboxes(self,
#                     bboxes: Union[np.ndarray, torch.Tensor],
#                     **kwargs) -> 'WandbVisualizer':
#
#         position = dict(
#             minX=int(bboxes[0]),
#             minY=int(bboxes[1]),
#             maxX=int(bboxes[2]),
#             maxY=int(bboxes[3]))
#
#         self._bbox_data = [{'position': position}]
