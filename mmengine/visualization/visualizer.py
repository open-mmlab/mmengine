# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import (LineCollection, PatchCollection,
                                    PolyCollection)
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from mmengine.registry import VISUALIZERS


@VISUALIZERS.register_module()
class Visualizer:
    task_dict: dict = {}

    def __init__(self, metadata=None, image=None, scale: float = 1.0) -> None:
        self._metadata = metadata
        self._image = image
        self._scale = scale

        if image is not None:
            self._setup_fig(image)

    @property
    def experiment(self):
        """Return the experiment object associated with this writer."""
        return self

    def close(self):
        pass

    def show(self, drawn_image=None, winname='win', wait_time=0):
        cv2.namedWindow(winname, 0)
        cv2.imshow(winname,
                   self.get_image() if drawn_image is None else drawn_image)
        cv2.waitKey(wait_time)

    @classmethod
    def register_task(cls, task_name, force=False):

        def _register(task_func):
            print(task_func.__class__)
            if (task_name not in cls.task_dict) or force:
                cls.task_dict[task_name] = task_func
            else:
                raise KeyError(
                    f'{task_name} is already registered in task_dict, '
                    'add "force=True" if you want to override it')
            return task_func

        return _register

    def set_image(self, image):
        self._setup_fig(image)

    def get_image(self) -> np.ndarray:
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype('uint8')

    def _setup_fig(self, image: np.ndarray) -> None:
        self.width, self.height = image.shape[1], image.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10 // self._scale)
        fig = Figure(frameon=False)

        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's
        # truncation (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches((self.width * self._scale + 1e-2) / self.dpi,
                            (self.height * self._scale + 1e-2) / self.dpi)
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis('off')
        self.fig = fig
        self.ax = ax
        img = image.astype('uint8')
        self.ax.imshow(
            img,
            extent=(0, self.width, self.height, 0),
            interpolation='nearest')

    def draw(self, data_sample, image=None, show_gt=True, show_pred=True):
        pass

    def draw_texts(
            self,
            text: Union[str, List[str]],
            position: Union[np.ndarray, torch.Tensor],
            font_size: Optional[Union[int, List[int]]] = None,
            color: Union[str, List[str]] = 'g',
            verticalalignment: Union[str, List[str]] = 'top',
            horizontalalignment: Union[str, List[str]] = 'left',
            rotation: Union[int, List[int]] = 0,
            bbox: Optional[Union[dict, List[dict]]] = None,
            family: Union[str, List[str]] = 'sans-serif') -> 'Visualizer':

        def check_and_expand(value, legal_type, expand_dim):

            if isinstance(value, legal_type):
                value = [value] * expand_dim
            else:
                assert isinstance(value, list)
                assert len(font_size) >= num_text
            return value

        if isinstance(text, str):
            text = [text]
        assert isinstance(text, list)
        num_text = len(text)
        if len(position.shape) == 1:
            position = position[None]
        assert position.shape == (num_text, 2)
        assert isinstance(position, (np.ndarray, torch.Tensor))
        position = position.tolist()

        if font_size is None:
            font_size = self._default_font_size
        font_size = check_and_expand(font_size, int, num_text)  # type: ignore
        font_size = [size * self._scale for size in font_size]  # type: ignore

        if bbox is None:
            bbox = dict(facecolor=color, alpha=0.6)
        bbox = check_and_expand(bbox, dict, num_text)  # type: ignore

        color = check_and_expand(color, str, num_text)
        verticalalignment = check_and_expand(verticalalignment, str, num_text)
        horizontalalignment = check_and_expand(horizontalalignment, str,
                                               num_text)
        rotation = check_and_expand(rotation, int, num_text)
        family = check_and_expand(family, str, num_text)
        for i in range(num_text):
            self.ax.text(
                position[i][0],
                position[i][1],
                text[i],
                size=font_size[i],  # type: ignore
                bbox=bbox[i],  # type: ignore
                verticalalignment=verticalalignment[i],
                horizontalalignment=horizontalalignment[i],
                family=family[i],
                color=color[i])
        return self

    def draw_lines(
        self,
        x_datas: Union[np.ndarray, torch.Tensor],
        y_datas: Union[np.ndarray, torch.Tensor],
        color: Union[str, List[str]] = 'g',
        linestyle: Union[str, List[str]] = '-',
        linewidth: Union[Union[int, float], List[Union[int, float]]] = 1
    ) -> 'Visualizer':
        assert isinstance(x_datas, (np.ndarray, torch.Tensor))
        assert isinstance(y_datas, (np.ndarray, torch.Tensor))
        assert x_datas.shape == y_datas.shape
        assert x_datas.shape[-1] == 2
        if len(x_datas.shape) == 1:
            x_datas = x_datas[None]
            y_datas = y_datas[None]

        x_datas = x_datas.tolist()
        y_datas = y_datas.tolist()
        lines = []
        for i in range(len(x_datas)):
            line = []
            for x, y in zip(x_datas[i], y_datas[i]):
                line.append([x, y])
            lines.append(line)

        line_collect = LineCollection(
            lines, colors=color, linestyles=linestyle, linewidths=linewidth)
        self.ax.add_collection(line_collect)
        return self

    def draw_circles(self,
                     circle_coord: Union[np.ndarray, torch.Tensor],
                     radius: Union[np.ndarray, torch.Tensor],
                     alpha: Union[float, int] = 0.8,
                     edgecolors: Union[str, List[str]] = 'g',
                     linestyles: Union[str, List[str]] = '-',
                     linewidths: Union[int, List[int]] = 1,
                     is_filling: bool = True) -> 'Visualizer':
        assert isinstance(circle_coord, (np.ndarray, torch.Tensor))
        assert isinstance(radius, (np.ndarray, torch.Tensor))

        if len(circle_coord.shape) == 1:
            circle_coord = circle_coord[None]
        assert circle_coord.shape == (radius.shape[0], 2)

        circle_coord = circle_coord.tolist()
        radius = radius.tolist()
        circles = []
        for i in range(len(circle_coord)):
            circles.append(Circle(tuple(circle_coord[i]), radius[i]))
        if is_filling:
            p = PatchCollection(circles, alpha=alpha, facecolor=edgecolors)
        else:
            if isinstance(linewidths, (int, float)):
                linewidths = [linewidths]
            linewidths = [
                min(max(linewidth, 1), self._default_font_size / 4) *
                self._scale for linewidth in linewidths
            ]
            p = PatchCollection(
                circles,
                alpha=alpha,
                facecolor='none',
                edgecolor=edgecolors,
                linewidth=linewidths,
                linestyles=linestyles)
        self.ax.add_collection(p)
        return self

    def draw_bboxes(self,
                    bbox: Union[np.ndarray, torch.Tensor],
                    alpha: Union[int, float] = 0.8,
                    edgecolors: Union[str, List[str]] = 'g',
                    linestyles: Union[str, List[str]] = '-',
                    linewidths: Union[int, List[int]] = 1,
                    is_filling: bool = True) -> 'Visualizer':
        assert isinstance(bbox, (np.ndarray, torch.Tensor))
        assert bbox.shape[-1] == 4
        if len(bbox.shape) == 1:
            bbox = bbox[None]
        if isinstance(bbox, np.ndarray):
            bbox = bbox.detach().cpu().numpy()
        poly = np.stack((bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 1],
                         bbox[:, 2], bbox[:, 3], bbox[:, 0], bbox[:, 3]),
                        axis=-1).reshape(-1, 4, 2)
        self.draw_polygons(
            poly,
            alpha=alpha,
            edgecolors=edgecolors,
            linestyles=linestyles,
            linewidths=linewidths,
            is_filling=is_filling)
        return self

    def draw_polygons(self,
                      segment: Union[Union[np.ndarray, torch.Tensor],
                                     List[Union[np.ndarray, torch.Tensor]]],
                      alpha: Union[int, float] = 0.8,
                      edgecolors: Union[str, List[str]] = 'g',
                      linestyles: Union[str, List[str]] = '-',
                      linewidths: Union[int, List[int]] = 1,
                      is_filling: bool = True) -> 'Visualizer':

        if is_filling:
            polygon_collection = PolyCollection(
                segment, alpha=alpha, facecolor=edgecolors)
        else:
            if isinstance(linewidths, (int, float)):
                linewidths = [linewidths]
            linewidths = [
                min(max(linewidth, 1), self._default_font_size / 4) *
                self._scale for linewidth in linewidths
            ]
            polygon_collection = PolyCollection(
                segment,
                alpha=alpha,
                facecolor='none',
                linestyles=linestyles,
                edgecolors=edgecolors,
                linewidths=linewidths)

        self.ax.add_collection(polygon_collection)
        return self

    def draw_binary_masks(self,
                          binary_mask: np.ndarray,
                          color: Sequence[int],
                          alpha: float = 0.5):
        img = self.get_image()
        assert img.shape[:2] == binary_mask.shape
        binary_mask_complement = 1 - binary_mask
        rgb = np.zeros_like(img)
        rgb[...] = color
        rgb = cv2.bitwise_and(rgb, rgb, mask=binary_mask)
        img_complement = cv2.bitwise_and(img, img, mask=binary_mask_complement)
        rgb = rgb + img_complement
        img_add = cv2.addWeighted(img, alpha, rgb, 1 - alpha, 0)
        self.ax.imshow(
            img_add,
            extent=(0, self.width, self.height, 0),
            interpolation='nearest')
        return self

    @staticmethod
    def draw_featmap(tensor_chw,
                     img=None,
                     mode='mean',
                     topk=10,
                     arrangement=(5, 2),
                     alpha=0.3) -> np.ndarray:

        def concat_heatmap(feat_map, img=None, alpha=0.5):
            norm_img = np.zeros(feat_map.shape)
            norm_img = cv2.normalize(feat_map.numpy(), norm_img, 0, 255,
                                     cv2.NORM_MINMAX)
            norm_img = np.asarray(norm_img, dtype=np.uint8)
            heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
            heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
            if img is not None:
                heat_img = cv2.addWeighted(img, alpha, heat_img, 1 - alpha, 0)
            return heat_img

        tensor_chw = tensor_chw.detach().cpu()
        if img is not None:
            assert img.shape[:2] == tensor_chw.shape[1:]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if mode is not None:
            assert mode in ['mean', 'max', 'min']
            if mode == 'max':
                feat_map = torch.max(tensor_chw, dim=0)
            elif mode == 'min':
                feat_map = torch.min(tensor_chw, dim=0)
            elif mode == 'mean':
                feat_map = torch.mean(tensor_chw, dim=0)
            return concat_heatmap(feat_map, img, alpha)

        if topk == -1:
            tensor_chw_channel = tensor_chw.shape[0]
            assert tensor_chw_channel in [1, 3]
            if tensor_chw_channel == 1:
                return concat_heatmap(tensor_chw, img, alpha)
            else:
                tensor_chw = tensor_chw.permute(1, 2, 0).numpy()
                norm_img = np.zeros(tensor_chw.shape)
                norm_img = cv2.normalize(tensor_chw, None, 0, 255,
                                         cv2.NORM_MINMAX)
                norm_img = np.asarray(norm_img, dtype=np.uint8)
                heat_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
                if img is not None:
                    heat_img = cv2.addWeighted(img, alpha, heat_img, 1 - alpha,
                                               0)
                return heat_img
        else:
            assert topk > 0
            row, col = arrangement
            height, width = tensor_chw.shape[1:]
            sum_channel = torch.sum(tensor_chw, dim=(1, 2))
            _, indices = torch.topk(sum_channel, topk)
            topk_tensor = tensor_chw[indices]
            fig = Figure(frameon=False)
            dpi = fig.get_dpi()
            fig.set_size_inches((width * col + 1e-2) / dpi,
                                (height * row + 1e-2) / dpi)
            canvas = FigureCanvasAgg(fig)
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.tight_layout(h_pad=0, w_pad=0)

            for i in range(topk):
                axes = fig.add_subplot(row, col, i + 1)
                axes.axis('off')
                axes.imshow(concat_heatmap(topk_tensor[i], img, alpha))
            s, (width, height) = canvas.print_to_buffer()
            buffer = np.frombuffer(s, dtype='uint8')
            img_rgba = buffer.reshape(height, width, 4)
            rgb, alpha = np.split(img_rgba, [3], axis=2)
            return rgb.astype('uint8')
