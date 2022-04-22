# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backend_bases import CloseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import (LineCollection, PatchCollection,
                                    PolyCollection)
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from mmengine.config import Config
from mmengine.data import BaseDataElement
from mmengine.registry import VISBACKENDS, VISUALIZERS
from mmengine.utils import ManagerMixin
from mmengine.visualization.utils import (check_type, check_type_and_length,
                                          color_val_matplotlib,
                                          convert_overlay_heatmap,
                                          str_color_to_rgb, tensor2ndarray,
                                          value2list)
from mmengine.visualization.vis_backend import BaseVisBackend


@VISUALIZERS.register_module()
class Visualizer(ManagerMixin):
    """MMEngine provides a Visualizer class that uses the ``Matplotlib``
    library as the backend. It has the following functions:

    - Basic info methods

      - set_image: sets the original image data
      - get_image: get the image data in Numpy format after drawing
      - show: visualization.
      - register_task: registers the drawing function.

    - Basic drawing methods

      - draw_bboxes: draw single or multiple bounding boxes
      - draw_texts: draw single or multiple text boxes
      - draw_lines: draw single or multiple line segments
      - draw_circles: draw single or multiple circles
      - draw_polygons: draw single or multiple polygons
      - draw_binary_masks: draw single or multiple binary masks
      - draw: The abstract drawing interface used by the user

    - Enhanced methods

      - draw_featmap: draw feature map

    All the basic drawing methods support chain calls, which is convenient for
    overlaydrawing and display. Each downstream algorithm library can inherit
    ``Visualizer`` and implement the draw logic in the draw interface. For
    example, ``DetVisualizer`` in MMDetection inherits from ``Visualizer``
    and implements functions, such as visual detection boxes, instance masks,
    and semantic segmentation maps in the draw interface.

    Args:
        metadata (dict, optional): A dict contains the meta information
            of single image. such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.

    Examples:
        >>> # Basic info methods
        >>> vis = Visualizer()
        >>> vis.set_image(image)
        >>> vis.get_image()
        >>> vis.show()

        >>> # Basic drawing methods
        >>> vis = Visualizer(metadata=metadata, image=image)
        >>> vis.draw_bboxes(np.array([0, 0, 1, 1]), edge_colors='g')
        >>> vis.draw_bboxes(bbox=np.array([[1, 1, 2, 2], [2, 2, 3, 3]]),
                            edge_colors=['g', 'r'], is_filling=True)
        >>> vis.draw_lines(x_datas=np.array([1, 3]),
                        y_datas=np.array([1, 3]),
                        colors='r', line_widths=1)
        >>> vis.draw_lines(x_datas=np.array([[1, 3], [2, 4]]),
                        y_datas=np.array([[1, 3], [2, 4]]),
                        colors=['r', 'r'], line_widths=[1, 2])
        >>> vis.draw_texts(text='MMEngine',
                        position=np.array([2, 2]),
                        colors='b')
        >>> vis.draw_texts(text=['MMEngine','OpenMMLab']
                        position=np.array([[2, 2], [5, 5]]),
                        colors=['b', 'b'])
        >>> vis.draw_circles(circle_coord=np.array([2, 2]), radius=np.array[1])
        >>> vis.draw_circles(circle_coord=np.array([[2, 2], [3, 5]),
                            radius=np.array[1, 2], colors=['g', 'r'],
                            is_filling=True)
        >>> vis.draw_polygons(np.array([0, 0, 1, 0, 1, 1, 0, 1]),
                            edge_colors='g')
        >>> vis.draw_polygons(bbox=[np.array([0, 0, 1, 0, 1, 1, 0, 1],
                                    np.array([2, 2, 3, 2, 3, 3, 2, 3]]),
                            edge_colors=['g', 'r'], is_filling=True)
        >>> vis.draw_binary_masks(binary_mask, alpha=0.6)

        >>> # chain calls
        >>> vis.draw_bboxes().draw_texts().draw_circle().draw_binary_masks()

        >>> # Enhanced method
        >>> vis = Visualizer(metadata=metadata, image=image)
        >>> heatmap = vis.draw_featmap(tensor_chw, img, mode='mean')
        >>> heatmap = vis.draw_featmap(tensor_chw, img, mode=None,
                                    topk=8, arrangement=(4, 2))
        >>> heatmap = vis.draw_featmap(tensor_chw, img, mode=None,
                                    topk=-1)

        >>> # inherit
        >>> class DetVisualizer2(Visualizer):
        >>>     def add_datasample(self,
        >>>             image: Optional[np.ndarray] = None,
        >>>             gt_sample: Optional['BaseDataElement'] = None,
        >>>             pred_sample: Optional['BaseDataElement'] = None,
        >>>             show_gt: bool = True,
        >>>             show_pred: bool = True,
        >>>             show:bool = True) -> None:
        >>>         pass
    """

    def __init__(
        self,
        name='visualizer',
        image: Optional[np.ndarray] = None,
        vis_backends: Optional[Dict] = None,
        save_dir: Optional[str] = None,
        fig_save_cfg=dict(frameon=False),
        fig_show_cfg=dict(frameon=False, num='show')
    ) -> None:
        super().__init__(name)
        self._dataset_meta: Union[None, dict] = None
        self._vis_backends: Union[Dict, Dict[str, 'BaseVisBackend']] = dict()

        if vis_backends:
            with_name = False
            without_name = False
            for vis_backend in vis_backends:
                if 'name' in vis_backend:
                    with_name = True
                else:
                    without_name = True
            if with_name and without_name:
                raise AssertionError

            for vis_backend in vis_backends:
                name = vis_backend.pop('name', vis_backend['type'])
                assert name not in self._vis_backends
                vis_backend.setdefault('save_dir', save_dir)
                self._vis_backends[name] = VISBACKENDS.build(vis_backend)

        self.is_inline = 'inline' in plt.get_backend()

        self.fig_save = None
        self.fig_show = None
        self.fig_save_num = fig_save_cfg.get('num', None)
        self.fig_show_num = fig_show_cfg.get('num', None)
        self.fig_save_cfg = fig_save_cfg
        self.fig_show_cfg = fig_show_cfg

        (self.fig_save, self.ax_save,
         self.fig_save_num) = self._initialize_fig(fig_save_cfg)
        self.dpi = self.fig_save.get_dpi()
        if image is not None:
            self.set_image(image)

    @property
    def dataset_meta(self) -> Optional[dict]:
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        self._dataset_meta = dataset_meta

    def show(self,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: int = 0,
             continue_key=' ') -> None:
        """Show the drawn image.

        Args:
            wait_time (int, optional): Delay in milliseconds. 0 is the special
                value that means "forever". Defaults to 0.
        """
        if self.is_inline:
            return
        if self.fig_show is None or not plt.fignum_exists(self.fig_show_num):
            (self.fig_show, self.ax_show,
             self.fig_show_num) = self._initialize_fig(self.fig_show_cfg)
        img = self.get_image() if drawn_img is None else drawn_img
        # dpi = self.fig_show.get_dpi()
        # height, width = img.shape[:2]
        # self.fig_show.set_size_inches((width + 1e-2) / dpi,
        #                               (height + 1e-2) / dpi)
        self.ax_show.cla()
        self.ax_show.axis(False)
        # self.ax_show.set_title(win_name)
        # self.fig_show.set_label(win_name)

        # Refresh canvas, necessary for Qt5 backend.
        self.ax_show.imshow(img)
        self.fig_show.canvas.draw()  # type: ignore
        self._wait_continue(timeout=wait_time, continue_key=continue_key)

    def set_image(self, image: np.ndarray) -> None:
        """Set the image to draw.

        Args:
            image (np.ndarray): The image to draw.
        """
        assert image is not None
        image = image.astype('uint8')
        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)

        # add a small 1e-2 to avoid precision lost due to matplotlib's
        # truncation (https://github.com/matplotlib/matplotlib/issues/15363)
        self.fig_save.set_size_inches(  # type: ignore
            (self.width + 1e-2) / self.dpi, (self.height + 1e-2) / self.dpi)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        self.ax_save.cla()
        self.ax_save.axis(False)
        self.ax_save.imshow(
            image,
            extent=(0, self.width, self.height, 0),
            interpolation='none')

    def get_image(self) -> np.ndarray:
        """Get the drawn image. The format is RGB.

        Returns:
            np.ndarray: the drawn image which channel is rgb.
        """
        assert self._image is not None, 'Please set image using `set_image`'
        canvas = self.fig_save.canvas  # type: ignore
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype('uint8')

    def _initialize_fig(self, fig_cfg):
        fig = plt.figure(**fig_cfg)
        ax = fig.add_subplot()
        ax.axis(False)

        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        return fig, ax, fig.number

    def get_backend(self, name) -> 'BaseVisBackend':
        return self._vis_backends.get(name)  # type: ignore

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

    def _wait_continue(self, timeout: int = 0, continue_key=' ') -> int:
        """Show the image and wait for the user's input.

        This implementation refers to
        https://github.com/matplotlib/matplotlib/blob/v3.5.x/lib/matplotlib/_blocking_input.py

        Args:
            timeout (int): If positive, continue after ``timeout`` seconds.
                Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.

        Returns:
            int: If zero, means time out or the user pressed ``continue_key``,
                and if one, means the user closed the show figure.
        """  # noqa: E501
        if self.is_inline:
            # If use inline backend, interactive input and timeout is no use.
            return 0

        if self.fig_show.canvas.manager:  # type: ignore
            # Ensure that the figure is shown
            self.fig_show.show()  # type: ignore

        while True:

            # Connect the events to the handler function call.
            event = None

            def handler(ev):
                # Set external event variable
                nonlocal event
                # Qt backend may fire two events at the same time,
                # use a condition to avoid missing close event.
                event = ev if not isinstance(event, CloseEvent) else event
                self.fig_show.canvas.stop_event_loop()

            cids = [
                self.fig_show.canvas.mpl_connect(name, handler)  # type: ignore
                for name in ('key_press_event', 'close_event')
            ]

            try:
                self.fig_show.canvas.start_event_loop(timeout)  # type: ignore
            finally:  # Run even on exception like ctrl-c.
                # Disconnect the callbacks.
                for cid in cids:
                    self.fig_show.canvas.mpl_disconnect(cid)  # type: ignore

            if isinstance(event, CloseEvent):
                return 1  # Quit for close.
            elif event is None or event.key == continue_key:
                return 0  # Quit for continue.

    def draw_points(self,
                    positions: Union[np.ndarray, torch.Tensor],
                    colors: Union[str, tuple, List[str], List[tuple]] = 'g',
                    marker: Optional[str] = None,
                    sizes: Optional[Union[np.ndarray, torch.Tensor]] = None):
        check_type('positions', positions, (np.ndarray, torch.Tensor))
        positions = tensor2ndarray(positions)

        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape[-1] == 2, (
            'The shape of `positions` should be (N, 2), '
            f'but got {positions.shape}')
        colors = color_val_matplotlib(colors)
        self.ax_save.scatter(
            positions[:, 0], positions[:, 1], c=colors, s=sizes, marker=marker)
        return self

    def draw_texts(
            self,
            texts: Union[str, List[str]],
            positions: Union[np.ndarray, torch.Tensor],
            font_sizes: Optional[Union[int, List[int]]] = None,
            colors: Union[str, tuple, List[str], List[tuple]] = 'g',
            vertical_alignments: Union[str, List[str]] = 'top',
            horizontal_alignments: Union[str, List[str]] = 'left',
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
            colors (Union[str, tuple, List[str], List[tuple]]): The colors
                of texts. ``colors`` can have the same length with texts or
                just single value. If ``colors`` is single value, all the
                texts will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g.
            vertical_alignments (Union[str, List[str]]): The verticalalignment
                of texts. verticalalignment controls whether the y positional
                argument for the text indicates the bottom, center or top side
                of the text bounding box.
                ``vertical_alignments`` can have the same length with
                texts or just single value. If ``vertical_alignments`` is
                single value, all the texts will have the same
                verticalalignment. verticalalignment can be 'center' or
                'top', 'bottom' or 'baseline'. Defaults to 'top'.
            horizontal_alignments (Union[str, List[str]]): The
                horizontalalignment of texts. Horizontalalignment controls
                whether the x positional argument for the text indicates the
                left, center or right side of the text bounding box.
                ``horizontal_alignments`` can have
                the same length with texts or just single value.
                If ``horizontal_alignments`` is single value, all the texts
                will have the same horizontalalignment. Horizontalalignment
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
        check_type_and_length('font_sizes', font_sizes, (int, float, list),
                              num_text)
        font_sizes = value2list(font_sizes, (int, float), num_text)

        check_type_and_length('colors', colors, (str, tuple, list), num_text)
        colors = value2list(colors, (str, tuple), num_text)
        colors = color_val_matplotlib(colors)

        check_type_and_length('vertical_alignments', vertical_alignments,
                              (str, list), num_text)
        vertical_alignments = value2list(vertical_alignments, str, num_text)

        check_type_and_length('horizontal_alignments', horizontal_alignments,
                              (str, list), num_text)
        horizontal_alignments = value2list(horizontal_alignments, str,
                                           num_text)

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
            self.ax_save.text(
                positions[i][0],
                positions[i][1],
                texts[i],
                size=font_sizes[i],  # type: ignore
                bbox=bboxes[i],  # type: ignore
                verticalalignment=vertical_alignments[i],
                horizontalalignment=horizontal_alignments[i],
                family=font_families[i],
                color=colors[i])
        return self

    def draw_lines(
        self,
        x_datas: Union[np.ndarray, torch.Tensor],
        y_datas: Union[np.ndarray, torch.Tensor],
        colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2
    ) -> 'Visualizer':
        """Draw single or multiple line segments.

        Args:
            x_datas (Union[np.ndarray, torch.Tensor]): The x coordinate of
                each line' start and end points.
            y_datas (Union[np.ndarray, torch.Tensor]): The y coordinate of
                each line' start and end points.
            colors (Union[str, tuple, List[str], List[tuple]]): The colors of
                lines. ``colors`` can have the same length with lines or just
                single value. If ``colors`` is single value, all the lines
                will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
        """
        check_type('x_datas', x_datas, (np.ndarray, torch.Tensor))
        x_datas = tensor2ndarray(x_datas)
        check_type('y_datas', y_datas, (np.ndarray, torch.Tensor))
        y_datas = tensor2ndarray(y_datas)
        assert x_datas.shape == y_datas.shape, (
            '`x_datas` and `y_datas` should have the same shape')
        assert x_datas.shape[-1] == 2, (
            f'The shape of `x_datas` should be (N, 2), but got {x_datas.shape}'
        )
        if len(x_datas.shape) == 1:
            x_datas = x_datas[None]
            y_datas = y_datas[None]
        colors = color_val_matplotlib(colors)
        lines = np.concatenate(
            (x_datas.reshape(-1, 2, 1), y_datas.reshape(-1, 2, 1)), axis=-1)
        if not self._is_posion_valid(lines):
            warnings.warn(
                'Warning: The line is out of bounds,'
                ' the drawn line may not be in the image', UserWarning)
        line_collect = LineCollection(
            lines.tolist(),
            colors=colors,
            linestyles=line_styles,
            linewidths=line_widths)
        self.ax_save.add_collection(line_collect)
        return self

    def draw_circles(
        self,
        center: Union[np.ndarray, torch.Tensor],
        radius: Union[np.ndarray, torch.Tensor],
        alpha: Union[float, int] = 0.8,
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none'
    ) -> 'Visualizer':
        """Draw single or multiple circles.

        Args:
            center (Union[np.ndarray, torch.Tensor]): The x coordinate of
            each line' start and end points.
            radius (Union[np.ndarray, torch.Tensor]): The y coordinate of
            each line' start and end points.
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of circles. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value,
                all the lines will have the same colors. Reference to
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details. Defaults to 'g.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            is_filling (bool): Whether to fill all the circles. Defaults to
                False.
        """
        check_type('center', center, (np.ndarray, torch.Tensor))
        center = tensor2ndarray(center)
        check_type('radius', radius, (np.ndarray, torch.Tensor))
        radius = tensor2ndarray(radius)
        if len(center.shape) == 1:
            center = center[None]
        assert center.shape == (radius.shape[0], 2), (
            'The shape of `center` should be (radius.shape, 2), '
            f'but got {center.shape}')
        if not (self._is_posion_valid(center -
                                      np.tile(radius.reshape((-1, 1)), (1, 2)))
                and self._is_posion_valid(
                    center + np.tile(radius.reshape((-1, 1)), (1, 2)))):
            warnings.warn(
                'Warning: The circle is out of bounds,'
                ' the drawn circle may not be in the image', UserWarning)

        center = center.tolist()
        radius = radius.tolist()
        edge_colors = color_val_matplotlib(edge_colors)
        face_colors = color_val_matplotlib(face_colors)
        circles = []
        for i in range(len(center)):
            circles.append(Circle(tuple(center[i]), radius[i]))

        if isinstance(line_widths, (int, float)):
            line_widths = [line_widths] * len(circles)
        line_widths = [
            min(max(linewidth, 1), self._default_font_size / 4)
            for linewidth in line_widths
        ]
        p = PatchCollection(
            circles,
            alpha=alpha,
            facecolors=face_colors,
            edgecolors=edge_colors,
            linewidths=line_widths,
            linestyles=line_styles)
        self.ax_save.add_collection(p)
        return self

    def draw_bboxes(
        self,
        bboxes: Union[np.ndarray, torch.Tensor],
        alpha: Union[int, float] = 0.8,
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none'
    ) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            bboxes (Union[np.ndarray, torch.Tensor]): The bboxes to draw with
                the format of(x1,y1,x2,y2).
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of bboxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'g'.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 1.
            is_filling (bool): Whether to fill all the bboxes. Defaults to
                False.
        """
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
            edge_colors=edge_colors,
            line_styles=line_styles,
            line_widths=line_widths,
            face_colors=face_colors)

    def draw_polygons(
        self,
        polygons: Union[Union[np.ndarray, torch.Tensor],
                        List[Union[np.ndarray, torch.Tensor]]],
        alpha: Union[int, float] = 0.8,
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'g',
        line_styles: Union[str, List[str]] = '-',
        line_widths: Union[Union[int, float], List[Union[int, float]]] = 2,
        face_colors: Union[str, tuple, List[str], List[tuple]] = 'none'
    ) -> 'Visualizer':
        """Draw single or multiple bboxes.

        Args:
            polygons (Union[Union[np.ndarray, torch.Tensor],
                List[Union[np.ndarray, torch.Tensor]]]): The polygons to draw
                with the format of (x1,y1,x2,y2,...,xn,yn).
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of polygons. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value,
                all the lines will have the same colors. Refer to
                `matplotlib.colors` for full list of formats that are accepted.
                Defaults to 'g.
            line_styles (Union[str, List[str]]): The linestyle
                of lines. ``line_styles`` can have the same length with
                texts or just single value. If ``line_styles`` is single
                value, all the lines will have the same linestyle.
                Reference to
                https://matplotlib.org/stable/api/collections_api.html?highlight=collection#matplotlib.collections.AsteriskPolygonCollection.set_linestyle
                for more details. Defaults to '-'.
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The linewidth of lines. ``line_widths`` can have
                the same length with lines or just single value.
                If ``line_widths`` is single value, all the lines will
                have the same linewidth. Defaults to 2.
            is_filling (bool): Whether to fill all the polygons. Defaults to
                False.
        """
        check_type('polygons', polygons, (list, np.ndarray, torch.Tensor))
        edge_colors = color_val_matplotlib(edge_colors)
        face_colors = color_val_matplotlib(face_colors)

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
        if isinstance(line_widths, (int, float)):
            line_widths = [line_widths] * len(polygons)
        line_widths = [
            min(max(linewidth, 1), self._default_font_size / 4)
            for linewidth in line_widths
        ]
        polygon_collection = PolyCollection(
            polygons,
            alpha=alpha,
            facecolor=face_colors,
            linestyles=line_styles,
            edgecolors=edge_colors,
            linewidths=line_widths)

        self.ax_save.add_collection(polygon_collection)
        return self

    def draw_binary_masks(
        self,
        binary_masks: Union[np.ndarray, torch.Tensor],
        alphas: Union[float, List[float]] = 0.8,
        colors: Union[str, tuple, List[str],
                      List[tuple]] = 'g') -> 'Visualizer':
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
        binary_mask_len = binary_masks.shape[0]

        check_type_and_length('colors', colors, (str, tuple, list),
                              binary_mask_len)
        colors = value2list(colors, (str, tuple), binary_mask_len)
        colors = [
            str_color_to_rgb(color) if isinstance(color, str) else color
            for color in colors
        ]
        for color in colors:
            assert len(color) == 3
            for channel in color:
                assert 0 <= channel <= 255  # type: ignore
        colors = np.array(colors)
        if colors.ndim == 1:  # type: ignore
            colors = np.tile(colors, (binary_mask_len, 1))
        if isinstance(alphas, float):
            alphas = [alphas] * binary_mask_len

        for binary_mask, color, alpha in zip(binary_masks, colors, alphas):
            binary_mask_complement = cv2.bitwise_not(binary_mask)
            rgb = np.zeros_like(img)
            rgb[...] = color
            rgb = cv2.bitwise_and(rgb, rgb, mask=binary_mask)
            img_complement = cv2.bitwise_and(
                img, img, mask=binary_mask_complement)
            rgb = rgb + img_complement
            img = cv2.addWeighted(img, 1 - alpha, rgb, alpha, 0)
        self.ax_save.imshow(
            img,
            extent=(0, self.width, self.height, 0),
            interpolation='nearest')
        return self

    @staticmethod
    def draw_featmap(tensor_chw: torch.Tensor,
                     image: Optional[np.ndarray] = None,
                     mode: str = 'mean',
                     topk: int = 10,
                     arrangement: Tuple[int, int] = (5, 2),
                     alpha: float = 0.8) -> np.ndarray:
        """Draw featmap. If img is not None, the final image will be the
        weighted sum of img and featmap. It support the mode:

        - if mode is not None, it will compress tensor_chw to single channel
          image and sum to image.
        - if mode is None.

          - if topk <= 0, tensor_chw is assert to be one or three
          channel and treated as image and will be sum to ``image``.
          - if topk > 0, it will select topk channel to show by the sum of
          each channel.

        Args:
            tensor_chw (torch.Tensor): The featmap to draw which format is
                (C, H, W).
            image (np.ndarray): The colors which binary_masks will convert to.
                ``colors`` can have the same length with binary_masks or just
                single value. If ``colors`` is single value, all the
                binary_masks will convert to the same colors. The colors format
                is rgb. Defaults to np.array([0, 255, 0]).
            mode (str): The mode to compress `tensor_chw` to single channel.
                Defaults to 'mean'.
            topk (int): If mode is not None and topk > 0, it will select topk
                channel to show by the sum of each channel. if topk <= 0,
                tensor_chw is assert to be one or three. Defaults to 10.
            arrangement (Tuple[int, int]): The arrangement of featmaps when
                mode is not None and topk > 0. Defaults to (5, 2).
            alphas (Union[int, List[int]]): The transparency of origin image.
                Defaults to 0.5.
        Returns:
            np.ndarray: featmap.
        """
        assert isinstance(
            tensor_chw,
            torch.Tensor), (f'`tensor_chw` should be {torch.Tensor} '
                            f' but got {type(tensor_chw)}')
        tensor_chw = tensor_chw.detach().cpu()
        assert tensor_chw.ndim == 3, 'Input dimension must be 3'
        if image is not None:
            assert image.shape[:2] == tensor_chw.shape[1:]
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if mode is not None:
            assert mode in [
                'mean', 'max', 'min'
            ], (f'Mode only support "mean", "max", "min", but got {mode}')
            if mode == 'max':
                feat_map, _ = torch.max(tensor_chw, dim=0)
            elif mode == 'mean':
                feat_map = torch.mean(tensor_chw, dim=0)
            return convert_overlay_heatmap(feat_map, image, alpha)

        if topk <= 0:
            tensor_chw_channel = tensor_chw.shape[0]
            assert tensor_chw_channel in [
                1, 3
            ], ('The input tensor channel dimension must be 1 or 3 '
                'when topk is less than 1, but the channel '
                f'dimension you input is {tensor_chw_channel}, you can use the'
                ' mode parameter or set topk greater than 0 to solve '
                'the error')
            if tensor_chw_channel == 1:
                return convert_overlay_heatmap(tensor_chw[0], image, alpha)
            else:
                tensor_chw = tensor_chw.permute(1, 2, 0).numpy()
                norm_img = cv2.normalize(tensor_chw, None, 0, 255,
                                         cv2.NORM_MINMAX)
                heat_img = np.asarray(norm_img, dtype=np.uint8)
                if image is not None:
                    heat_img = cv2.addWeighted(image, 1 - alpha, heat_img,
                                               alpha, 0)
                return heat_img
        else:
            row, col = arrangement
            channel, height, width = tensor_chw.shape
            assert row * col >= topk
            sum_channel = torch.sum(tensor_chw, dim=(1, 2))
            topk = min(channel, topk)
            _, indices = torch.topk(sum_channel, topk)
            topk_tensor = tensor_chw[indices]
            fig = Figure(frameon=False)
            fig.subplots_adjust(
                left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            dpi = fig.get_dpi()
            fig.set_size_inches((width * col + 1e-2) / dpi,
                                (height * row + 1e-2) / dpi)
            canvas = FigureCanvasAgg(fig)
            fig.subplots_adjust(wspace=0, hspace=0)
            fig.tight_layout(h_pad=0, w_pad=0)

            for i in range(topk):
                axes = fig.add_subplot(row, col, i + 1)
                axes.axis('off')
                axes.imshow(
                    convert_overlay_heatmap(topk_tensor[i], image, alpha))
            s, (width, height) = canvas.print_to_buffer()
            buffer = np.frombuffer(s, dtype='uint8')
            img_rgba = buffer.reshape(height, width, 4)
            rgb, alpha = np.split(img_rgba, [3], axis=2)
            return rgb.astype('uint8')

    def add_config(self, config: Config, **kwargs):
        """Record parameters.

        Args:
            config (Config): The Config object.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_config(config, **kwargs)  # type: ignore

    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record graph data.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_graph(model, data_batch, **kwargs)  # type: ignore

    def add_image(self, name: str, image: np.ndarray, step: int = 0) -> None:
        """Record image.

        Args:
            name (str): The unique identifier for the image to save.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(name, image, step)  # type: ignore

    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record scalar data.

        Args:
            name (str): The unique identifier for the scalar to save.
            value (float, int): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_scalar(name, value, step, **kwargs)  # type: ignore

    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Default to None.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_scalars(  # type: ignore
                scalar_dict, step, file_path, **kwargs)

    def add_datasample(self,
                       name,
                       image: np.ndarray,
                       gt_sample: Optional['BaseDataElement'] = None,
                       pred_sample: Optional['BaseDataElement'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        pass

    def close(self) -> None:
        """close an opened object."""
        plt.close(self.fig_save)
        if self.fig_show is not None:
            plt.close(self.fig_show)
        for vis_backend in self._vis_backends.values():
            vis_backend.close()  # type: ignore

    @classmethod
    def get_instance(cls, name: str, **kwargs) -> 'Visualizer':
        """Make subclass can get latest created instance by
        ``Visualizer.get_current_instance()``.

        Downstream codebase may need to get the latest created instance
        without knowing the specific Visualizer type. For example, mmdetection
        builds visualizer in runner and some component which cannot access
        runner wants to get latest created visualizer. In this case,
        the component does not know which type of visualizer has been built
        and cannot get target instance. Therefore, :class:`Visualizer`
        overrides the :meth:`get_instance` and its subclass will register
        the created instance to :attr:`_instance_dict` additionally.
        :meth:`get_current_instance` will return the latest created subclass
        instance.

        Examples:
            >>> class DetLocalVisualizer(Visualizer):
            >>>     def __init__(self, name):
            >>>         super().__init__(name)
            >>>
            >>> visualizer1 = DetLocalVisualizer.get_instance('name1')
            >>> visualizer2 = Visualizer.get_current_instance()
            >>> visualizer3 = DetLocalVisualizer.get_current_instance()
            >>> assert id(visualizer1) == id(visualizer2) == id(visualizer3)

        Args:
            name (str): Name of instance. Defaults to ''.

        Returns:
            object: Corresponding name instance.
        """
        instance = super().get_instance(name, **kwargs)
        Visualizer._instance_dict[name] = instance
        return instance
