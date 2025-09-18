# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import os.path as osp
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from matplotlib.font_manager import FontProperties

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mmengine.config import Config
from mmengine.dist import master_only
from mmengine.registry import VISBACKENDS, VISUALIZERS
from mmengine.structures import BaseDataElement
from mmengine.utils import ManagerMixin, is_seq_of
from mmengine.visualization.utils import (check_type, check_type_and_length,
                                          color_str2rgb, color_val_matplotlib,
                                          convert_overlay_heatmap,
                                          img_from_canvas, tensor2ndarray,
                                          value2list, wait_continue)
from mmengine.visualization.vis_backend import BaseVisBackend

VisBackendsType = Union[List[Union[List, BaseDataElement]], BaseDataElement,
                        dict, None]


@VISUALIZERS.register_module()
class Visualizer(ManagerMixin):
    """MMEngine provides a Visualizer class that uses the ``Matplotlib``
    library as the backend. It has the following functions:

    - Basic drawing methods

      - draw_bboxes: draw single or multiple bounding boxes
      - draw_texts: draw single or multiple text boxes
      - draw_points: draw single or multiple points
      - draw_lines: draw single or multiple line segments
      - draw_circles: draw single or multiple circles
      - draw_polygons: draw single or multiple polygons
      - draw_binary_masks: draw single or multiple binary masks
      - draw_featmap: draw feature map

    - Basic visualizer backend methods

      - add_configs: write config to all vis storage backends
      - add_graph: write model graph to all vis storage backends
      - add_image: write image to all vis storage backends
      - add_scalar: write scalar to all vis storage backends
      - add_scalars: write scalars to all vis storage backends
      - add_datasample: write datasample to all vis storage \
         backends. The abstract drawing interface used by the user

    - Basic info methods

      - set_image: sets the original image data
      - get_image: get the image data in Numpy format after drawing
      - show: visualization
      - close: close all resources that have been opened
      - get_backend: get the specified vis backend


    All the basic drawing methods support chain calls, which is convenient for
    overlaydrawing and display. Each downstream algorithm library can inherit
    ``Visualizer`` and implement the add_datasample logic. For example,
    ``DetLocalVisualizer`` in MMDetection inherits from ``Visualizer``
    and implements functions, such as visual detection boxes, instance masks,
    and semantic segmentation maps in the add_datasample interface.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        fig_save_cfg (dict): Keyword parameters of figure for saving.
            Defaults to empty dict.
        fig_show_cfg (dict): Keyword parameters of figure for showing.
            Defaults to empty dict.

    Examples:
        >>> # Basic info methods
        >>> vis = Visualizer()
        >>> vis.set_image(image)
        >>> vis.get_image()
        >>> vis.show()

        >>> # Basic drawing methods
        >>> vis = Visualizer(image=image)
        >>> vis.draw_bboxes(np.array([0, 0, 1, 1]), edge_colors='g')
        >>> vis.draw_bboxes(bbox=np.array([[1, 1, 2, 2], [2, 2, 3, 3]]),
        >>>                    edge_colors=['g', 'r'])
        >>> vis.draw_lines(x_datas=np.array([1, 3]),
        >>>                y_datas=np.array([1, 3]),
        >>>                colors='r', line_widths=1)
        >>> vis.draw_lines(x_datas=np.array([[1, 3], [2, 4]]),
        >>>                y_datas=np.array([[1, 3], [2, 4]]),
        >>>                colors=['r', 'r'], line_widths=[1, 2])
        >>> vis.draw_texts(text='MMEngine',
        >>>               position=np.array([2, 2]),
        >>>               colors='b')
        >>> vis.draw_texts(text=['MMEngine','OpenMMLab'],
        >>>                position=np.array([[2, 2], [5, 5]]),
        >>>                colors=['b', 'b'])
        >>> vis.draw_circles(circle_coord=np.array([2, 2]), radius=np.array[1])
        >>> vis.draw_circles(circle_coord=np.array([[2, 2], [3, 5]),
        >>>                  radius=np.array[1, 2], colors=['g', 'r'])
        >>> square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        >>> vis.draw_polygons(polygons=square, edge_colors='g')
        >>> squares = [np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
        >>>            np.array([[0, 0], [50, 0], [50, 50], [0, 50]])]
        >>> vis.draw_polygons(polygons=squares, edge_colors=['g', 'r'])
        >>> vis.draw_binary_masks(binary_mask, alpha=0.6)
        >>> heatmap = vis.draw_featmap(featmap, img,
        >>>                            channel_reduction='select_max')
        >>> heatmap = vis.draw_featmap(featmap, img, channel_reduction=None,
        >>>                            topk=8, arrangement=(4, 2))
        >>> heatmap = vis.draw_featmap(featmap, img, channel_reduction=None,
        >>>                            topk=-1)

        >>> # chain calls
        >>> vis.draw_bboxes().draw_texts().draw_circle().draw_binary_masks()

        >>> # Backend related methods
        >>> vis = Visualizer(vis_backends=[dict(type='LocalVisBackend')],
        >>>                                save_dir='temp_dir')
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> vis.add_config(cfg)
        >>> image=np.random.randint(0, 256, size=(10, 10, 3)).astype(np.uint8)
        >>> vis.add_image('image',image)
        >>> vis.add_scaler('mAP', 0.6)
        >>> vis.add_scalars({'loss': 0.1,'acc':0.8})

        >>> # inherit
        >>> class DetLocalVisualizer(Visualizer):
        >>>      def add_datasample(self,
        >>>                         name,
        >>>                         image: np.ndarray,
        >>>                         gt_sample:
        >>>                             Optional['BaseDataElement'] = None,
        >>>                         pred_sample:
        >>>                             Optional['BaseDataElement'] = None,
        >>>                         draw_gt: bool = True,
        >>>                         draw_pred: bool = True,
        >>>                         show: bool = False,
        >>>                         wait_time: int = 0,
        >>>                         step: int = 0) -> None:
        >>>         pass
    """

    def __init__(
        self,
        name='visualizer',
        image: Optional[np.ndarray] = None,
        vis_backends: VisBackendsType = None,
        save_dir: Optional[str] = None,
        fig_save_cfg=dict(frameon=False),
        fig_show_cfg=dict(frameon=False)
    ) -> None:
        super().__init__(name)
        self._dataset_meta: Optional[dict] = None
        self._vis_backends: Dict[str, BaseVisBackend] = {}

        if vis_backends is None:
            vis_backends = []

        if isinstance(vis_backends, (dict, BaseVisBackend)):
            vis_backends = [vis_backends]  # type: ignore

        if not is_seq_of(vis_backends, (dict, BaseVisBackend)):
            raise TypeError('vis_backends must be a list of dicts or a list '
                            'of BaseBackend instances')
        if save_dir is not None:
            save_dir = osp.join(save_dir, 'vis_data')

        for vis_backend in vis_backends:  # type: ignore
            name = None
            if isinstance(vis_backend, dict):
                name = vis_backend.pop('name', None)
                vis_backend.setdefault('save_dir', save_dir)
                vis_backend = VISBACKENDS.build(vis_backend)

            # If vis_backend requires `save_dir` (with no default value)
            # but is initialized with None, then don't add this
            # vis_backend to the visualizer.
            save_dir_arg = inspect.signature(
                vis_backend.__class__.__init__).parameters.get('save_dir')
            if (save_dir_arg is not None
                    and save_dir_arg.default is save_dir_arg.empty
                    and getattr(vis_backend, '_save_dir') is None):
                warnings.warn(f'Failed to add {vis_backend.__class__}, '
                              'please provide the `save_dir` argument.')
                continue

            type_name = vis_backend.__class__.__name__
            name = name or type_name

            if name in self._vis_backends:
                raise RuntimeError(f'vis_backend name {name} already exists')
            self._vis_backends[name] = vis_backend  # type: ignore

        self.fig_save = None
        self.fig_save_cfg = fig_save_cfg
        self.fig_show_cfg = fig_show_cfg

        (self.fig_save_canvas, self.fig_save,
         self.ax_save) = self._initialize_fig(fig_save_cfg)
        self.dpi = self.fig_save.get_dpi()

        if image is not None:
            self.set_image(image)

    @property  # type: ignore
    @master_only
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter  # type: ignore
    @master_only
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the Visualizer."""
        self._dataset_meta = dataset_meta

    @master_only
    def show(self,
             drawn_img: Optional[np.ndarray] = None,
             win_name: str = 'image',
             wait_time: float = 0.,
             continue_key: str = ' ',
             backend: str = 'matplotlib') -> None:
        """Show the drawn image.

        Args:
            drawn_img (np.ndarray, optional): The image to show. If drawn_img
                is None, it will show the image got by Visualizer. Defaults
                to None.
            win_name (str):  The image title. Defaults to 'image'.
            wait_time (float): Delay in seconds. 0 is the special
                value that means "forever". Defaults to 0.
            continue_key (str): The key for users to continue. Defaults to
                the space key.
            backend (str): The backend to show the image. Defaults to
                'matplotlib'. `New in version 0.7.3.`
        """
        if backend == 'matplotlib':
            import matplotlib.pyplot as plt
            is_inline = 'inline' in plt.get_backend()
            img = self.get_image() if drawn_img is None else drawn_img
            self._init_manager(win_name)
            fig = self.manager.canvas.figure
            # remove white edges by set subplot margin
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            fig.clear()
            ax = fig.add_subplot()
            ax.axis(False)
            ax.imshow(img)
            self.manager.canvas.draw()

            # Find a better way for inline to show the image
            if is_inline:
                return fig
            wait_continue(fig, timeout=wait_time, continue_key=continue_key)
        elif backend == 'cv2':
            # Keep images are shown in the same window, and the title of window
            # will be updated with `win_name`.
            cv2.namedWindow(winname=f'{id(self)}')
            cv2.setWindowTitle(f'{id(self)}', win_name)
            cv2.imshow(
                str(id(self)),
                self.get_image() if drawn_img is None else drawn_img)
            cv2.waitKey(int(np.ceil(wait_time * 1000)))
        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')

    @master_only
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

    @master_only
    def get_image(self) -> np.ndarray:
        """Get the drawn image. The format is RGB.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        assert self._image is not None, 'Please set image using `set_image`'
        return img_from_canvas(self.fig_save_canvas)  # type: ignore

    def _initialize_fig(self, fig_cfg) -> tuple:
        """Build figure according to fig_cfg.

        Args:
            fig_cfg (dict): The config to build figure.

        Returns:
             tuple: build canvas figure and axes.
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(**fig_cfg)
        ax = fig.add_subplot()
        ax.axis(False)

        # remove white edges by set subplot margin
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        canvas = FigureCanvasAgg(fig)
        return canvas, fig, ax

    def _init_manager(self, win_name: str) -> None:
        """Initialize the matplot manager.

        Args:
            win_name (str): The window name.
        """
        from matplotlib.figure import Figure
        from matplotlib.pyplot import new_figure_manager
        if getattr(self, 'manager', None) is None:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)

        try:
            self.manager.set_window_title(win_name)
        except Exception:
            self.manager = new_figure_manager(
                num=1, FigureClass=Figure, **self.fig_show_cfg)
            self.manager.set_window_title(win_name)

    @master_only
    def get_backend(self, name) -> 'BaseVisBackend':
        """Get vis backend by name.

        Args:
            name (str): The name of vis backend

        Returns:
             BaseVisBackend: The vis backend.
        """
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

    @master_only
    def draw_points(self,
                    positions: Union[np.ndarray, torch.Tensor],
                    colors: Union[str, tuple, List[str],
                                  List[tuple]] = (0, 255, 0),
                    marker: Optional[str] = None,
                    sizes: Optional[Union[np.ndarray, torch.Tensor]] = None,
                    backend: str = 'matplotlib'):
        """Draw single or multiple points on the plot or image.

        Args:
            positions (Union[np.ndarray, torch.Tensor]):
                Positions of the points to be drawn.
                It should be a 2D array-like object with shape (N, 2),
                where N is the number of points.
            colors (Union[str, tuple, List[str], List[tuple]]):
                The colors of the points.
                It can have the same length as the number of points
                or just a single value.
                If a single value is provided,
                all points will have the same color.
                Colors can be specified using named colors
                from the matplotlib color library.
                See
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details.
                When using the 'cv2' backend,
                colors should be a tuple of integers (e.g., (0, 255, 0)).
                Defaults to 'g' (green).
            marker (str, optional):
                The marker style to be used for the points.
                Refer to the matplotlib markers documentation
                for available marker styles.
                Defaults to None, which results in circular markers.
            sizes (Optional[Union[np.ndarray, torch.Tensor]]):
                The size of the markers.
                It can be a 1D array-like object with length N,
                specifying the size for each point,
                or a single value for a uniform marker size across all points.
                Defaults to None, which uses the default marker size.
            backend (str): The backend to use for drawing the points.
                Supported backends are 'matplotlib' and 'cv2' (OpenCV).
                Defaults to 'matplotlib'.

        Raises:
            ValueError:
                If the shape of the `positions` array is not (N, 2),
                where N is the number of points.
            ValueError:
                If the `backend` argument is not one of 'matplotlib' or 'cv2'.

        Returns:
            self: Returns the instance of the class, allowing method chaining.

        Note:
            - When using the 'matplotlib' backend,
              the points are drawn on a plot.
            - When using the 'cv2' backend, the points are drawn on an image.
            - The 'cv2' backend expects the positions to be in
                normalized coordinates,
                with values ranging from 0 to 1,
                representing the relative position within the image.
        """

        check_type('positions', positions, (np.ndarray, torch.Tensor))
        positions = tensor2ndarray(positions)

        if len(positions.shape) == 1:
            positions = positions[None]
        assert positions.shape[-1] == 2, (
            'The shape of `positions` should be (N, 2), '
            f'but got {positions.shape}')
        if backend == 'matplotlib':
            colors = color_val_matplotlib(colors)  # type: ignore
            self.ax_save.scatter(
                positions[:, 0],
                positions[:, 1],
                c=colors,
                s=sizes,
                marker=marker)
            return self

        elif backend == 'cv2':
            if marker is not None:
                warnings.warn(
                    "The 'marker' argument is not applicable in the 'cv2'"
                    'backend and will be ignored.', UserWarning)
            positions[:, 0] *= self._image.shape[1]
            positions[:, 1] *= self._image.shape[0]
            for i in range(len(positions)):
                x = int(positions[i, 0])
                y = int(positions[i, 1])
                if sizes is not None:
                    if isinstance(sizes, (np.ndarray, torch.Tensor)):
                        size = int(sizes[i])
                    else:
                        size = int(sizes[i])
                else:
                    size = 0  # Use default marker size
                if colors[i] == 'g':
                    colors[i] = (0, 255, 0)  # type: ignore
                elif colors[i] == 'r':
                    colors[i] = (0, 0, 255)  # type: ignore
                elif colors[i] == 'b':
                    colors[i] = (255, 0, 0)  # type: ignore
                cv2.circle(self._image, (x, y), size, colors[i], -1)

        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')

    @master_only
    def draw_texts(self,
                   texts: Union[str, List[str]],
                   positions: Union[np.ndarray, torch.Tensor],
                   font_sizes: Optional[Union[int, List[int]]] = 1,
                   colors: Union[str, tuple, List[str],
                                 List[tuple]] = (0, 255, 0),
                   vertical_alignments: Union[str, List[str]] = 'top',
                   horizontal_alignments: Union[str, List[str]] = 'left',
                   font_families: Union[str, List[str]] = 'sans-serif',
                   bboxes: Optional[Union[dict, List[dict]]] = None,
                   font_properties: Optional[Union[
                       'FontProperties', List['FontProperties']]] = None,
                   backend: str = 'matplotlib') -> 'Visualizer':
        """Draw single or multiple text boxes.

        Args:
            texts (Union[str, List[str]]):
                Texts to draw.
            positions (Union[np.ndarray, torch.Tensor]):
                The positions to draw the texts.
                It should have the same length as `texts`,
                and each dimension should contain the x and y coordinates.
            font_sizes (Optional[Union[int, List[int]]]):
                The font sizes of the texts.
                `font_sizes` can have the same length as `texts`
                or just a single value.
                If `font_sizes` is a single value, all the texts will
                have the same font size. Defaults to None.
            colors (Union[str, tuple, List[str], List[tuple]]):
                The colors of the texts.
                `colors` can have the same length as `texts`
                or just a single value.
                If `colors` is a single value, all the texts will
                have the same colors.
                When using the 'matplotlib' backend, colors can be specified
                using named colors from the matplotlib color library.
                See
                https://matplotlib.org/stable/gallery/color/named_colors.html for more details.
                Defaults to 'g'.
            vertical_alignments (Union[str, List[str]]):
                The vertical alignment of the texts.
                `vertical_alignments` can have the same length
                as `texts` or just a single value.
                If `vertical_alignments` is a single value,
                all the texts will have the same vertical alignment.
                Valid values for vertical alignment
                are 'center', 'top', 'bottom', or 'baseline'.
                Defaults to 'top'.
            horizontal_alignments (Union[str, List[str]]):
                The horizontal alignment of the texts.
                `horizontal_alignments` can have the same length
                as `texts` or just a single value.
                If `horizontal_alignments` is a single value,
                all the texts will have the same horizontal alignment.
                Valid values for horizontal alignment
                are 'center', 'right', or 'left'. Defaults to 'left'.
            font_families (Union[str, List[str]]):
                The font family of the texts.
                `font_families` can have the same length as `texts`
                or just a single value.
                If `font_families` is a single value, all the texts will
                have the same font family.
                Valid font family values are 'serif', 'sans-serif',
                'cursive', 'fantasy', or 'monospace'.
                Defaults to 'sans-serif'.
            bboxes (Optional[Union[dict, List[dict]]]):
                The bounding box of the texts.
                If `bboxes` is None, there will be no bounding box
                around the texts.
                `bboxes` can have the same length as `texts`
                or just a single value.
                If `bboxes` is a single value,
                all the texts will have the same bounding box.
                When using the 'matplotlib' backend,
                `bboxes` can be a dictionary with properties for
                the FancyBboxPatch.
                See matplotlib.patches.FancyBboxPatch for more details.
                Defaults to None.
            font_properties (Optional[Union[FontProperties, List[FontProperties]]]):
                The font properties of the texts.
                `font_properties` can have the same length as `texts`
                or just a single value.
                If `font_properties` is a single value,
                all the texts will
                have the same font properties.
                When using the 'matplotlib' backend, `font_properties`
                should be a matplotlib.font_manager.FontProperties object.
                This can be used to specify
                custom font files for drawing Chinese characters, for example.
                Defaults to None.
            backend (str):
                The backend to use for drawing the texts. Supported options are 'matplotlib' and 'cv2'.
                Defaults to 'matplotlib'.

        Returns:
            self (Visualizer):
                Returns the instance of the class,
                allowing method chaining.

        Raises:
            ValueError:
                If the `backend` argument is not
                one of the supported backends: 'matplotlib' or 'cv2'.
            AssertionError:
                If the shapes of the input arrays are not valid.

        Notes:
            - When using the 'matplotlib' backend,
              the drawn texts may not appear within the image boundaries
              if they extend beyond the bounds of the image.
            - The `positions` argument should be provided
              as a numpy array or tensor with shape (N, 2),
              where N is the number of texts and
              each position is represented as (x, y) coordinates.
            - The `font_sizes`, `colors`, `vertical_alignments`,
              `horizontal_alignments`, `font_families`,
              `bboxes`, and `font_properties` arguments
              can be specified as a single value or a list
              to provide different values for each text.

        Warnings:
            - When using the 'matplotlib' backend,
              a warning will be displayed if any of the texts extend
              beyond the bounds of the image,
              indicating that the drawn text may not appear within the image
              boundaries.
        """  # noqa: E501
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
        check_type_and_length('font_sizes', font_sizes, (int, float, list),
                              num_text)

        check_type_and_length('colors', colors, (str, tuple, list), num_text)
        check_type_and_length('vertical_alignments', vertical_alignments,
                              (str, list), num_text)
        check_type_and_length('horizontal_alignments', horizontal_alignments,
                              (str, list), num_text)
        check_type_and_length('font_families', font_families, (str, list),
                              num_text)

        if backend == 'matplotlib':
            from matplotlib.font_manager import FontProperties
            if font_sizes is None:
                font_sizes = self._default_font_size
            font_sizes = value2list(font_sizes, (int, float), num_text)

            colors = value2list(colors, (str, tuple), num_text)
            colors = color_val_matplotlib(colors)  # type: ignore

            vertical_alignments = value2list(vertical_alignments, str,
                                             num_text)

            horizontal_alignments = value2list(horizontal_alignments, str,
                                               num_text)

            font_families = value2list(font_families, str, num_text)

            if font_properties is None:
                font_properties = [None
                                   for _ in range(num_text)]  # type: ignore
            else:
                check_type_and_length('font_properties', font_properties,
                                      (FontProperties, list), num_text)
                font_properties = value2list(font_properties, FontProperties,
                                             num_text)

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
                    fontproperties=font_properties[i],
                    color=colors[i])
            return self

        elif backend == 'cv2':
            if font_sizes is None:
                font_sizes = [16] * num_text
            elif isinstance(font_sizes, int):
                font_sizes = [font_sizes] * num_text
            else:
                font_sizes = font_sizes[:num_text]

            if colors is None:
                colors = [(0, 0, 255)] * num_text

            if vertical_alignments is None:
                vertical_alignments = ['bottom'] * num_text

            if horizontal_alignments is None:
                horizontal_alignments = [
                    'left'
                ] * num_text  # Default horizontal alignment is left

            if font_families is None or font_families == 'sans-serif':
                font_families = [
                    cv2.FONT_HERSHEY_SIMPLEX
                ] * num_text  # Default font family is HERSHEY_SIMPLEX

            for i in range(num_text):
                position = tuple(positions[i])
                text = texts[i]
                font_size = font_sizes[i]
                if colors[i] == 'r':
                    color = (0, 0, 255)
                elif colors[i] == 'g':
                    color = (0, 255, 0)
                elif colors[i] == 'b':
                    color = (255, 0, 0)
                else:
                    color = colors[i]  # type: ignore
                vertical_alignment = vertical_alignments[i]
                horizontal_alignment = horizontal_alignments[i]
                font_family = font_families[i]

                print(f'color: {color}')
                # Calculate text size to determine the bounding box
                (text_width,
                 text_height), _ = cv2.getTextSize(text, font_family,
                                                   font_size, 1)

                # Adjust position based on alignment
                if vertical_alignment == 'top':
                    position = (position[0], position[1] + text_height)
                elif vertical_alignment == 'center':
                    position = (position[0], position[1] + text_height // 2)

                if horizontal_alignment == 'right':
                    position = (position[0] - text_width, position[1])
                elif horizontal_alignment == 'center':
                    position = (position[0] - text_width // 2, position[1])

                # Draw the text on the image
                cv2.putText(self._image, text, position, font_family,
                            font_size, color, 1, cv2.LINE_AA)

            return self

        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')

    @master_only
    def draw_lines(self,
                   x_datas: Union[np.ndarray, torch.Tensor],
                   y_datas: Union[np.ndarray, torch.Tensor],
                   colors: Union[str, tuple, List[str],
                                 List[tuple]] = (0, 255, 0),
                   line_styles: Union[str, List[str]] = '-',
                   line_widths: Union[Union[int, float],
                                      List[Union[int, float]], int] = 2,
                   backend: str = 'matplotlib') -> 'Visualizer':
        """Draw single or multiple line segments on the plot or image.

        Args:
            x_datas (Union[np.ndarray, torch.Tensor]):
                The x-coordinate of the start and end points
                of each line segment.
            y_datas (Union[np.ndarray, torch.Tensor]):
                The y-coordinate of the start and end points
                of each line segment.
            colors (Union[str, tuple, List[str], List[tuple]]):
                The colors of the lines.
                `colors` can have the same length as the number of lines
                or just a single value.
                If a single value is provided,
                all lines will have the same color.
                When using the 'matplotlib' backend,
                colors can be specified using named colors
                from the matplotlib color library.
                See
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details.
                When using the 'cv2' backend,
                colors should be a tuple of integers (e.g., (0, 255, 0)).
                Defaults to 'g' (green).
            line_styles (Union[str, List[str]]): The line style of the lines.
                `line_styles` can have the same length as the number of lines
                or just a single value.
                If a single value is provided,
                all lines will have the same line style.
                Refer to the matplotlib documentation
                for available line styles.
                Defaults to '-' (solid line).
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The line width of the lines.
                `line_widths` can have the same length as the number of lines
                or just a single value.
                If a single value is provided,
                all lines will have the same line width.
                Defaults to 2.
            backend (str): The backend to use for drawing the lines.
                Supported backends are 'matplotlib' and 'cv2' (OpenCV).
                Defaults to 'matplotlib'.

        Raises:
            TypeError: If the types of `line_widths` are invalid.
            ValueError: If the shapes of `x_datas` and `y_datas` do not match.
            ValueError: If the `backend` argument is not one of 'matplotlib'
                        or 'cv2'.

        Returns:
            self: Returns the instance of the class, allowing method chaining.

        Note:
            - When using the 'matplotlib' backend,
              the lines are drawn on a plot.
            - When using the 'cv2' backend, the lines are drawn on an image.
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
        lines = np.concatenate(
            (x_datas.reshape(-1, 2, 1), y_datas.reshape(-1, 2, 1)), axis=-1)
        if not self._is_posion_valid(lines):
            warnings.warn(
                'Warning: The line is out of bounds,'
                ' the drawn line may not be in the image', UserWarning)
        if backend == 'matplotlib':
            from matplotlib.collections import LineCollection
            colors = color_val_matplotlib(colors)  # type: ignore
            line_collect = LineCollection(
                lines.tolist(),
                colors=colors,
                linestyles=line_styles,
                linewidths=line_widths)
            self.ax_save.add_collection(line_collect)
            return self

        elif backend == 'cv2':
            image_with_lines = self._image.copy(
            )  # Create a copy of the original image

            for i, line in enumerate(lines):
                start = (int(line[0, 0]), int(line[0, 1]))
                end = (int(line[1, 0]), int(line[1, 1]))

                # Get the line width for the current line segment
                if isinstance(line_widths, int):
                    current_line_width = line_widths
                elif isinstance(line_widths, list):
                    if isinstance(line_widths[0], int):
                        current_line_width = int(
                            line_widths[i])  # Use the width at index i
                    elif isinstance(line_widths[0], list):
                        current_line_width = [
                            int(width) for width in line_widths[i]
                        ]
                    else:
                        raise TypeError('Invalid type for line_widths')
                else:
                    raise TypeError('Invalid type for line_widths')
                # Draw the line segment

                if colors == 'r':
                    colors = (0, 0, 255)
                elif colors == 'g':
                    colors = (0, 255, 0)
                elif colors == 'b':
                    colors = (255, 0, 0)
                cv2.line(
                    image_with_lines,
                    start,
                    end,
                    color=colors,
                    thickness=current_line_width)

                self._image = image_with_lines

            return self

        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')

    @master_only
    def draw_circles(self,
                     center: Union[np.ndarray, torch.Tensor],
                     radius: Union[np.ndarray, torch.Tensor],
                     edge_colors: Union[str, tuple, List[str],
                                        List[tuple]] = (0, 255, 0),
                     line_styles: Union[str, List[str]] = '-',
                     line_widths: Union[Union[int, float],
                                        List[Union[int, float]]] = 2,
                     face_colors: Union[str, tuple, List[str],
                                        List[tuple]] = (0, 255, 0),
                     alpha: Union[float, int] = 0.8,
                     backend: str = 'matplotlib') -> 'Visualizer':
        """Draw single or multiple circles on the plot or image.

        Args:
            center (Union[np.ndarray, torch.Tensor]):
                The x and y coordinates of the center of each circle.
            radius (Union[np.ndarray, torch.Tensor]):
                The radius of each circle.
            edge_colors (Union[str, tuple, List[str], List[tuple]]):
                The edge colors of the circles.
                `edge_colors` can have the same length as the number of circles
                or just a single value.
                If a single value is provided,
                all circles will have the same edge color.
                When using the 'matplotlib' backend,
                colors can be specified using named colors
                from the matplotlib color library.
                See
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details.
                When using the 'cv2' backend,
                colors should be a tuple of integers (e.g., (0, 255, 0))
                since 'cv2' does not support named colors.
                Defaults to 'g' (green).
            line_styles (Union[str, List[str]]):
                The line style of the circles' edges.
                `line_styles` can have the same length as the number of circles
                or just a single value.
                If a single value is provided,
                all circles will have the same line style.
                Refer to the matplotlib documentation
                for available line styles.
                Defaults to '-' (solid line).
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The line width of the circles' edges.
                `line_widths` can have the same length as the number of circles
                or just a single value.
                If a single value is provided,
                all circles will have the same line width.
                Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors of the circles.
                `face_colors` can have the same length as the number of circles
                or just a single value.
                If a single value is provided,
                all circles will have the same face color.
                When using the 'matplotlib' backend,
                colors can be specified using named colors from the matplotlib
                color library.
                See
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details.
                When using the 'cv2' backend,
                colors should be a tuple of integers (e.g., (0, 255, 0))
                since 'cv2' does not support named colors.
                Defaults to 'none'.
            alpha (Union[float, int]):
                The transparency of the circles.
                Valid values are between 0 and 1.
                Defaults to 0.8.
            backend (str): The backend to use for drawing the circles.
                Supported backends are 'matplotlib' and 'cv2' (OpenCV).
                Defaults to 'matplotlib'.

        Raises:
            TypeError: If the types of `center` or `radius` are invalid.
            ValueError: If the shapes of `center` and `radius` do not match.
            ValueError: If the `backend` argument is not one of 'matplotlib'
                        or 'cv2'.

        Returns:
            self: Returns the instance of the class,
                  allowing method chaining.

        Note:
            - When using the 'matplotlib' backend,
              the circles are drawn on a plot.
            - When using the 'cv2' backend,
              the circles are drawn on an image.

        Warning:
            - If a circle is out of bounds,
              the drawn circle may not be fully visible in the image.
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
        if backend == 'matplotlib':
            from matplotlib.collections import PatchCollection
            from matplotlib.patches import Circle
            edge_colors = color_val_matplotlib(edge_colors)  # type: ignore
            face_colors = color_val_matplotlib(face_colors)  # type: ignore
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

        elif backend == 'cv2':
            if not isinstance(edge_colors, list):
                edge_colors = [edge_colors] * len(center)  # type: ignore
            if not isinstance(face_colors, list):
                face_colors = [face_colors] * len(center)  # type: ignore
            for i in range(len(center)):
                x, y = int(center[i][0]), int(center[i][1])
                r = int(radius[i])

                # Get the edge color and face color for the current circle
                current_edge_color = edge_colors[i]
                current_face_color = face_colors[i]

                # Get the line width for the current circle
                current_line_width = line_widths[i] if isinstance(
                    line_widths, list) else int(line_widths)

                # Draw the circle for the edge
                if current_edge_color == 'g':
                    current_edge_color = (0, 255, 0)
                elif current_edge_color == 'r':
                    current_edge_color = (0, 0, 255)
                elif current_edge_color == 'b':
                    current_edge_color = (255, 0, 0)

                cv2.circle(
                    self._image, (x, y),
                    r,
                    current_edge_color,
                    thickness=current_line_width)

                # Draw the circle for the face
                cv2.circle(self._image, (x, y), abs(r - current_line_width),
                           current_face_color, -1)

            return self

        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')

    @master_only
    def draw_bboxes(self,
                    bboxes: Union[np.ndarray, torch.Tensor],
                    edge_colors: Union[str, tuple, List[str],
                                       List[tuple]] = (0, 255, 0),
                    line_styles: Union[str, List[str]] = '-',
                    line_widths: Union[Union[int, float],
                                       List[Union[int, float]]] = 2,
                    face_colors: Union[str, tuple, List[str],
                                       List[tuple]] = (0, 255, 0),
                    alpha: Union[int, float] = 0.8,
                    backend: str = 'matplotlib') -> 'Visualizer':
        """Draw single or multiple bounding boxes.

        Args:
            bboxes (Union[np.ndarray, torch.Tensor]):
                The bounding boxes to draw.
                The format of each bbox should be (x1, y1, x2, y2).
            edge_colors (Union[str, tuple, List[str], List[tuple]]):
                The edge colors of the bounding boxes.
                `edge_colors` can have the same length
                as the number of bboxes or just a single value.
                If a single value is provided,
                all bounding boxes will have the same edge color.
                When using the 'matplotlib' backend,
                colors can be specified using named colors
                from the matplotlib color library.
                See
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details.
                When using the 'cv2' backend,
                colors should be a tuple of integers (e.g., (0, 255, 0)).
                Defaults to 'g' (green).
            line_styles (Union[str, List[str]]):
                The line style of the bounding box edges.
                `line_styles` can have the same length
                as the number of bboxes or just a single value.
                If a single value is provided,
                all bounding boxes will have the same line style.
                Refer to the matplotlib documentation
                for available line styles.
                Defaults to '-' (solid line).
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The line width of the bounding box edges.
                `line_widths` can have the same length
                as the number of bboxes or just a single value.
                If a single value is provided,
                all bounding boxes will have the same line width.
                Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors of the bounding boxes.
                `face_colors` can have the same length
                as the number of bboxes or just a single value.
                If a single value is provided,
                all bounding boxes will have the same face color.
                When using the 'matplotlib' backend,
                colors can be specified using named colors
                from the matplotlib color library.
                See
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details.
                When using the 'cv2' backend,
                colors should be a tuple of integers (e.g., (0, 255, 0)).
                Defaults to 'none'.
            alpha (Union[int, float]):
                The transparency of the bounding boxes.
                Valid values are between 0 and 1.
                Defaults to 0.8.
            backend (str):
                The backend to use for drawing the bounding boxes.
                Supported backends are 'matplotlib' and 'cv2' (OpenCV).
                Defaults to 'matplotlib'.

        Returns:
            self: Returns the instance of the class,
                  allowing method chaining.

        Note:
            - When using the 'matplotlib' backend,
              the bounding boxes are drawn on a plot.
            - When using the 'cv2' backend,
              the bounding boxes are drawn on an image.

        Warning:
            - If a bounding box is out of bounds,
              the drawn bounding box may not be fully visible in the image.

        Raises:
            ValueError: If the `backend` argument is not one of 'matplotlib'
                        or 'cv2'.
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
        if backend == 'matplotlib':
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

        elif backend == 'cv2':
            if not isinstance(edge_colors, list):
                edgeColors = [edge_colors] * len(bboxes)
            if not isinstance(face_colors, list):
                faceColors = [face_colors] * len(bboxes)
            if not isinstance(line_widths, list):
                line_widths = [line_widths] * len(bboxes)

            for points, edge_color, face_color, line_width in zip(
                    bboxes, edgeColors, faceColors, line_widths):
                start = (int(points[0]), int(points[1]))
                end = (int(points[2]), int(points[3]))
                cv2.rectangle(self._image, start, end, face_color,
                              -1)  # Fill the rectangle
                cv2.rectangle(self._image, start, end, edge_color,
                              line_width)  # Outline the rectangle

            return self

        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')

    @master_only
    def draw_polygons(self,
                      polygons: Union[Union[np.ndarray, torch.Tensor],
                                      List[Union[np.ndarray, torch.Tensor]]],
                      edge_colors: Union[str, tuple, List[str],
                                         List[tuple]] = (0, 255, 0),
                      line_styles: Union[str, List[str]] = '-',
                      line_widths: Union[Union[int, float],
                                         List[Union[int, float]]] = 2,
                      face_colors: Union[str, tuple, List[str],
                                         List[tuple]] = (0, 255, 0),
                      alpha: Union[int, float] = 0.8,
                      backend: str = 'matplotlib') -> 'Visualizer':
        """Draw single or multiple polygons.

        Args:
            polygons (Union[Union[np.ndarray, torch.Tensor],
                      List[Union[np.ndarray, torch.Tensor]]]):
                The polygons to draw.
                Each polygon should be a numpy array
                or tensor of shape (N, 2),
                where N is the number of vertices
                and each vertex is represented as (x, y) coordinates.
            edge_colors (Union[str, tuple, List[str], List[tuple]]):
                The colors of the polygon edges.
                `edge_colors` can have the same length
                as the number of polygons
                or just a single value. If a single value is provided,
                all polygons will have the same edge color.
                When using the 'matplotlib' backend,
                colors can be specified using named colors from the matplotlib
                color library.
                See
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details.
                When using the 'cv2' backend,
                colors should be specified as a tuple of integers
                (e.g., (0, 255, 0)).
                Defaults to 'g' (green).
            line_styles (Union[str, List[str]]):
                The line style of the polygon edges.
                `line_styles` can have the same length
                as the number of polygons
                or just a single value.
                If a single value is provided,
                all polygons will have the same line style.
                Refer to the matplotlib documentation
                for available line styles.
                Defaults to '-' (solid line).
            line_widths (Union[Union[int, float], List[Union[int, float]]]):
                The line width of the polygon edges.
                `line_widths` can have the same length as the number
                of polygons
                or just a single value.
                If a single value is provided,
                all polygons will have the same line width.
                Defaults to 2.
            face_colors (Union[str, tuple, List[str], List[tuple]]):
                The face colors of the polygons.
                `face_colors` can have the same length as
                the number of polygons
                or just a single value.
                If a single value is provided,
                all polygons will have the same face color.
                When using the 'matplotlib' backend,
                colors can be specified using named colors from the matplotlib
                color library.
                See
                https://matplotlib.org/stable/gallery/color/named_colors.html
                for more details.
                When using the 'cv2' backend,
                colors should be specified as a tuple of integers
                (e.g., (0, 255, 0)).
                Defaults to 'none'.
            alpha (Union[int, float]):
                The transparency of the polygons.
                Valid values are between 0 and 1.
                Defaults to 0.8.
            backend (str):
                The backend to use for drawing the polygons.
                Supported backends are 'matplotlib' and 'cv2' (OpenCV).
                Defaults to 'matplotlib'.

        Returns:
            self: Returns the instance of the class, allowing method chaining.

        Notes:
            - The `polygons` argument should be provided
              as a list of numpy arrays or tensors,
              even when drawing
              a single polygon.
              If a single numpy array or tensor is provided,
              it will be converted to a list
              internally to handle multiple polygons.
            - The shape of each polygon in `polygons` should be (N, 2),
              where N is the number of vertices and
              each vertex is represented as (x, y) coordinates.

        Warnings:
            - If any of the polygons extend beyond the bounds of the image,
              a warning will be displayed indicating
              that the drawn polygon may not appear within
              the image boundaries.

        Raises:
            - ValueError:
              If the `backend` argument is not one of the supported backends:
              'matplotlib' or 'cv2'.
            - AssertionError:
              If the shapes of the polygons in `polygons` are not valid.
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
        if backend == 'matplotlib':
            from matplotlib.collections import PolyCollection
            edge_colors = color_val_matplotlib(edge_colors)  # type: ignore
            face_colors = color_val_matplotlib(face_colors)  # type: ignore
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

        elif backend == 'cv2':
            if isinstance(line_widths, (int, float)):
                line_widths = [line_widths]
            else:
                line_widths = line_widths

            if isinstance(edge_colors, (tuple)):
                edge_colors = [edge_colors]
            else:
                edge_colors = edge_colors

            if isinstance(face_colors, (tuple)):
                face_colors = [face_colors]
            else:
                face_colors = face_colors

            if len(edge_colors) != len(polygons):
                edge_colors = edge_colors * len(polygons)

            if len(face_colors) != len(polygons):
                face_colors = face_colors * len(polygons)

            if len(line_widths) != len(polygons):
                line_widths = line_widths * len(polygons)

            assert len(polygons) == len(
                edge_colors), 'Number of polygons and edge colors must match'
            assert len(polygons) == len(
                line_widths), 'Number of polygons and line widths must match'
            if face_colors is not None:
                assert len(polygons) == len(
                    face_colors
                ), 'Number of polygons and face colors must match'

            for i, polygon in enumerate(polygons):
                assert polygon.shape[
                    1] == 2, f'Invalid shape for polygon {i + 1}'
                polygon = polygon.reshape((-1, 1, 2)).astype(np.int32)
                if face_colors is not None:
                    if edge_colors[i] == 'r':
                        edge_colors[i] = (0, 0, 255)  # type: ignore
                    elif edge_colors[i] == 'g':
                        edge_colors[i] = (0, 255, 0)  # type: ignore
                    elif edge_colors[i] == 'b':
                        edge_colors[i] = (255, 0, 0)  # type: ignore
                    cv2.fillPoly(self._image, [polygon], face_colors[i])
                cv2.polylines(self._image, [polygon], True, edge_colors[i],
                              line_widths[i])

            return self

        else:
            raise ValueError('backend should be "matplotlib" or "cv2", '
                             f'but got {backend} instead')

    @master_only
    def draw_binary_masks(
            self,
            binary_masks: Union[np.ndarray, torch.Tensor],
            colors: Union[str, tuple, List[str], List[tuple]] = 'g',
            alphas: Union[float, List[float]] = 0.8) -> 'Visualizer':
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
            alphas (Union[int, List[int]]): The transparency of masks.
                Defaults to 0.8.
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
                                1:], '`binary_masks` must have ' \
                                     'the same shape with image'
        binary_mask_len = binary_masks.shape[0]

        check_type_and_length('colors', colors, (str, tuple, list),
                              binary_mask_len)
        colors = value2list(colors, (str, tuple), binary_mask_len)
        colors = [
            color_str2rgb(color) if isinstance(color, str) else color
            for color in colors
        ]
        for color in colors:
            assert len(color) == 3
            for channel in color:
                assert 0 <= channel <= 255  # type: ignore

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
    @master_only
    def draw_featmap(featmap: torch.Tensor,
                     overlaid_image: Optional[np.ndarray] = None,
                     channel_reduction: Optional[str] = 'squeeze_mean',
                     topk: int = 20,
                     arrangement: Tuple[int, int] = (4, 5),
                     resize_shape: Optional[tuple] = None,
                     alpha: float = 0.5) -> np.ndarray:
        """Draw featmap.

        - If `overlaid_image` is not None, the final output image will be the
          weighted sum of img and featmap.

        - If `resize_shape` is specified, `featmap` and `overlaid_image`
          are interpolated.

        - If `resize_shape` is None and `overlaid_image` is not None,
          the feature map will be interpolated to the spatial size of the image
          in the case where the spatial dimensions of `overlaid_image` and
          `featmap` are different.

        - If `channel_reduction` is "squeeze_mean" and "select_max",
          it will compress featmap to single channel image and weighted
          sum to `overlaid_image`.

        - If `channel_reduction` is None

          - If topk <= 0, featmap is assert to be one or three
            channel and treated as image and will be weighted sum
            to ``overlaid_image``.
          - If topk > 0, it will select topk channel to show by the sum of
            each channel. At the same time, you can specify the `arrangement`
            to set the window layout.

        Args:
            featmap (torch.Tensor): The featmap to draw which format is
                (C, H, W).
            overlaid_image (np.ndarray, optional): The overlaid image.
                Defaults to None.
            channel_reduction (str, optional): Reduce multiple channels to a
                single channel. The optional value is 'squeeze_mean'
                or 'select_max'. Defaults to 'squeeze_mean'.
            topk (int): If channel_reduction is not None and topk > 0,
                it will select topk channel to show by the sum of each channel.
                if topk <= 0, tensor_chw is assert to be one or three.
                Defaults to 20.
            arrangement (Tuple[int, int]): The arrangement of featmap when
                channel_reduction is None and topk > 0. Defaults to (4, 5).
            resize_shape (tuple, optional): The shape to scale the feature map.
                Defaults to None.
            alpha (Union[int, List[int]]): The transparency of featmap.
                Defaults to 0.5.

        Returns:
            np.ndarray: RGB image.
        """
        import matplotlib.pyplot as plt
        assert isinstance(featmap,
                          torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                          f' but got {type(featmap)}')
        assert featmap.ndim == 3, f'Input dimension must be 3, ' \
                                  f'but got {featmap.ndim}'
        featmap = featmap.detach().cpu()

        if overlaid_image is not None:
            if overlaid_image.ndim == 2:
                overlaid_image = cv2.cvtColor(overlaid_image,
                                              cv2.COLOR_GRAY2RGB)

            if overlaid_image.shape[:2] != featmap.shape[1:]:
                warnings.warn(
                    f'Since the spatial dimensions of '
                    f'overlaid_image: {overlaid_image.shape[:2]} and '
                    f'featmap: {featmap.shape[1:]} are not same, '
                    f'the feature map will be interpolated. '
                    f'This may cause mismatch problems !')
                if resize_shape is None:
                    featmap = F.interpolate(
                        featmap[None],
                        overlaid_image.shape[:2],
                        mode='bilinear',
                        align_corners=False)[0]

        if resize_shape is not None:
            featmap = F.interpolate(
                featmap[None],
                resize_shape,
                mode='bilinear',
                align_corners=False)[0]
            if overlaid_image is not None:
                overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

        if channel_reduction is not None:
            assert channel_reduction in [
                'squeeze_mean', 'select_max'], \
                f'Mode only support "squeeze_mean", "select_max", ' \
                f'but got {channel_reduction}'
            if channel_reduction == 'select_max':
                sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
                _, indices = torch.topk(sum_channel_featmap, 1)
                feat_map = featmap[indices]
            else:
                feat_map = torch.mean(featmap, dim=0)
            return convert_overlay_heatmap(feat_map, overlaid_image, alpha)
        elif topk <= 0:
            featmap_channel = featmap.shape[0]
            assert featmap_channel in [
                1, 3
            ], ('The input tensor channel dimension must be 1 or 3 '
                'when topk is less than 1, but the channel '
                f'dimension you input is {featmap_channel}, you can use the'
                ' channel_reduction parameter or set topk greater than '
                '0 to solve the error')
            return convert_overlay_heatmap(featmap, overlaid_image, alpha)
        else:
            row, col = arrangement
            channel, height, width = featmap.shape
            assert row * col >= topk, 'The product of row and col in ' \
                                      'the `arrangement` is less than ' \
                                      'topk, please set the ' \
                                      '`arrangement` correctly'

            # Extract the feature map of topk
            topk = min(channel, topk)
            sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
            _, indices = torch.topk(sum_channel_featmap, topk)
            topk_featmap = featmap[indices]

            fig = plt.figure(frameon=False)
            # Set the window layout
            fig.subplots_adjust(
                left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            dpi = fig.get_dpi()
            fig.set_size_inches((width * col + 1e-2) / dpi,
                                (height * row + 1e-2) / dpi)
            for i in range(topk):
                axes = fig.add_subplot(row, col, i + 1)
                axes.axis('off')
                axes.text(2, 15, f'channel: {indices[i]}', fontsize=10)
                axes.imshow(
                    convert_overlay_heatmap(topk_featmap[i], overlaid_image,
                                            alpha))
            image = img_from_canvas(fig.canvas)
            plt.close(fig)
            return image

    @master_only
    def add_config(self, config: Config, **kwargs):
        """Record the config.

        Args:
            config (Config): The Config object.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_config(config, **kwargs)

    @master_only
    def add_graph(self, model: torch.nn.Module, data_batch: Sequence[dict],
                  **kwargs) -> None:
        """Record the model graph.

        Args:
            model (torch.nn.Module): Model to draw.
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_graph(model, data_batch, **kwargs)

    @master_only
    def add_image(self, name: str, image: np.ndarray, step: int = 0) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(name, image, step)  # type: ignore

    @master_only
    def add_scalar(self,
                   name: str,
                   value: Union[int, float],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data.

        Args:
            name (str): The scalar identifier.
            value (float, int): Value to save.
            step (int): Global step value to record. Defaults to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_scalar(name, value, step, **kwargs)  # type: ignore

    @master_only
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars' data.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values.
            step (int): Global step value to record. Defaults to 0.
            file_path (str, optional): The scalar's data will be
                saved to the `file_path` file at the same time
                if the `file_path` parameter is specified.
                Defaults to None.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_scalars(scalar_dict, step, file_path, **kwargs)

    @master_only
    def add_datasample(self,
                       name,
                       image: np.ndarray,
                       data_sample: Optional['BaseDataElement'] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: int = 0,
                       step: int = 0) -> None:
        """Draw datasample."""
        pass

    def close(self) -> None:
        """Close an opened object."""
        for vis_backend in self._vis_backends.values():
            vis_backend.close()

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
            name (str): Name of instance.

        Returns:
            object: Corresponding name instance.
        """
        instance = super().get_instance(name, **kwargs)
        Visualizer._instance_dict[name] = instance
        return instance
