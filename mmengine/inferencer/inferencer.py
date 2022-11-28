# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import (Any, Callable, Dict, Iterator, List, Optional, Sequence,
                    Tuple, Union)

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track

from mmengine.config import Config, ConfigDict
from mmengine.config.utils import PKG2PROJECT
from mmengine.dataset import COLLATE_FUNCTIONS, pseudo_collate
from mmengine.device import get_device
from mmengine.fileio import load
from mmengine.registry import MODELS, VISUALIZERS, DefaultScope
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from mmengine.utils import get_installed_path
from mmengine.visualization import Visualizer

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray, torch.Tensor]
InputsType = Union[InputType, Sequence[InputType]]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ResType = Union[Dict, List[Dict]]
ConfigType = Union[Config, ConfigDict]
ModelType = Tuple[Union[dict, ConfigType], str]


class InferencerMeta(ABCMeta):
    """Check the legality of the inferencer.

    All Inferencer should not define duplicated keys for
    ``preprocess_kwargs``,``forward_kwargs``, ``visualize_kwargs`` and
    ``postprocess_kwargs``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.preprocess_kwargs, set)
        assert isinstance(self.forward_kwargs, set)
        assert isinstance(self.visualize_kwargs, set)
        assert isinstance(self.postprocess_kwargs, set)
        assert not (self.preprocess_kwargs
                    & self.forward_kwargs
                    & self.visualize_kwargs
                    & self.postprocess_kwargs), (
                        f'Class define error! {self.__name__} should not '
                        'define duplicated keys for `preprocess_kwargs`, '
                        '`forward_kwargs`, `visualize_kwargs` and '
                        '`postprocess_kwargs` are not allowed.')


class BaseInferencer(metaclass=InferencerMeta):
    """Base inferencer for downstream tasks.

    The BaseInferencer provide the standard workflow for inference as follows:

    1. Preprocess the input data by :meth:`preprocess`.
    2. Forward the data to the model by :meth:`forward`. ``BaseInferencer``
       assumes the model inherits from :class:`mmengine.models.BaseModel` and
       will call `model.test_step` in :meth:`forward` by default.
    3. Visualize the results by :meth:`visualize`.
    4. Postprocess and return the results by :meth:`postprocess`.

    When we call the subclasses inherited from BaseInferencer(not override the
    ``__call__``), the workflow will be executed in order.

    All subclasses of BaseInferencer could define the following class
    attributes for customization:

    - ``preprocess_kwargs``: The keys of the kwargs that will be passed to
      :meth:`preprocess`.
    - ``forward_kwargs``: The keys of the kwargs that will be passed to
      :meth:`forward`
    - ``visualize_kwargs``: The keys of the kwargs that will be passed to
      :meth:visualize
    - ``postprocess_kwargs``: The keys of the kwargs that will be passed to
      :meth:`postprocess`

    All attributes mentioned above should be a ``set`` of keys(strings),
    and each key should not be duplicated. Actually, :meth:`__call__` will
    dispatch all the arguments to the corresponding methods according to the
    ``xxx_kwargs`` mentioned above, therefore, the key in sets should
    be unique to avoid ambiguous dispatching.

    Warning:
        If subclasses defined the class attributes mentioned above with
        duplicated keys, an ``AssertionError`` will be raised during import
        process.

    Subclasses inherited from ``BaseInferencer`` should implement
    :meth:`_init_pipeline`, :meth:`visualize` and :meth:`postprocess`:

    - _init_pipeline: Return a callable object to preprocess the input data.
    - visualize: Visualize the results returned by :meth:`forward`.
    - postprocess: Postprocess the results returned by :meth:`forward` and
      :meth:`visualzie`.

    Args:
        model (str, optional): Path to the config file or the model name
            defined in metafile. Take the `mmdet metafile <https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/metafile.yml>`_
            as an example, the `model` could be `retinanet_r18_fpn_1x_coco` or
            its alias.
        weights: Path to the checkpoint. If it is not specified and model is a
            model name of metafile, the weights will be loaded from metafile.
            Defaults to None
        device (str, optional): Device to run inference. If None, the best
            device will be automatically used. Defaults to None.

    Note:
        Since `Inferencer` could be used to inference batch data, therefore
        `collate_fn` should be defined. `collate_fn` is not defined in config
        file, the `collate_fn` will be `pseudo_collate` by default.
    """  # noqa: E501

    preprocess_kwargs: set = set()
    forward_kwargs: set = set()
    visualize_kwargs: set = set()
    postprocess_kwargs: set = set()

    def __init__(
        self,
        model: Union[ModelType, str],
        weights: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        # Load config to cfg
        cfg: ConfigType
        if isinstance(model, str):
            if osp.isfile(model):
                cfg = Config.fromfile(model)
            else:
                # Load config and weights from metafile. If `weights` is given
                # weights defined in metafile will be ignored.
                cfg, _weights = self._load_model_from_metafile(model)
                if weights is not None:
                    weights = _weights
        elif isinstance(model, (Config, ConfigDict)):
            cfg = model
        elif isinstance(model, dict):
            cfg = ConfigDict(model)
        else:
            raise TypeError('config must be a filename or any ConfigType'
                            f'object, but got {type(cfg)}')

        if cfg.model.get('pretrained') is not None:
            cfg.model.pretrained = None

        if device is None:
            device = get_device()

        self.model = self._init_model(cfg, weights, device)  # type: ignore
        self.pipeline = self._init_pipeline(cfg)
        self.collate_fn = self._init_collate(cfg)
        self.visualizer = self._init_visualizer(cfg)

    def _load_model_from_metafile(self, model: str) -> Tuple[Config, str]:
        """Load config and weights from metafile.

        Args:
            model (str): model name defined in metafile.

        Returns:
            Tuple[Config, str]: Loaded Config and weights path defined in
                metafile.
        """
        scope = DefaultScope.get_current_instance().scope_name  # type: ignore
        assert scope is not None, ('scope should be initialized if you want '
                                   'to load config from metafile.')
        project = PKG2PROJECT[scope]
        package_path = get_installed_path(project)
        meta_indexes = load(osp.join(package_path, '.mim', 'model-index.yml'))
        for meta_path in meta_indexes['Import']:
            # meta_path example: mmcls/.mim/configs/conformer/metafile.yml
            meta_path = osp.join(package_path, '.mim', meta_path)
            metainfo = load(meta_path)
            for model_cfg in metainfo['Models']:
                if model_cfg['Name'] == model or model in model_cfg.get(
                        'Alias', []):
                    cfg = Config.fromfile(
                        osp.join(package_path, '.mim', model_cfg['Config']))
                    weights = model_cfg['Weights']
                    return cfg, weights
        raise ValueError(f'Cannot find model: {model} in {project}')

    def _init_model(
        self,
        cfg: ConfigType,
        weights: str,
        device: str = 'cpu',
    ) -> nn.Module:
        """Initialize the model with the given config and checkpoint on the
        specific device.

        Args:
            cfg (ConfigType): Config contained the model information.
            weights (str): Path to the checkpoint.
            device (str, optional): Device to run inference. Defaults to 'cpu'.

        Returns:
            nn.Module: Model loaded with checkpoint.
        """
        model = MODELS.build(cfg.model)
        load_checkpoint(model, weights, map_location='cpu')
        model.cfg = cfg.model
        model.to(device)
        model.eval()
        return model

    def _init_collate(self, cfg: ConfigType) -> Callable:
        """Initialize the collate_fn with the given config.

        Args:
            cfg (ConfigType): Config which could contained the `collate_fn`
                information. If `collate_fn` is not defined in config, it will
                be :func:`pseudo_collate`.

        Returns:
            Callable: Collate function.
        """
        try:
            collate_fn = COLLATE_FUNCTIONS.get(cfg.test_dataloader.collate_fn)
        except AttributeError:
            collate_fn = pseudo_collate
        return collate_fn  # type: ignore

    @abstractmethod
    def _init_pipeline(self, cfg: ConfigType) -> Callable:
        """Initialize the test pipeline."""
        raise NotImplementedError('_init_pipeline is not implemented!')

    def _get_chunk_data(self, dataset: Iterator, chunk_size: int):
        """Get batch data from dataset.

        Args:
            dataset (Iterator): A iterable dataset.
            chunk_size (int): Equivalent to batch size.

        Yields:
            list: batch data.
        """
        dataset_iter = iter(dataset)
        while True:
            try:
                chunk_data = []
                for _ in range(chunk_size):
                    processed_data = next(dataset_iter)
                    chunk_data.append(processed_data)
                yield chunk_data
            except StopIteration:
                if chunk_data:
                    yield chunk_data
                break

    def _init_visualizer(self, cfg: ConfigType) -> Optional[Visualizer]:
        """Initialize visualizers.

        Args:
            cfg (ConfigType): Config contained the visualizer information.

        Returns:
            Visualizer or None: Visualizer initialized with config.
        """
        if 'visualizer' not in cfg:
            return None
        timestamp = str(datetime.timestamp(datetime.now()))
        cfg.visualizer['name'] = f'inferencer-{timestamp}'
        return VISUALIZERS.build(cfg.visualizer)

    def _dispatch_kwargs(self, **kwargs) -> Tuple[Dict, Dict, Dict, Dict]:
        """Dispatch kwargs to preprocess(), forward(), visualize() and
        postprocess() according to the actual demands.

        Returns:
            Tuple[Dict, Dict, Dict, Dict]: kwargs passed to preprocess,
            forward, visualize and postprocess respectively.
        """
        # Ensure each argument only matches one function
        for key in kwargs.keys():
            method_kwargs = [
                self.preprocess_kwargs,
                self.forward_kwargs,
                self.visualize_kwargs,
                self.postprocess_kwargs,
            ]
            matches = tuple(kwarg for kwarg in method_kwargs if key in kwarg)
            if len(matches) == 0:
                raise ValueError(
                    f'unknown argument {key} for `preprocess`, `forward`, '
                    '`visualize` and `postprocess`')
            if len(matches) > 1:
                raise ValueError(f'Ambiguous argument {key} for {matches}')

        preprocess_kwargs = {}
        forward_kwargs = {}
        visualize_kwargs = {}
        postprocess_kwargs = {}

        for key, value in kwargs.items():
            if key in self.preprocess_kwargs:
                preprocess_kwargs[key] = value
            elif key in self.forward_kwargs:
                forward_kwargs[key] = value
            elif key in self.visualize_kwargs:
                visualize_kwargs[key] = value
            else:
                postprocess_kwargs[key] = value

        return (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        )

    def __call__(
        self,
        inputs: InputsType,
        return_datasamples: bool = False,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        """Call the inferencer.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            return_datasamples (bool): Whether to return results as
                datasamples. Defaults to False.
            batch_size (int): Batch size. Defaults to 1.

        Returns:
            dict: Inference and visualization results.
        """
        (
            preprocess_kwargs,
            forward_kwargs,
            visualize_kwargs,
            postprocess_kwargs,
        ) = self._dispatch_kwargs(**kwargs)
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = self.preprocess(
            inputs, batch_size=batch_size, **preprocess_kwargs)
        preds = []
        for data in track(inputs, description='Inference'):
            preds.extend(self.forward(data, **forward_kwargs))
        visualization = self.visualize(inputs, preds, **visualize_kwargs)
        results = self.postprocess(preds, visualization, return_datasamples,
                                   **postprocess_kwargs)
        return results

    def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
        """Process the inputs into a model-feedable format.

        Args:
            inputs (InputsType): Inputs given by user.
            batch_size (int): batch size. Defaults to 1.

        Yields:
            Any: Data processed by the ``pipeline`` and ``collate_fn``.
        """
        dataloader = self._get_chunk_data(
            map(self.pipeline, inputs), batch_size)
        for data in dataloader:
            yield self.collate_fn(data)

    @torch.no_grad()
    def forward(self, inputs: InputsType, **kwargs) -> Any:
        """Forward the inputs to the model."""
        return self.model.test_step(inputs)

    @abstractmethod
    def visualize(self,
                  inputs: InputsType,
                  preds: Any,
                  show: bool = False,
                  **kwargs) -> List[np.ndarray]:
        """Visualize predictions.

        Args:
            inputs (InputsType): Inputs for the inferencer.
            preds (Any): Predictions of the model.
            show (bool): Whether to display the image in a popup window.
                Defaults to False.
        """
        raise NotImplementedError('visualize is not implemented!')

    @abstractmethod
    def postprocess(
        self,
        preds: Any,
        visualization: Optional[List[np.ndarray]] = None,
        return_datasample=False,
        **kwargs,
    ) -> dict:
        """Postprocess predictions.

        Args:
            preds (List[Dict]): Predictions of the model.
            visualization (Optional[np.ndarray]): Visualized predictions.
            return_datasample (bool): Whether to return results as datasamples.

        Returns:
            dict: Inference and visualization results with key ``predictions``
            and ``visualization``
        """
        raise NotImplementedError('postprocess is not implemented!')
