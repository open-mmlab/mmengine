# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmengine.registry import VISUALIZERS
from .visualizer import Visualizer
from datetime import datetime

def build_visualizer(
        visualizer: Optional[Union[Visualizer,
                                    Dict]] = None,
        experiment_name: str = datetime.now().strftime(r'%Y%m%d_%H%M%S'),
        log_dir: str = 'work_dirs'
        ) -> Visualizer:
    """Build a global asscessable Visualizer.

    Args:
        visualizer (Visualizer or dict, optional): A Visualizer object
            or a dict to build Visualizer object. If ``visualizer`` is a
            Visualizer object, just returns itself. If not specified,
            default config will be used to build Visualizer object.
            Defaults to None.

    Returns:
        Visualizer: A Visualizer object build from ``visualizer``.
    """
    if visualizer is None:
        visualizer = dict(
            name=experiment_name,
            vis_backends=[dict(type='LocalVisBackend')],
            save_dir=log_dir)
        return Visualizer.get_instance(**visualizer)

    if isinstance(visualizer, Visualizer):
        return visualizer

    if isinstance(visualizer, dict):
        # ensure visualizer containing name key
        visualizer.setdefault('name', experiment_name)
        visualizer.setdefault('save_dir', log_dir)
        return VISUALIZERS.build(visualizer)
    else:
        raise TypeError(
            'visualizer should be Visualizer object, a dict or None, '
            f'but got {visualizer}')
