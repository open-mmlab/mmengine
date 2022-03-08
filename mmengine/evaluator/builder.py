# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from ..registry import EVALUATORS
from .base import BaseEvaluator
from .composed_evaluator import ComposedEvaluator


def build_evaluator(
    cfg: Union[dict, list],
    default_scope: Optional[str] = None
) -> Union[BaseEvaluator, ComposedEvaluator]:
    """Build function of evaluator.

    When the evaluator config is a list, it will automatically build composed
    evaluators.

    Args:
        cfg (dict | list): Config of evaluator. When the config is a list, it
            will automatically build composed evaluators.
        default_scope (str, optional): The ``default_scope`` is used to
            reset the current registry. Defaults to None.

    Returns:
        BaseEvaluator or ComposedEvaluator: The built evaluator.
    """
    if isinstance(cfg, list):
        evaluators = [
            EVALUATORS.build(_cfg, default_scope=default_scope) for _cfg in cfg
        ]
        return ComposedEvaluator(evaluators=evaluators)
    else:
        return EVALUATORS.build(cfg, default_scope=default_scope)
