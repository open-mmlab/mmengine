# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

from ..registry import EVALUATORS
from .base import BaseEvaluator
from .composed_evaluator import ComposedEvaluator


def build_evaluator(
        cfg: Union[dict, list]) -> Union[BaseEvaluator, ComposedEvaluator]:
    """Build function of evaluator.

    When the evaluator config is a list, it will automatically build composed
    evaluators.
    """
    if isinstance(cfg, list):
        evaluators = [EVALUATORS.build(_cfg) for _cfg in cfg]
        return ComposedEvaluator(evaluators=evaluators)
    else:
        return EVALUATORS.build(cfg)
