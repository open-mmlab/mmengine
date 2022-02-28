# Copyright (c) OpenMMLab. All rights reserved.
from ..registry import EVALUATORS
from .composed_evaluator import ComposedEvaluator


def build_evaluator(cfg: dict) -> object:
    """Build function of evaluator.

    When the evaluator config is a list, it will automatically build composed
    evaluators.
    """
    if isinstance(cfg, list):
        evaluators = [EVALUATORS.build(_cfg) for _cfg in cfg]
        return ComposedEvaluator(evaluators=evaluators)
    else:
        return EVALUATORS.build(cfg)
