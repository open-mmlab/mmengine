# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

from mmengine.registry import EVALUATOR
from .evaluator import Evaluator


def build_evaluator(evaluator: Union[Dict, List, Evaluator]) -> Evaluator:
    """Build evaluator.

    Examples of ``evaluator``::

        # evaluator could be a built Evaluator instance
        evaluator = Evaluator(metrics=[ToyMetric()])

        # evaluator can also be a list of dict
        evaluator = [
            dict(type='ToyMetric1'),
            dict(type='ToyEvaluator2')
        ]

        # evaluator can also be a list of built metric
        evaluator = [ToyMetric1(), ToyMetric2()]

        # evaluator can also be a dict with key metrics
        evaluator = dict(metrics=ToyMetric())
        # metric is a list
        evaluator = dict(metrics=[ToyMetric()])

    Args:
        evaluator (Evaluator or dict or list): An Evaluator object or a
            config dict or list of config dict used to build an Evaluator.

    Returns:
        Evaluator: Evaluator build from ``evaluator``.
    """
    if isinstance(evaluator, Evaluator):
        return evaluator
    elif isinstance(evaluator, dict):
        # if `metrics` in dict keys, it means to build customized evalutor
        if 'metrics' in evaluator:
            evaluator.setdefault('type', 'Evaluator')
            return EVALUATOR.build(evaluator)
        # otherwise, default evalutor will be built
        else:
            return Evaluator(evaluator)  # type: ignore
    elif isinstance(evaluator, list):
        # use the default `Evaluator`
        return Evaluator(evaluator)  # type: ignore
    else:
        raise TypeError(
            'evaluator should be one of dict, list of dict, and Evaluator'
            f', but got {evaluator}')
