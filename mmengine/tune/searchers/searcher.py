# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict

from mmengine.registry import Registry

HYPER_SEARCHERS = Registry('hyper parameter searcher')


class Searcher:
    """Base hyper parameter searcher class.

    All hyper parameter searchers inherit from this class.
    """

    rules_supported = ['greater', 'less']

    def __init__(self, rule: str, hparam_spec: Dict[str, Dict]):
        assert rule in self.rules_supported, \
            f"rule must be 'less' or 'greater', but got {rule}"
        self._rule = rule
        self._validate_hparam_spec(hparam_spec)
        self._hparam_spec = hparam_spec

    def _validate_hparam_spec(self, hparam_spec: Dict[str, Dict]):
        """Validate hparam_spec.

        An example of hparam_spec:

            1. discrete:
            hparam_spec = {
                'lr': {
                    'type': 'discrete',
                    'values': [0.01, 0.02, 0.03]
                }
            }

            2. continuous:
            hparam_spec = {
                'lr': {
                    'type': 'continuous',
                    'lower': 0.01,
                    'upper': 0.1
                }
            }

        Args:
            hparam_spec (Dict[str, Dict]): The hyper parameter specification.
        """
        for _, v in hparam_spec.items():
            assert v.get('type', None) in [
                'discrete', 'continuous'
            ], \
                'hparam_spec must have a key "type" and ' \
                f'its value must be "discrete" or "continuous", but got {v}'
            if v['type'] == 'discrete':
                assert 'values' in v, \
                    'if hparam_spec["type"] is "discrete", ' +\
                    f'hparam_spec must have a key "values", but got {v}'
            else:
                assert 'lower' in v and 'upper' in v, \
                    'if hparam_spec["type"] is "continuous", ' +\
                    'hparam_spec must have keys "lower" and "upper", ' +\
                    f'but got {v}'

    @property
    def hparam_spec(self) -> Dict[str, Dict]:
        """Dict: The hyper parameter specification."""
        return self._hparam_spec

    @property
    def rule(self) -> str:
        """str: The rule of the searcher, 'less' or 'greater'."""
        return self._rule

    def record(self, hparam: Dict, score: float):
        """Record hparam and score to solver.

        Args:
            hparam (Dict): The hparam to be updated
            score (float): The score to be updated
        """

    def suggest(self) -> Dict:
        """Suggest a new hparam based on solver's strategy.

        Returns:
            Dict: suggested hparam
        """
