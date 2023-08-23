# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import json
from typing import Dict

from mmengine.registry import Registry

try:
    import nevergrad as ng
except ImportError:
    ng = None

HYPER_SEARCHERS = Registry('hyper parameter searcher')


class _Searcher:

    rules_supported = ['greater', 'less']

    def __init__(self, rule: str, hparam_spec: Dict[str, Dict]):
        assert rule in self.rules_supported, \
            f"rule must be 'less' or 'greater', but got {rule}"
        self._rule = rule
        self._validate_hparam_spec(hparam_spec)
        self._hparam_spec = hparam_spec

    def _validate_hparam_spec(self, hparam_spec):
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
        return self._hparam_spec

    @property
    def rule(self) -> str:
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


@HYPER_SEARCHERS.register_module()
class NevergradSearcher(_Searcher):

    def __init__(self,
                 rule: str,
                 hparam_spec: Dict[str, Dict],
                 num_trials: int,
                 solver_type: str = 'NGOpt',
                 *args,
                 **kwargs):
        super().__init__(rule, hparam_spec)
        assert ng is not None, 'nevergrad is not installed'
        self._optimizer = self._build_optimizer(solver_type, num_trials)
        self._records = dict()  # type: ignore

        if self.rule == 'less':
            self._rule_op = 1.0
        else:
            self._rule_op = -1.0

    def _build_optimizer(self, solver_type: str, num_trials: int):
        converted_hparam_spec = ng.p.Dict(
            **{
                k: ng.p.Scalar(lower=v['lower'], upper=v['upper'])
                if v['type'] == 'continuous' else ng.p.Choice(v['values'])
                for k, v in self.hparam_spec.items()
            })
        solver = ng.optimization.optimizerlib.registry[solver_type](
            parametrization=converted_hparam_spec, budget=num_trials)
        return solver

    def _hash_dict(self, d: dict) -> str:
        serialized_data = json.dumps(d, sort_keys=True).encode()
        hashed = hashlib.md5(serialized_data).hexdigest()
        return hashed

    def suggest(self) -> Dict:
        hparam = self._optimizer.ask()
        hash_key = self._hash_dict(hparam.value)
        self._records[hash_key] = hparam
        return hparam.value

    def record(self, hparam: Dict, score: float):
        hash_key = self._hash_dict(hparam)
        assert hash_key in self._records, \
            f'hparam {hparam} is not in the record'
        self._optimizer.tell(self._records[hash_key], score * self._rule_op)
