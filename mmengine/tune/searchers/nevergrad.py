# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import json
from typing import Dict
from .searcher import Searcher, HYPER_SEARCHERS

try:
    import nevergrad as ng
except ImportError:
    ng = None  # type: ignore


@HYPER_SEARCHERS.register_module()
class NevergradSearcher(Searcher):

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