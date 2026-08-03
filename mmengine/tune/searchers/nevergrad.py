# Copyright (c) OpenMMLab. All rights reserved.
import hashlib
import json
from typing import Dict

from .searcher import HYPER_SEARCHERS, Searcher

try:
    import nevergrad as ng
except ImportError:
    ng = None  # type: ignore


@HYPER_SEARCHERS.register_module()
class NevergradSearcher(Searcher):
    """Support hyper parameter searchering with nevergrad.

    Note:
        The detailed usage of nevergrad can be found at
        https://facebookresearch.github.io/nevergrad/.

    Args:
        rule (str): The rule to compare the score.
            Options are 'greater', 'less'.
        hparam_spec (Dict[str, Dict]): The hyper parameter specification.
        num_trials (int): The number of trials.
        solver_type (str): The type of solver.
    """

    def __init__(self,
                 rule: str,
                 hparam_spec: Dict[str, Dict],
                 num_trials: int,
                 solver_type: str = 'NGOpt',
                 *args,
                 **kwargs):
        super().__init__(rule, hparam_spec)
        assert ng is not None, 'nevergrad is not installed'
        self._solver = self._build_solver(solver_type, num_trials)
        self._records = dict()  # type: ignore

        if self.rule == 'less':
            self._rule_op = 1.0
        else:
            self._rule_op = -1.0

    def _build_solver(self, solver_type: str, num_trials: int):
        """Build the solver of nevergrad.

        Args:
            solver_type (str): The type of solver.
            num_trials (int): The number of trials.
        """
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
        """Hash the dict.

        Args:
            d (dict): The dict to be hashed.

        Returns:
            str: The hashed string.
        """
        serialized_data = json.dumps(d, sort_keys=True).encode()
        hashed = hashlib.md5(serialized_data).hexdigest()
        return hashed

    def record(self, hparam: Dict, score: float):
        """Record hparam and score to solver.

        Args:
            hparam (Dict): The hparam to be updated
            score (float): The score to be updated
        """
        hash_key = self._hash_dict(hparam)
        assert hash_key in self._records, \
            f'hparam {hparam} is not in the record'
        self._solver.tell(self._records[hash_key], score * self._rule_op)

    def suggest(self) -> Dict:
        """Suggest a new hparam based on solver's strategy.

        Returns:
            Dict: suggested hparam
        """
        hparam = self._solver.ask()
        hash_key = self._hash_dict(hparam.value)
        self._records[hash_key] = hparam
        return hparam.value
