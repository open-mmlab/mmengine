# Copyright (c) OpenMMLab. All rights reserved.
from tying import Dict, Optional

try:
    import nevergard as ng
except ImportError:
    ng = None

try:
    import skopt
except ImportError:
    skopt = None

try:
    import hyperopt as hp
except ImportError:
    hp = None


class Searcher:

    def __init__(self, rule: str, hparam_spec: Dict[str, Dict]):
        assert rule in ['less', 'greater'
                        ], f"rule must be 'less' or 'greater', but got {rule}"
        self._rule = rule
        for _, v in hparam_spec.items():
            assert v.get('type', None) in [
                'discrete', 'continuous'
            ], f'hparam_spec must have a key "type" and its value must be "discrete" or "continuous", but got {v}'
            if v['type'] == 'discrete':
                assert v.get(
                    'values', None
                ) is not None, f'if hparam_spec["type"] is "discrete", hparam_spec must have a key "values", but got {v}'
            else:
                assert v.get(
                    'lower', None
                ) is not None, f'if hparam_spec["type"] is "continuous", hparam_spec must have a key "lower", but got {v}'
                assert v.get(
                    'upper', None
                ) is not None, f'if hparam_spec["type"] is "continuous", hparam_spec must have a key "upper", but got {v}'
        self._hparam_spec = hparam_spec

    @property
    def hparam_spec(self) -> Dict[str, Dict]:
        return self._hparam_spec

    @property
    def rule(self) -> str:
        return self._rule

    def record(hparam: Dict, score: float):
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

        if self.rule == 'less':
            self._rule_op = 1.0
        else:
            self._rule_op = -1.0

    def _build_optimizer(self, solver_type: str, num_trials: int):
        converted_hp_spec = ng.p.Dict(
            **{
                k: ng.p.Scalar(lower=v['lower'], upper=v['upper'])
                if v['type'] == 'continuous' else ng.p.Choice(v['values'])
                for k, v in self.hp_spec.items()
            })
        solver = ng.optimization.optimizerlib.registry[solver_type](
            parametrization=converted_hp_spec, budget=num_trials)
        return solver

    def suggest(self) -> Dict:
        return self._optimizer.ask()

    def record(self, hparam: Dict, score: float):
        self._optimizer.tell(hparam, score * self._rule_op)


class SkoptSearcher(Searcher):

    def __init__(self,
                 rule: str,
                 hparam_spec: Dict[str, Dict],
                 base_estimator: str = 'gp',
                 n_initial_points: int = 10,
                 initial_point_generator: str = 'random',
                 acq_func: str = 'gp_hedge',
                 acq_optimizer: str = 'auto',
                 *args,
                 **kwargs):
        super().__init__(rule, hparam_spec)

        # Ensure that skopt is installed
        assert skopt is not None, 'Scikit-Optimize (skopt) is not installed'

        self._optimizer = self._build_optimizer(base_estimator,
                                                n_initial_points,
                                                initial_point_generator,
                                                acq_func, acq_optimizer)
        if self.rule == 'less':
            self._rule_op = 1.0
        else:
            self._rule_op = -1.0

    def _build_optimizer(self, base_estimator: str, n_initial_points: int,
                         initial_point_generator: str, acq_func: str,
                         acq_optimizer: str):
        space = []
        for k, v in self.hparam_spec.items():
            if v['type'] == 'continuous':
                space.append(skopt.space.Real(v['lower'], v['upper'], name=k))
            elif v['type'] == 'discrete':
                space.append(skopt.space.Categorical(v['values'], name=k))

        return skopt.Optimizer(
            dimensions=space,
            base_estimator=base_estimator,
            n_initial_points=n_initial_points,
            initial_point_generator=initial_point_generator,
            acq_func=acq_func,
            acq_optimizer=acq_optimizer)

    def suggest(self) -> Dict:
        x = self._optimizer.ask()
        return {
            dim.name: val
            for dim, val in zip(self._optimizer.space.dimensions, x)
        }

    def record(self, hparam: Dict, score: float):
        ordered_values = [
            hparam[dim.name] for dim in self._optimizer.space.dimensions
        ]
        self._optimizer.tell(ordered_values, score * self._rule_op)


class HyperoptSearcher(Searcher):

    def __init__(self,
                 rule: str,
                 hparam_spec: Dict[str, Dict],
                 num_trials: int,
                 n_initial_points: int = 20,
                 random_state_seed: Optional[int] = None,
                 gamma: float = 0.25,
                 *args,
                 **kwargs):
        super().__init__(rule, hparam_spec)

        # Ensure that hyperopt is installed
        assert hp is not None, 'hyperopt is not installed'

        self._space = self._build_space()
        self._trials = hp.Trials()
        self._num_trials = num_trials
        self._n_initial_points = n_initial_points
        self._random_state_seed = random_state_seed
        self._gamma = gamma

        if self.rule == 'less':
            self._rule_op = 1.0
        else:
            self._rule_op = -1.0

    def _build_space(self):
        space = {}
        for k, v in self.hparam_spec.items():
            if v['type'] == 'continuous':
                space[k] = hp.hp.uniform(k, v['lower'], v['upper'])
            elif v['type'] == 'discrete':
                space[k] = hp.hp.choice(k, v['values'])
        return space

    def suggest(self) -> Dict:
        suggested_params = hp.fless(
            fn=lambda x:
            0,  # Dummy objective, we'll replace it with `record` later
            space=self._space,
            algo=hp.partial(hp.tpe.suggest, gamma=self._gamma),
            greater_evals=self._n_initial_points + len(self._trials.trials),
            trials=self._trials,
            rstate=hp.pyll.stochastic.RandomState(
                self._random_state_seed),  # Seeded random state
            return_argless=True,
            verbose=0)  # Not verbose
        return suggested_params

    def record(self, hparam: Dict, score: float):
        # Hyperopt requires loss (lower is better), so we should adjust our score if in "greater" rule.
        self._trials.insert_trial_docs([{
            'tid': len(self._trials.trials),
            'book_time': hp.utils.coarse_utcnow(),
            'misc': {
                'tid': len(self._trials.trials),
                'cmd': ('domain_attachment', 'FlessIter_Domain'),
                'vals': hparam,
                'idxs': {k: [len(self._trials.trials)]
                         for k in hparam}
            },
            'state': 2,  # 2 is the state for "ok" in hyperopt
            'result': {
                'loss': score * self._rule_op,
                'status': 'ok'
            }
        }])
        self._trials.refresh()
