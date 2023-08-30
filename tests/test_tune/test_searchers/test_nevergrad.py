# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import List
from unittest import TestCase, skipIf

from mmengine.tune.searchers import NevergradSearcher

try:
    import nevergrad  # noqa: F401
    NEVERGRAD_AVAILABLE = True
except ImportError:
    NEVERGRAD_AVAILABLE = False


@skipIf(not NEVERGRAD_AVAILABLE, 'nevergrad is not installed')
class TestNevergradSearcher(TestCase):

    def noisy_sphere_function(self, x: List[float]):
        """Sphere function with noise: f(x) = sum(x_i^2) + noise"""
        noise = random.gauss(0, 0.1)  # Gaussian noise with mean 0 and std 0.1
        return sum([x_i**2 for x_i in x]) + noise

    def one_max_function(self, x: List[int]):
        """OneMax function: f(x) = sum(x_i) for binary x_i"""
        return sum(x)

    @property
    def target_solver_types(self):
        return ['OnePlusOne', 'CMA', 'BO', 'DE', 'PSO', 'NGO']

    def test_hash_dict(self):
        searcher = NevergradSearcher(
            rule='less',
            hparam_spec={},
            num_trials=100,
            solver_type='OnePlusOne')

        # Check different dicts yield different hashes
        d1 = {'x': 1, 'y': 2}
        d2 = {'x': 1, 'y': 3}
        self.assertNotEqual(searcher._hash_dict(d1), searcher._hash_dict(d2))

        # Check same dict yields same hash
        self.assertEqual(searcher._hash_dict(d1), searcher._hash_dict(d1))

        # Check order doesn't matter
        d3 = {'y': 2, 'x': 1}
        self.assertEqual(searcher._hash_dict(d1), searcher._hash_dict(d3))

    def test_noisy_sphere_function(self):
        hparam_continuous_space = {
            'x1': {
                'type': 'continuous',
                'lower': -5.0,
                'upper': 5.0
            },
            'x2': {
                'type': 'continuous',
                'lower': -5.0,
                'upper': 5.0
            }
        }
        for solver_type in self.target_solver_types:
            searcher = NevergradSearcher(
                rule='less',
                hparam_spec=hparam_continuous_space,
                num_trials=100,
                solver_type=solver_type)
            for _ in range(100):
                hparam = searcher.suggest()
                score = self.noisy_sphere_function(
                    [v for _, v in hparam.items()])
                searcher.record(hparam, score)
            # For the noisy sphere function,
            # the optimal should be close to x1=0 and x2=0
            best_hparam = searcher.suggest()
            self.assertAlmostEqual(best_hparam['x1'], 0.0, places=1)
            self.assertAlmostEqual(best_hparam['x2'], 0.0, places=1)

    def test_one_max_function(self):
        # Define the discrete search space for OneMax
        hparam_discrete_space = {
            'x1': {
                'type': 'discrete',
                'values': [0, 1]
            },
            'x2': {
                'type': 'discrete',
                'values': [0, 1]
            },
            'x3': {
                'type': 'discrete',
                'values': [0, 1]
            },
            'x4': {
                'type': 'discrete',
                'values': [0, 1]
            }
        }
        for solver_type in self.target_solver_types:
            searcher = NevergradSearcher(
                rule='greater',
                hparam_spec=hparam_discrete_space,
                num_trials=100,
                solver_type=solver_type)
            for _ in range(100):
                hparam = searcher.suggest()
                score = self.one_max_function([v for _, v in hparam.items()])
                searcher.record(hparam, score)
            # For the OneMax function, the optimal solution is x1=x2=x3=x4=1
            best_hparam = searcher.suggest()
            self.assertEqual(best_hparam['x1'], 1)
            self.assertEqual(best_hparam['x2'], 1)
            self.assertEqual(best_hparam['x3'], 1)
            self.assertEqual(best_hparam['x4'], 1)
