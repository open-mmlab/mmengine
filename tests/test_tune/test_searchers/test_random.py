# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmengine.tune.searchers import RandomSearcher


class TestRandomSearcher(TestCase):
    def test_suggest(self):
        searcher = RandomSearcher(
            rule='greater',
            hparam_spec={
                'x1': {
                    'type': 'discrete',
                    'values': [0.01, 0.02, 0.03]
                },
                'x2': {
                    'type': 'continuous',
                    'lower': 0.01,
                    'upper': 0.1
                }
            })
        
        for _ in range(100):
            hparam = searcher.suggest()
            self.assertTrue(hparam['x1'] in [0.01, 0.02, 0.03])
            self.assertTrue(hparam['x2'] >= 0.01 and hparam['x2'] <= 0.1)