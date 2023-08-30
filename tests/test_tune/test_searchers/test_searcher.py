# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

from mmengine.tune.searchers import Searcher

class TestSearcher(TestCase):

    def test_rule(self):
        valid_hparam_spec_1 = {
            'lr': {
                'type': 'discrete',
                'values': [0.01, 0.02, 0.03]
            }
        }
        # Invalid cases
        with self.assertRaises(AssertionError):
            Searcher(rule='invalid_rule', hparam_spec=valid_hparam_spec_1)
        Searcher(rule='greater', hparam_spec=valid_hparam_spec_1)
        Searcher(rule='less', hparam_spec=valid_hparam_spec_1)

    def test_validate_hparam_spec(self):
        # Unknown hparam spec type
        invalid_hparam_spec_1 = {
            'lr': {
                'type': 'unknown_type',
                'values': [0.01, 0.02, 0.03]
            }
        }
        with self.assertRaises(AssertionError):
            Searcher(rule='greater', hparam_spec=invalid_hparam_spec_1)

        # Missing keys in continuous hparam_spec 
        invalid_hparam_spec_2 = {
            'lr': {
                'type': 'continuous',
                'lower': 0.01
            }
        }
        with self.assertRaises(AssertionError):
            Searcher(rule='less', hparam_spec=invalid_hparam_spec_2)

        # Invalid discrete hparam_spec
        invalid_hparam_spec_3 = {
            'lr': {
                'type': 'discrete',
                'values': []  # Empty list
            }
        }
        with self.assertRaises(AssertionError):
            Searcher(rule='greater', hparam_spec=invalid_hparam_spec_3)

        # Invalid continuous hparam_spec
        invalid_hparam_spec_4 = {
            'lr': {
                'type': 'continuous',
                'lower': 0.1,
                'upper': 0.01  # lower is greater than upper
            }
        }
        with self.assertRaises(AssertionError):
            Searcher(rule='less', hparam_spec=invalid_hparam_spec_4)

        # Invalid data type in continuous hparam_spec
        invalid_hparam_spec_5 = {
            'lr': {
                'type': 'continuous',
                'lower': '0.01',  # String instead of number
                'upper': 0.1
            }
        }
        with self.assertRaises(AssertionError):
            Searcher(rule='less', hparam_spec=invalid_hparam_spec_5)
    
    def test_hparam_spec_property(self):
        hparam_spec = {
            'lr': {
                'type': 'discrete',
                'values': [0.01, 0.02, 0.03]
            }
        }
        searcher = Searcher(rule='greater', hparam_spec=hparam_spec)
        self.assertEqual(searcher.hparam_spec, hparam_spec)

    def test_rule_property(self):
        searcher = Searcher(rule='greater', hparam_spec={})
        self.assertEqual(searcher.rule, 'greater')