# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict
from unittest import mock

from mmengine.testing import RunnerTestCase
from mmengine.tune import Tuner
from mmengine.tune.searchers import HYPER_SEARCHERS, Searcher


class ToySearcher(Searcher):

    def suggest(self) -> Dict:
        hparam = dict()
        for k, v in self.hparam_spec.items():
            if v['type'] == 'discrete':
                hparam[k] = v['values'][0]
            else:
                hparam[k] = (v['lower'] + v['upper']) / 2
        return hparam


class TestTuner(RunnerTestCase):

    def setUp(self) -> None:
        super().setUp()
        HYPER_SEARCHERS.register_module(module=ToySearcher)
        self.hparam_spec = {
            'optim_wrapper.optimizer.lr': {
                'type': 'discrete',
                'values': [0.1, 0.2, 0.3]
            }
        }

    def tearDown(self):
        super().tearDown()
        HYPER_SEARCHERS.module_dict.pop('ToySearcher', None)

    def test_init(self):
        with self.assertRaises(ValueError):
            Tuner(
                runner_cfg=dict(),
                hparam_spec=dict(),
                monitor='loss',
                rule='invalid_rule',
                num_trials=2,
                searcher_cfg=dict(type='ToySearcher'))

        # Initializing with correct parameters
        tuner = Tuner(
            runner_cfg=self.epoch_based_cfg,
            hparam_spec=self.hparam_spec,
            monitor='loss',
            rule='less',
            num_trials=2,
            searcher_cfg=dict(type='ToySearcher'))

        # Verify the properties
        self.assertEqual(tuner.hparam_spec, self.hparam_spec)
        self.assertEqual(tuner.monitor, 'loss')
        self.assertEqual(tuner.rule, 'less')
        self.assertEqual(tuner.num_trials, 2)

        # Ensure a searcher of type ToySearcher is used
        self.assertIsInstance(tuner.searcher, ToySearcher)

    def mock_is_main_process(self, return_value=True):
        return mock.patch(
            'mmengine.dist.is_main_process', return_value=return_value)

    def mock_broadcast(self, side_effect=None):
        return mock.patch(
            'mmengine.dist.broadcast_object_list', side_effect=side_effect)

    def test_inject_config(self):
        # Inject into a single level
        cfg = {'a': 1}
        Tuner.inject_config(cfg, 'a', 2)
        self.assertEqual(cfg['a'], 2)

        # Inject into a nested level
        cfg = {'level1': {'level2': {'level3': 3}}}
        Tuner.inject_config(cfg, 'level1.level2.level3', 4)
        self.assertEqual(cfg['level1']['level2']['level3'], 4)

        # Inject into a non-existent key
        cfg = {}
        with self.assertRaises(KeyError):
            Tuner.inject_config(cfg, 'a', 1)

        # Inject into a sequence
        cfg = {'sequence': [1, 2, 3]}
        Tuner.inject_config(cfg, 'sequence.1', 5)
        self.assertEqual(cfg['sequence'][1], 5)

    @mock.patch('mmengine.runner.Runner.train')
    @mock.patch('mmengine.tune._report_hook.ReportingHook.report_score')
    def test_successful_run(self, mock_report_score, mock_train):
        tuner = Tuner(
            runner_cfg=self.epoch_based_cfg,
            hparam_spec=self.hparam_spec,
            monitor='loss',
            rule='less',
            num_trials=2,
            searcher_cfg=dict(type='ToySearcher'))

        tuner.searcher.suggest = mock.MagicMock(
            return_value={'optim_wrapper.optimizer.lr': 0.1})
        tuner.searcher.record = mock.MagicMock()

        mock_report_score.return_value = 0.05

        with self.mock_is_main_process(), self.mock_broadcast():
            hparam, score, error = tuner._run_trial()

        self.assertEqual(hparam, {'optim_wrapper.optimizer.lr': 0.1})
        self.assertEqual(score, 0.05)
        self.assertIsNone(error)
        tuner.searcher.record.assert_called_with(
            {'optim_wrapper.optimizer.lr': 0.1}, 0.05)

    @mock.patch('mmengine.runner.Runner.train')
    @mock.patch('mmengine.tune._report_hook.ReportingHook.report_score')
    def test_run_with_exception(self, mock_report_score, mock_train):
        mock_train.side_effect = Exception('Error during training')

        tuner = Tuner(
            runner_cfg=self.epoch_based_cfg,
            hparam_spec=self.hparam_spec,
            monitor='loss',
            rule='less',
            num_trials=2,
            searcher_cfg=dict(type='ToySearcher'))

        tuner.searcher.suggest = mock.MagicMock(
            return_value={'optim_wrapper.optimizer.lr': 0.1})
        tuner.searcher.record = mock.MagicMock()

        with self.mock_is_main_process(), self.mock_broadcast():
            hparam, score, error = tuner._run_trial()

        self.assertEqual(hparam, {'optim_wrapper.optimizer.lr': 0.1})
        self.assertEqual(score, float('inf'))
        self.assertTrue(isinstance(error, Exception))
        tuner.searcher.record.assert_called_with(
            {'optim_wrapper.optimizer.lr': 0.1}, float('inf'))

    @mock.patch('mmengine.runner.Runner.train')
    @mock.patch('mmengine.tune._report_hook.ReportingHook.report_score')
    def test_tune(self, mock_report_score, mock_train):
        mock_scores = [0.05, 0.03, 0.04, 0.06]
        mock_hparams = [{
            'optim_wrapper.optimizer.lr': 0.1
        }, {
            'optim_wrapper.optimizer.lr': 0.05
        }, {
            'optim_wrapper.optimizer.lr': 0.2
        }, {
            'optim_wrapper.optimizer.lr': 0.3
        }]

        mock_report_score.side_effect = mock_scores

        tuner = Tuner(
            runner_cfg=self.epoch_based_cfg,
            hparam_spec=self.hparam_spec,
            monitor='loss',
            rule='less',
            num_trials=4,
            searcher_cfg=dict(type='ToySearcher'))

        mock_run_trial_return_values = [
            (mock_hparams[0], mock_scores[0], None),
            (mock_hparams[1], mock_scores[1],
             Exception('Error during training')),
            (mock_hparams[2], mock_scores[2], None),
            (mock_hparams[3], mock_scores[3], None)
        ]
        tuner._run_trial = mock.MagicMock(
            side_effect=mock_run_trial_return_values)

        with self.mock_is_main_process(), self.mock_broadcast():
            result = tuner.tune()

        self.assertEqual(tuner._history, [(mock_hparams[0], mock_scores[0]),
                                          (mock_hparams[1], mock_scores[1]),
                                          (mock_hparams[2], mock_scores[2]),
                                          (mock_hparams[3], mock_scores[3])])

        self.assertEqual(result, {
            'hparam': mock_hparams[1],
            'score': mock_scores[1]
        })

        tuner = Tuner(
            runner_cfg=self.epoch_based_cfg,
            hparam_spec=self.hparam_spec,
            monitor='loss',
            rule='greater',
            num_trials=4,
            searcher_cfg=dict(type='ToySearcher'))
        tuner._run_trial = mock.MagicMock(
            side_effect=mock_run_trial_return_values)
        with self.mock_is_main_process(), self.mock_broadcast():
            result = tuner.tune()
        self.assertEqual(result, {
            'hparam': mock_hparams[3],
            'score': mock_scores[3]
        })

    def test_clear(self):
        tuner = Tuner(
            runner_cfg=self.epoch_based_cfg,
            hparam_spec=self.hparam_spec,
            monitor='loss',
            rule='less',
            num_trials=2,
            searcher_cfg=dict(type='ToySearcher'))

        tuner.history.append(({'optim_wrapper.optimizer.lr': 0.1}, 0.05))
        tuner.clear()
        self.assertEqual(tuner.history, [])

    def test_with_runner(self):
        tuner = Tuner(
            runner_cfg=self.epoch_based_cfg,
            hparam_spec=self.hparam_spec,
            monitor='acc',
            rule='greater',
            num_trials=10,
            searcher_cfg=dict(type='ToySearcher'))

        with self.mock_is_main_process(), self.mock_broadcast():
            result = tuner.tune()

        self.assertTrue({
            hparam['optim_wrapper.optimizer.lr']
            for hparam, _ in tuner.history
        }.issubset(
            set(self.hparam_spec['optim_wrapper.optimizer.lr']['values'])))
        self.assertEqual(result['score'], 1)
