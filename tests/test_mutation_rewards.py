import json
import unittest
from unittest.mock import Mock, patch

from mutation_rewards import make_mutation_grpo_reward, mutation_evidence, penalty
from rl_rewards import JudgeUnavailableError, make_grpo_reward
from test_rl_rewards import GOOD, REJECT, SOURCE, ScriptedJudge
from train_grpo import assert_matches_baseline


def witness(original, labels, feasible=True):
    return {'original_holds': original, 'verdicts': [{'feasible': feasible, 'holds': label} for label in labels]}


def evidence(*witnesses, errors=0):
    return {'status': 'grounded', 'witnesses': list(witnesses), 'api_errors': errors}


class MutationRewardTests(unittest.TestCase):
    def test_exact_fraction_floor_invalid_and_fallback(self):
        rows = evidence(witness(True, [False] * 3), witness(True, [True] * 3),
                        witness(True, [False, False, True]), witness(True, [None] * 3),
                        witness(True, [False] * 3, feasible=False))
        result = penalty(.2, True, rows)
        self.assertEqual(result['usable_witnesses'], 2)
        self.assertEqual(result['E'], .5)
        self.assertAlmostEqual(result['reward'], .15)
        self.assertAlmostEqual(penalty(1., True, rows)['reward'], .95)
        self.assertEqual(penalty(.06, True, evidence(witness(True, [False] * 3)))['reward'], .05)
        self.assertEqual(penalty(1., False, rows)['reward'], 0.)
        self.assertEqual(penalty(.2, True, evidence())['reward'], .2)

    def test_existing_base_reward_is_wrapped_and_successes_cached(self):
        judge = ScriptedJudge(REJECT)
        provider = Mock(return_value=evidence(witness(True, [False] * 3)))
        events = []
        reward = make_mutation_grpo_reward(make_grpo_reward(judge), events.append, witness_provider=provider)
        chat = [[{'role': 'assistant', 'content': GOOD}]]
        self.assertAlmostEqual(reward(chat, [SOURCE], id=['a'])[0], 0.)
        self.assertAlmostEqual(reward(chat, [SOURCE], id=['a'])[0], 0.)
        self.assertEqual(len(judge.calls), 1)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(events[0]['base_reward'], 0.)
        self.assertEqual(events[0]['error_witnesses'], 1)

    def test_optional_judge_failure_falls_back_without_caching_failure(self):
        provider = Mock(side_effect=[evidence(errors=1), evidence(witness(True, [False] * 3))])
        reward = make_mutation_grpo_reward(make_grpo_reward(ScriptedJudge(REJECT)), witness_provider=provider)
        self.assertEqual(reward([GOOD], [SOURCE]), [0.])
        self.assertEqual(reward([GOOD], [SOURCE]), [0.])
        self.assertEqual(provider.call_count, 2)

    def test_base_failure_still_aborts_and_invalid_smt_skips_witnesses(self):
        provider = Mock()
        failed = make_mutation_grpo_reward(Mock(side_effect=JudgeUnavailableError('offline')), witness_provider=provider)
        with self.assertRaises(JudgeUnavailableError): failed([GOOD], [SOURCE])
        reward = make_mutation_grpo_reward(make_grpo_reward(ScriptedJudge()), witness_provider=provider)
        self.assertEqual(reward([GOOD[:-1]], [SOURCE]), [0.])
        provider.assert_not_called()

    def test_source_context_and_assumptions_separate_cached_evidence(self):
        provider = Mock(return_value=evidence())
        reward = make_mutation_grpo_reward(lambda completions, **_: [.2] * len(completions), witness_provider=provider)
        reward([GOOD], [SOURCE])
        reward([GOOD], [SOURCE], source_context=[['Definition']])
        reward([GOOD], [SOURCE], declared_assumptions=[['Explicit assumption']])
        self.assertEqual(provider.call_count, 3)

    def test_duplicate_states_judged_once_and_formulas_hidden(self):
        state = dict(status='distinguishing_state', values={'ratio': '4'}, candidate_a_holds=True)
        prepared = dict(schema={'status': 'grounded', 'symbols': []}, mutations=[state, state])
        response = dict(feasible=True, holds=False, source_quotes=[SOURCE], reason='Above limit.')
        with patch('mutation_rewards.prepare', return_value=prepared), patch('mutation_rewards.ask', return_value=response) as ask:
            result = mutation_evidence({'text': SOURCE}, GOOD)
        self.assertEqual(len(result['witnesses']), 1)
        self.assertEqual(ask.call_count, 3)
        for call in ask.call_args_list:
            payload = call.args[2]
            self.assertNotIn('candidate_smt2', payload)
            self.assertNotIn('original_holds', payload)
            self.assertNotIn('mutations', payload)

    def test_baseline_matching_allows_only_output_location_change(self):
        baseline = {field: [] for field in ('record_ids', 'evaluation_record_ids', 'excluded_development_record_ids')}
        baseline.update(sft_checkpoint='frozen', source_sha256='data', evaluation_sha256='dev', versions={}, judge={},
                        training=dict(output_dir='baseline', max_steps=50, num_generations=2))
        current = json.loads(json.dumps(baseline))
        current['training']['output_dir'] = 'mutation'
        assert_matches_baseline(current, baseline)
        current['training']['max_steps'] = 51
        with self.assertRaisesRegex(ValueError, 'max_steps'): assert_matches_baseline(current, baseline)


if __name__ == '__main__':
    unittest.main()
