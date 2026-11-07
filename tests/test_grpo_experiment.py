"""Leakage exclusions and usable reward signals for the small GRPO experiment."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probe_rewards import summarize
from train_grpo import make_logged_reward, training_indices


class ExperimentTests(unittest.TestCase):
    def test_dense_semantic_contrasts_include_intermediate_grades(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            output = Path(directory)
            reward = make_logged_reward(lambda **_: [7/15, 11/15], output, ending_token_ids={99})
            reward(['smt'] * 2, ['source'] * 2, id=['a'] * 2, completion_ids=[[1,99]] * 2)
            group = json.loads((output/'reward_groups.jsonl').read_text())
            self.assertEqual(group['semantic_reward_groups'], 0)
            self.assertEqual(group['graded_semantic_reward_groups'], 1)
            self.assertEqual(group['effective_graded_semantic_reward_groups'], 1)

    def test_missing_context_group_stays_unscored(self):
        from rl_rewards import InsufficientContextError
        verdict = {'needs_context': True, 'semantic_score': None, 'reason': 'Missing threshold'}
        def missing(**kwargs):
            raise InsufficientContextError(verdict)
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            output = Path(directory)
            reward = make_logged_reward(missing, output, num_generations=8, ending_token_ids={99},
                                        abstain_on_missing_context=True)
            self.assertEqual(reward(['smt']*8, ['source']*8, id=['a']*8,
                                    completion_ids=[[1,99]]*8), [None]*8)
            group = json.loads((output/'reward_groups.jsonl').read_text())
            self.assertEqual(group['unscored_groups'], 1)
            self.assertEqual(group['unscored_completions'], 8)
            self.assertEqual(group['effective_informative_groups'], 0)
            self.assertEqual(group['graded_semantic_reward_groups'], 0)
            self.assertEqual(group['context_abstention'], verdict)
            rows = [json.loads(s) for s in (output/'rewards.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['reward'] is None for r in rows))
            with self.assertRaises(InsufficientContextError):
                reward(['smt']*8, ['source']*8, id=['a']*4+['b']*4)

    def test_dense_logging_does_not_hide_judge_failure(self):
        from rl_rewards import JudgeUnavailableError
        def failed(**kwargs):
            raise JudgeUnavailableError('Transport failure')
        with tempfile.TemporaryDirectory() as directory:
            reward = make_logged_reward(failed, Path(directory), abstain_on_missing_context=True)
            with self.assertRaises(JudgeUnavailableError):
                reward(['smt']*2, ['source']*2, id=['a']*2)

    def test_exclusion_follows_contract_duplicate_connections(self):
        def row(i, contract, text):
            return dict(id=i, contract_id=contract, text=text)
        heldout = [row('dev', 'a', 'Limit 3.5')]
        records = [row('keep', 'd', 'Limit 3.75'), row('chain', 'c', 'Another rule'),
                   row('duplicate', 'b', ' LIMIT   3.5 '), row('bridge', 'b', 'Another rule'),
                   row('sibling', 'a', 'Different covenant'), row('dev', 'a', 'Limit 3.5')]
        self.assertEqual(training_indices(records, heldout), [0])

    def test_logging_distinguishes_masked_and_effective_groups(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            output = Path(directory)
            callback = make_logged_reward(lambda **_: [0., .2, 0., 1.], output,
                                          phase_getter=lambda: 'eval', ending_token_ids={99})
            callback(['smt'] * 4, ['source'] * 4, id=['a', 'a', 'b', 'b'],
                     completion_ids=[[1], [2], [3], [4, 99]])
            groups = json.loads((output / 'reward_groups.jsonl').read_text())
            self.assertEqual(groups['phase'], 'eval')
            self.assertEqual(groups['informative_groups'], 2)
            self.assertEqual(groups['effective_informative_groups'], 1)
            self.assertEqual(groups['masked_completions'], 3)

    def test_probe_separates_failures_abstentions_and_masking(self):
        statuses = ['invalid', 'valid_rejected', 'invalid', 'valid_accepted',
                    'judge_abstained', 'judge_failure']
        rewards = [0., .2, 0., 1., .2, None]
        rows = [dict(id=i // 2, status=status, reward=reward,
                     validation={'valid': status != 'invalid'}, hit_token_limit=i < 3,
                     judge_failed_attempts=2 if status == 'judge_failure' else 0)
                for i, (status, reward) in enumerate(zip(statuses, rewards))]
        result = summarize(rows)
        self.assertEqual(result['informative_groups'], 2)
        self.assertEqual(result['effective_informative_groups'], 1)
        self.assertEqual(result['semantic_decided'], 2)
        self.assertEqual(result['semantic_acceptance_among_decided_valid'], .5)
        self.assertEqual(result['judge_failure_rate_among_valid'], .25)
        self.assertEqual(result['judge_abstention_rate_among_valid'], .25)
        self.assertFalse(result['reward_probe_passed'])

    def test_four_completions_form_one_reward_group(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            output = Path(directory)
            reward = make_logged_reward(lambda **_: [0., .2, 1., 1.], output,
                                        num_generations=4, ending_token_ids={99})
            reward(['smt'] * 4, ['source'] * 4, id=['a'] * 4,
                   completion_ids=[[1, 99]] * 4)
            group = json.loads((output / 'reward_groups.jsonl').read_text())
            self.assertEqual(group['complete_groups'], 1)
            self.assertEqual(group['effective_informative_groups'], 1)
            self.assertEqual(group['incomplete_groups'], 0)

    def test_eight_sample_semantic_contrasts_require_both_valid_reward_levels(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            output = Path(directory)
            values = [0., .2, 1., 1., 0., .2, 0., 0.] + [0., 1., 1., 0., 1., 0., 0., 1.]
            reward = make_logged_reward(lambda **_: values, output,
                                        num_generations=8, ending_token_ids={99})
            tokens = [[1] if i in (2, 3) else [1, 99] for i in range(16)]
            reward(['smt'] * 16, ['source'] * 16, id=['a'] * 8 + ['b'] * 8, completion_ids=tokens)
            group = json.loads((output / 'reward_groups.jsonl').read_text())
            self.assertEqual(group['complete_groups'], 2)
            self.assertEqual(group['groups_with_reward_one'], 2)
            self.assertEqual(group['semantic_reward_groups'], 1)
            self.assertEqual(group['effective_semantic_reward_groups'], 0)


if __name__ == '__main__':
    unittest.main()
