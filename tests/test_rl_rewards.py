"""Baseline reward behavior with controlled judge responses; no model calls."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from autoformalization import CONVENTION, PROTOCOL, JUDGE
from rl_rewards import JudgeUnavailableError, InsufficientContextError, DENSE_RUBRIC, make_grpo_reward, score_autoformalization
from train_grpo import make_logged_reward

SOURCE = 'The borrower must maintain a leverage ratio not exceeding 3.5 at every fiscal quarter end.'
GOOD = '''(set-logic QF_LRA)
(define-fun covenant_holds ((leverage_ratio Real) (is_quarter_end Bool)) Bool
  (=> is_quarter_end (<= leverage_ratio 3.5)))'''
ACCEPT = dict(equivalent=True, complete=True, no_unsupported_assumptions=True,
              needs_context=False, issues=[], reason='The restriction and applicability match.')
REJECT = dict(ACCEPT, equivalent=False, issues=['Changed restriction or applicability.'])


class ScriptedJudge:
    model = 'test-judge-v1'

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []
        self.run_metadata = {'reasoning_effort': 'medium'}

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        answer = next(self.responses)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer) if not isinstance(answer, str) else answer


class RewardTests(unittest.TestCase):
    def test_dense_grades_and_exact_endpoints(self):
        for grade in range(4):
            verdict = dict(ACCEPT if grade == 3 else REJECT, semantic_score=grade)
            result = score_autoformalization(GOOD, {'text': SOURCE}, lambda *_: verdict, dense=True)
            self.assertAlmostEqual(result['reward'], [.2, 7/15, 11/15, 1.][grade])
            if grade == 3:
                self.assertEqual(result['reward'], 1.)
        # Additional grading fields do not change the historical binary default.
        verdict = dict(REJECT, semantic_score=2)
        self.assertEqual(score_autoformalization(GOOD, {'text': SOURCE}, lambda *_: verdict)['reward'], 0.)

    def test_dense_invalid_output_never_calls_judge(self):
        client = ScriptedJudge()
        self.assertEqual(make_grpo_reward(client, dense=True)([GOOD[:-1]], [SOURCE]), [0.])
        self.assertEqual(client.calls, [])

    def test_dense_schema_and_consistency_failures_retry_without_caching(self):
        invalid = [dict(REJECT), *[dict(REJECT, semantic_score=s) for s in [True, 1.0, '2', -1, 4]],
                   dict(REJECT, semantic_score=3), dict(ACCEPT, semantic_score=2),
                   dict(REJECT, needs_context=True, semantic_score=0)]
        for verdict in invalid:
            with self.subTest(verdict=verdict):
                client = ScriptedJudge(verdict, dict(REJECT, semantic_score=2))
                reward = make_grpo_reward(client, dense=True, judge_prompt=JUDGE + DENSE_RUBRIC)
                with self.assertLogs('rl_rewards', level='WARNING'):
                    self.assertAlmostEqual(reward([GOOD], [SOURCE])[0], 11/15)
                self.assertAlmostEqual(reward([GOOD], [SOURCE])[0], 11/15)
                self.assertEqual(len(client.calls), 2)
                self.assertIn(DENSE_RUBRIC, client.calls[0][0].content)

    def test_dense_context_abstention_aborts_batch_without_repeated_call(self):
        verdict = dict(REJECT, needs_context=True, semantic_score=None)
        client = ScriptedJudge(dict(ACCEPT, semantic_score=3), verdict)
        reward = make_grpo_reward(client, dense=True)
        with self.assertRaises(InsufficientContextError) as caught:
            reward([GOOD, GOOD.replace('3.5', '3.6')], [SOURCE, SOURCE])
        self.assertEqual(caught.exception.verdict, verdict)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(make_grpo_reward(ScriptedJudge(verdict))([GOOD], [SOURCE]), [0.])

    def test_dense_probe_keeps_known_context_cases_unscored(self):
        from probe_rewards import score_one, summarize
        prediction = {'id': 'fin_1', 'smt2': GOOD, 'hit_token_limit': False}
        with patch('verifiqa.generation.llm_client.CodexCliClient',
                   return_value=ScriptedJudge(dict(ACCEPT, semantic_score=3))):
            result = score_one({'text': SOURCE}, prediction, False, dense=True, exclude_semantic=True)
        self.assertIsNone(result['reward'])
        self.assertIsNone(result['semantic_score'])
        self.assertEqual(result['status'], 'context_excluded')
        self.assertEqual(result['semantic']['semantic_score'], 3)  # Raw verdict preserved, never used as a label.
        summary = summarize([result])
        self.assertEqual(summary['graded_valid'], 0)
        self.assertEqual(summary['semantic_accepted'], 0)
        self.assertEqual(summary['context_excluded'], 1)

    def test_non_temporal_probe_records_reduced_scope(self):
        from probe_rewards import score_one, summarize, NON_TEMPORAL_SCOPE
        client = ScriptedJudge(dict(ACCEPT, semantic_score=3))
        prediction = {'id': 'fin_1', 'smt2': GOOD, 'hit_token_limit': False}
        with patch('verifiqa.generation.llm_client.CodexCliClient', return_value=client):
            result = score_one({'text': SOURCE}, prediction, False, dense=True, non_temporal=True)
        self.assertTrue(client.calls[0][0].content.endswith(NON_TEMPORAL_SCOPE))
        self.assertEqual(result['semantic_scope'], 'non_temporal')
        summary = summarize([result])
        self.assertEqual(summary['non_temporal_accepted'], 1)
        self.assertIn('not comparable', summary['limitation'])

    def test_dense_probe_separates_new_abstention_and_technical_failure(self):
        from probe_rewards import score_one
        prediction = {'id': 'fin_1', 'smt2': GOOD, 'hit_token_limit': False}
        with patch('verifiqa.generation.llm_client.CodexCliClient',
                   return_value=ScriptedJudge(dict(REJECT, needs_context=True, semantic_score=None))):
            result = score_one({'text': SOURCE}, prediction, False, dense=True)
        self.assertEqual(result['status'], 'judge_abstained')
        self.assertIsNone(result['reward'])
        self.assertEqual(result['judge_failed_attempts'], 0)
        with patch('verifiqa.generation.llm_client.CodexCliClient',
                   return_value=ScriptedJudge(TimeoutError(), TimeoutError())), self.assertLogs('rl_rewards', level='WARNING'):
            result = score_one({'text': SOURCE}, prediction, False, dense=True, exclude_semantic=True)
        self.assertEqual(result['status'], 'judge_failure')
        self.assertEqual(result['judge_failed_attempts'], 2)

    def test_conversational_completion_through_logged_callback(self):
        from autoformalization import validate_smt
        client = ScriptedJudge(ACCEPT, REJECT)
        completions = [[{'role': 'assistant', 'content': smt}]
                       for smt in [GOOD, GOOD.replace('3.5', '3.6')]]
        assumptions = ['All relevant quarter ends are supplied.']
        context = ['Leverage Ratio is the ratio stated in this covenant.']
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            output = Path(directory)
            callback = make_logged_reward(make_grpo_reward(client), output)
            with patch('rl_rewards.validate_smt', wraps=validate_smt) as validator:
                scores = callback(completions, [SOURCE] * 2, id=['fin_1'] * 2,
                                  declared_assumptions=[assumptions] * 2, source_context=[context] * 2,
                                  trainer_state=SimpleNamespace(global_step=3))
            self.assertEqual(scores, [1., 0.])
            self.assertEqual([call.args[0] for call in validator.call_args_list],
                             [GOOD, GOOD.replace('3.5', '3.6')])
            payload = json.loads(client.calls[0][1].content)
            self.assertEqual(payload['covenant'], SOURCE)
            self.assertEqual(payload['candidate_smt2'], GOOD)
            self.assertEqual(payload['declared_assumptions'], assumptions)
            self.assertEqual(payload['source_context'], context)
            self.assertEqual(payload['formalization_protocol'], PROTOCOL)
            rows = [json.loads(line) for line in (output / 'rewards.jsonl').read_text().splitlines()]
            self.assertEqual([r['completion'] for r in rows], completions)
            groups = json.loads((output / 'reward_groups.jsonl').read_text())
            self.assertEqual(groups['informative_groups'], 1)
            self.assertEqual(groups['complete_groups'], 1)
            self.assertEqual(groups['informative_fraction'], 1.)

    def test_group_variation_uses_ids_not_batch_reward_distribution(self):
        with tempfile.TemporaryDirectory() as directory, patch('builtins.print'):
            output = Path(directory)
            callback = make_logged_reward(lambda **_: [0., .2, 1., 0., 1., 1., 0.], output)
            # Interleaved IDs: only b has variation; d is incomplete and must be excluded.
            callback(['smt'] * 7, [SOURCE] * 7, id=['a', 'b', 'c', 'a', 'b', 'c', 'd'])
            groups = json.loads((output / 'reward_groups.jsonl').read_text())
            self.assertEqual(groups['complete_groups'], 3)
            self.assertEqual(groups['informative_groups'], 1)
            self.assertEqual(groups['informative_fraction'], 1 / 3)
            self.assertEqual(groups['incomplete_groups'], 1)

    def test_controlled_correct_and_incorrect_formulas(self):
        candidates = [GOOD, GOOD.replace('3.5', '3.6'), GOOD.replace('<=', '>='),
                      GOOD.replace('(=> is_quarter_end (<= leverage_ratio 3.5))',
                                   '(<= leverage_ratio 3.5)')]
        client = ScriptedJudge(ACCEPT, REJECT, REJECT, REJECT)
        self.assertEqual(make_grpo_reward(client)(candidates, [SOURCE] * 4), [1., 0., 0., 0.])
        self.assertEqual(len(client.calls), 4)

    def test_invalid_trivial_and_unsupported_outputs_never_call_judge(self):
        candidates = [GOOD[:-1],
                      '(set-logic QF_LRA)(define-fun covenant_holds () Bool true)',
                      GOOD.replace('(=> is_quarter_end (<= leverage_ratio 3.5))', 'true'),
                      GOOD.replace('(=> is_quarter_end (<= leverage_ratio 3.5))', 'false'),
                      GOOD.replace('Real', 'Int'),
                      GOOD.replace('3.5', 'true'), GOOD + '(check-sat)',
                      GOOD.replace('(<= leverage_ratio 3.5)', '(forall ((x Real)) (<= x 3.5))')]
        client = ScriptedJudge()
        self.assertEqual(make_grpo_reward(client)(candidates, [SOURCE] * len(candidates)),
                         [0.] * len(candidates))
        self.assertEqual(client.calls, [])

    def test_judge_receives_protocol_context_and_assumptions(self):
        client = ScriptedJudge(ACCEPT)
        context = [{'text': 'Leverage Ratio means total debt divided by EBITDA.'}]
        assumptions = ['The supplied history contains all fiscal quarter ends.']
        reward = make_grpo_reward(client)
        reward([GOOD], [SOURCE], declared_assumptions=[assumptions], source_context=[context])
        system, message = client.calls[0]
        payload = json.loads(message.content)
        self.assertEqual(system.content, JUDGE)
        self.assertIn(CONVENTION, system.content)
        self.assertEqual(payload['formalization_protocol'], PROTOCOL)
        self.assertEqual(payload['instantiation_convention'], CONVENTION)
        self.assertEqual(payload['source_context'], context)
        self.assertEqual(payload['declared_assumptions'], assumptions)

    def test_json_schema_and_transport_failures_retry(self):
        malformed = ['not JSON', 'null', '[]', '{}', dict(ACCEPT, equivalent='true'),
                     dict(ACCEPT, complete=1), dict(ACCEPT, issues=''),
                     dict(ACCEPT, issues=[2]), dict(ACCEPT, reason=' '), TimeoutError('offline')]
        for answer in malformed:
            with self.subTest(answer=answer):
                events = []
                client = ScriptedJudge(answer, ACCEPT)
                reward = make_grpo_reward(client, events.append)
                with self.assertLogs('rl_rewards', level='WARNING'):
                    self.assertEqual(reward([GOOD], [SOURCE]), [1.])
                self.assertEqual(len(client.calls), 2)
                self.assertEqual(events[0]['status'], 'judge_failure')
                self.assertIsNone(events[0]['reward'])

    def test_exhausted_judge_failure_aborts_batch_and_is_not_cached(self):
        client = ScriptedJudge(ACCEPT, TimeoutError('offline'), 'malformed', REJECT)
        reward = make_grpo_reward(client)
        wrong = GOOD.replace('3.5', '3.6')
        with self.assertLogs('rl_rewards', level='WARNING'), self.assertRaises(JudgeUnavailableError):
            reward([GOOD, wrong], [SOURCE, SOURCE])
        # The successful first judgment survives, but the failed second one must be retried.
        self.assertEqual(reward([GOOD, wrong], [SOURCE, SOURCE]), [1., 0.])
        self.assertEqual(len(client.calls), 4)

    def test_direct_scoring_rejects_malformed_feedback(self):
        with self.assertRaises(JudgeUnavailableError):
            score_autoformalization(GOOD, {'text': SOURCE}, lambda *_: {'equivalent': False})

    def test_identical_completions_and_rejections_are_cached(self):
        client = ScriptedJudge(REJECT)
        reward = make_grpo_reward(client)
        chat_completion = [{'role': 'assistant', 'content': GOOD}]
        self.assertEqual(reward([GOOD, chat_completion, GOOD], [SOURCE] * 3), [0.] * 3)
        self.assertEqual(reward([GOOD], [SOURCE]), [0.])
        self.assertEqual(len(client.calls), 1)

    def test_cache_separates_evidence_candidate_model_and_prompt(self):
        client = ScriptedJudge(*([ACCEPT] * 8))
        reward = make_grpo_reward(client)
        reward([GOOD], [SOURCE])
        reward([GOOD], [SOURCE + ' Additional source.'])
        reward([GOOD], [SOURCE], declared_assumptions=[['Explicit assumption']])
        reward([GOOD], [SOURCE], source_context=[['A relevant definition']])
        reward([GOOD.replace('3.5', '3.6')], [SOURCE])
        client.model = 'test-judge-v2'
        reward([GOOD], [SOURCE])
        client.run_metadata['reasoning_effort'] = 'high'
        reward([GOOD], [SOURCE])
        with patch('rl_rewards.JUDGE', JUDGE + '\nRevised judge rubric.'):
            reward([GOOD], [SOURCE])
        self.assertEqual(len(client.calls), 8)

    def test_cache_is_bounded_and_audit_cannot_mutate_it(self):
        client = ScriptedJudge(ACCEPT, ACCEPT, ACCEPT)
        def mutate_audit(result):
            result['semantic']['equivalent'] = False
        reward = make_grpo_reward(client, mutate_audit, cache_size=1)
        self.assertEqual(reward([GOOD], [SOURCE]), [1.])
        self.assertEqual(reward([GOOD], [SOURCE]), [1.])
        reward([GOOD], [SOURCE + ' Another covenant.'])
        self.assertEqual(reward([GOOD], [SOURCE]), [1.])
        self.assertEqual(len(client.calls), 3)

    def test_misaligned_source_columns_fail_before_judging(self):
        client = ScriptedJudge()
        reward = make_grpo_reward(client)
        for kwargs in [{'text': []}, {'text': [SOURCE], 'declared_assumptions': []},
                       {'text': [SOURCE], 'source_context': []}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                reward([GOOD], **kwargs)
        self.assertEqual(client.calls, [])


if __name__ == '__main__':
    unittest.main()
