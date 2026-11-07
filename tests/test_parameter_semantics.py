import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoformalization import source_payload, student_prompt as original_prompt
from generate_smt_pairs import generate_one, publish
from parameter_semantics import RULES, comment_coverage, student_prompt
from rl_rewards import make_grpo_reward


class ParameterSemanticsTests(unittest.TestCase):
    def test_each_parameter_requires_its_own_meaning(self):
        parameters = {'ratio': 'Real', 'quarter_end': 'Bool'}
        result = comment_coverage('; All inputs follow the source.\n; ratio: stated Leverage Ratio.\n', parameters)
        self.assertFalse(result['complete'])
        self.assertEqual(result['missing_parameters'], ['quarter_end'])
        result = comment_coverage('; ratio: stated Leverage Ratio.\n; quarter_end: true at each fiscal quarter end.\n', parameters)
        self.assertTrue(result['complete'])

    def test_empty_duplicate_or_unrelated_comments_do_not_complete_schema(self):
        self.assertFalse(comment_coverage('; ratio:   \n; other: a number.\n', {'ratio': 'Real'})['complete'])
        result = comment_coverage('; ratio: leverage ratio.\n; ratio: interest coverage ratio.\n', {'ratio': 'Real'})
        self.assertEqual(result['duplicate_parameters'], ['ratio'])
        self.assertFalse(result['complete'])

    def test_inline_declaration_comments_are_equally_valid(self):
        result = comment_coverage('(ratio Real) ; Stated leverage ratio.\n(quarter_end Bool) ; End of the Borrower fiscal quarter.\n',
                                  {'ratio': 'Real', 'quarter_end': 'Bool'})
        self.assertTrue(result['complete'])
        self.assertEqual(result['meanings']['ratio'], 'Stated leverage ratio.')

    def test_standardization_preserves_source_and_original_prompt(self):
        record = {'text': 'The leverage ratio shall not exceed 3.5.'}
        before = original_prompt(record)
        changed = student_prompt(record)
        self.assertEqual(changed[1], before[1])
        self.assertIn(RULES, changed[0]['content'])
        self.assertEqual(original_prompt(record), before)

    def test_reward_uses_requested_judge_prompt_without_changing_scale(self):
        class Client:
            model = 'test'
            def chat(self, messages, **kwargs):
                self.messages = messages
                return '{"equivalent": true, "complete": true, "no_unsupported_assumptions": true, "needs_context": false, "issues": [], "reason": "Source correspondence checked."}'
        client = Client()
        reward = make_grpo_reward(client, judge_prompt='Shared parameter meanings are required.')
        smt = '(set-logic QF_LRA) (define-fun covenant_holds ((ratio Real)) Bool (<= ratio 3.5))'
        self.assertEqual(reward([smt], ['Ratio shall not exceed 3.5.']), [1.0])
        self.assertEqual(client.messages[0].content, 'Shared parameter meanings are required.')

    def test_teacher_target_without_comments_is_not_accepted(self):
        candidate = {'status': 'formalized', 'smt2': '(set-logic QF_LRA) (define-fun covenant_holds ((ratio Real)) Bool (<= ratio 3.5))'}
        with patch('generate_smt_pairs.request', return_value=json.dumps(candidate)) as request, \
                patch('verifiqa.generation.llm_client.CodexCliClient'):
            result = generate_one({'id': 'r', 'text': 'Ratio <= 3.5.'}, 'train', 'teacher', 'judge', 'medium', True)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['rejection'], 'missing_or_duplicate_parameter_comments')
        self.assertEqual(request.call_count, 1)

    def test_separate_sft_output_uses_new_prompt_and_preserves_sources(self):
        record = {'id': 'r', 'contract_id': 'c', 'text': 'Ratio shall not exceed 3.5.'}
        audit = {'id': 'r', 'split': 'train', 'accepted': True, 'parameter_comments': True,
                 'source_sha256': hashlib.sha256(json.dumps(source_payload(record), sort_keys=True).encode()).hexdigest(),
                 'teacher_model': 'teacher', 'judge_model': 'judge', 'judge_independence': 'separate_model',
                 'candidate': {'smt2': '; ratio: stated ratio.\n(set-logic QF_LRA) (define-fun covenant_holds ((ratio Real)) Bool (<= ratio 3.5))'}}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'; source.mkdir()
            output = Path(directory)/'new'; output.mkdir()
            original = json.dumps(record)+'\n'
            (source/'train_unlabeled.jsonl').write_text(original)
            for split in ('validation', 'test'):
                (source/f'{split}.jsonl').write_text('')
            publish(output, {'train:r': audit}, source)
            pair = json.loads((output/'sft_synthetic.jsonl').read_text())
            self.assertEqual(pair['messages'][:-1], student_prompt(record))
            self.assertEqual((source/'train_unlabeled.jsonl').read_text(), original)
            self.assertFalse((source/'sft_synthetic.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
