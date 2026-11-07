import unittest
from autoformalization import validate_smt, judge_accepts, student_prompt
from build_rl_datasets import grouped_split
from rl_rewards import score_autoformalization
from evaluate_smt_predictions import compare

GOOD = '(set-logic QF_LRA)\n(define-fun covenant_holds ((leverage_ratio Real)) Bool (<= leverage_ratio 3.50))'
ACCEPT = dict(equivalent=True, complete=True, no_unsupported_assumptions=True,
              needs_context=False, issues=[], reason='Threshold and inequality match.')


class AutoformalizationTests(unittest.TestCase):
    def test_validity_is_not_semantics(self):
        self.assertTrue(validate_smt(GOOD)['valid'])
        wrong = GOOD.replace('<=', '>=')
        self.assertTrue(validate_smt(wrong)['valid'])
        reject = dict(ACCEPT, equivalent=False, issues=['Reversed inequality'])
        self.assertEqual(score_autoformalization(wrong, {'text': 'not exceed 3.50'}, lambda *_: reject)['reward'], 0.)
        self.assertEqual(score_autoformalization(GOOD, {'text': 'not exceed 3.50'}, lambda *_: ACCEPT)['reward'], 1.)

    def test_invalid_and_degenerate(self):
        for bad in [GOOD + '\n(reset)', GOOD.replace('3.50', 'true'), GOOD.replace('(<= leverage_ratio 3.50)', 'true'),
                    GOOD.replace('(<= leverage_ratio 3.50)', 'false'), GOOD.replace('leverage_ratio 3.50', 'missing 3.50'),
                    '(set-logic QF_LRA)(declare-const x Real)(assert (<= (* x x) 3))']:
            self.assertFalse(validate_smt(bad)['valid'], bad)
        self.assertFalse(judge_accepts(dict(ACCEPT, equivalent='true')))
        self.assertFalse(judge_accepts(dict(ACCEPT, no_unsupported_assumptions=False)))

    def test_no_added_assumptions(self):
        prompt = student_prompt({'text': 'Leverage ratio shall not exceed 3.50.'})
        self.assertIn('"declared_assumptions": []', prompt[1]['content'])
        self.assertNotIn('(>= leverage_ratio 0)', GOOD)
        self.assertFalse(validate_smt('; NEEDS_CONTEXT missing definition')['valid'])

    def test_equivalence_and_counterexamples(self):
        same = GOOD.replace('(<= leverage_ratio 3.50)', '(>= 3.50 leverage_ratio)')
        self.assertEqual(compare(same, GOOD)['status'], 'equivalent_to_reference')
        self.assertEqual(compare(GOOD.replace('3.50', '3.60'), GOOD)['status'], 'differs_from_reference')
        renamed = GOOD.replace('leverage_ratio', 'ratio')
        self.assertEqual(compare(renamed, GOOD)['status'], 'needs_symbol_alignment')
        self.assertEqual(compare(renamed, GOOD, {'ratio': 'leverage_ratio'})['status'], 'equivalent_to_reference')
        # An unsupported nonnegativity assumption really changes the formula.
        extra = GOOD.replace('(<= leverage_ratio 3.50)', '(and (<= leverage_ratio 3.50) (>= leverage_ratio 0))')
        self.assertEqual(compare(extra, GOOD)['status'], 'differs_from_reference')

    def test_grouped_split(self):
        rows = [{'id': str(i), 'contract_id': str(i // 2), 'text': f'Rule {i}'} for i in range(100)]
        rows[21]['text'] = rows[0]['text']
        splits, _, _ = grouped_split(rows, 42)
        seen_contracts, seen_text = {}, {}
        for split, records in splits.items():
            for row in records:
                self.assertEqual(seen_contracts.setdefault(row['contract_id'], split), split)
                self.assertEqual(seen_text.setdefault(row['text'], split), split)
        self.assertEqual(sum(map(len, splits.values())), 100)
        self.assertEqual(splits, grouped_split(rows, 42)[0])


if __name__ == '__main__':
    unittest.main()
