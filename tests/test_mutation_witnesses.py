import unittest
from audit_mutation_witnesses import check_schema, mutate, partial_holds, value_map
from audit_witness_feasibility import distinguish
from pointwise import commands

RULE = '''(set-logic QF_LRA)
; quarter_end is the fiscal quarter end.
(define-fun covenant_holds ((ratio Real) (quarter_end Bool)) Bool
  (=> quarter_end (<= ratio 4.0)))'''


class MutationTests(unittest.TestCase):
    def test_two_local_mutations_preserve_input_schema(self):
        original = commands(RULE)
        mutations = mutate(RULE)
        self.assertEqual([m['kind'] for m in mutations], ['strictness', 'threshold_minus_0.01'])
        self.assertEqual(mutations[1]['after'], '3.99')
        for mutation in mutations:
            self.assertTrue(mutation['validation']['valid'])
            self.assertEqual(commands(mutation['smt2'])[1][2], original[1][2])
            self.assertIn('; quarter_end is the fiscal quarter end.', mutation['smt2'])
        self.assertEqual(commands(RULE), original)

    def test_mutation_is_a_test_source_not_an_assumed_reference(self):
        mapping = dict(ratio='ratio', quarter_end='quarter_end')
        alignment = dict(mapping_a=mapping, mapping_b=mapping,
                         symbols=[dict(name='ratio', sort='Real'), dict(name='quarter_end', sort='Bool')])
        state = distinguish(RULE, mutate(RULE)[0]['smt2'], alignment)
        self.assertEqual(state['status'], 'distinguishing_state')
        self.assertTrue(state['candidate_a_holds'])
        # The source's actual 3.5 limit rejects this state although the student allows it.
        reference = RULE.replace('4.0', '3.5')
        self.assertFalse(partial_holds(reference, mapping, state['values'])['holds'])

    def test_only_one_repeated_threshold_occurrence_changes(self):
        rule = RULE.replace('(<= ratio 4.0)', '(and (<= ratio 4.0) (>= ratio (- 4.0)))')
        changed = commands(mutate(rule)[1]['smt2'])[1][4]
        self.assertEqual(changed[2][1], ['<=', 'ratio', '3.99'])
        self.assertEqual(changed[2][2], ['>=', 'ratio', ['-', '4.0']])

    def test_boolean_only_fallback_keeps_types(self):
        rule = '(set-logic QF_LRA)(define-fun covenant_holds ((q Bool) (active Bool)) Bool (and q active))'
        changes = mutate(rule)
        self.assertEqual(len(changes), 2)
        self.assertTrue(all(m['validation']['valid'] for m in changes))
        self.assertEqual(changes[0]['kind'], 'boolean_input_negation')

    def test_missing_inputs_are_not_fabricated(self):
        mapping = dict(ratio='ratio', quarter_end='quarter_end')
        self.assertIsNone(partial_holds(RULE, mapping, {'ratio': '5'})['holds'])
        self.assertTrue(partial_holds(RULE, mapping, {'ratio': '3'})['holds'])
        ratio = '(set-logic QF_NRA)(define-fun covenant_holds ((x Real) (d Real)) Bool (<= (/ x d) 3))'
        self.assertIsNone(partial_holds(ratio, {'x': 'x', 'd': 'd'}, {'x': '1', 'd': '0'})['holds'])

    def test_conflicting_aliases_and_schema_changes_are_rejected(self):
        with self.assertRaises(ValueError): value_map({'a': True, 'b': False}, {'a': 'x', 'b': 'x'})
        self.assertEqual(value_map({'a': '0.5', 'b': '1/2'}, {'a': 'x', 'b': 'x'}), {'x': '1/2'})
        with self.assertRaises(ValueError):
            check_schema(dict(status='grounded', reason='renamed', symbols=[dict(name='other', sort='Real')]),
                         {'text': 'ratio'}, {'ratio': 'Real'})


if __name__ == '__main__':
    unittest.main()
