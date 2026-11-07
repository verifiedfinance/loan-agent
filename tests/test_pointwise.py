import unittest
from pointwise import evaluate_history, validate_smt, CONVENTION
from evaluate_smt_predictions import compare

QUARTER = '''(set-logic QF_LRA)
(define-fun covenant_holds ((leverage_ratio Real) (is_quarter_end Bool)) Bool
  (=> is_quarter_end (<= leverage_ratio 3.5)))'''


class PointwiseTests(unittest.TestCase):
    def test_user_three_quarter_example(self):
        self.assertTrue(validate_smt(QUARTER)['valid'])
        result = evaluate_history(QUARTER, [dict(leverage_ratio=x, is_quarter_end=True)
                                           for x in ('3.2', '3.4', '3.8')])
        self.assertFalse(result['compliant'])
        self.assertEqual(result['failing_observation_indices'], [2])

    def test_applicability_and_boundary(self):
        result = evaluate_history(QUARTER, [dict(leverage_ratio='9', is_quarter_end=False),
                                           dict(leverage_ratio='3.5', is_quarter_end=True)])
        self.assertTrue(result['compliant'])
        stricter = QUARTER.replace('<=', '<')
        self.assertEqual(compare(stricter, QUARTER)['status'], 'differs_from_reference')

    def test_no_fabricated_calendar_values(self):
        with self.assertRaises(ValueError): evaluate_history(QUARTER, [dict(leverage_ratio='3.2')])
        with self.assertRaises(ValueError): evaluate_history(QUARTER, [dict(leverage_ratio='3.2', is_quarter_end='true')])
        self.assertTrue(evaluate_history(QUARTER, [])['empty_history'])
        self.assertIn('finite financial history', CONVENTION)

    def test_undefined_division_and_inapplicable_case(self):
        smt = '(set-logic QF_NRA)(define-fun covenant_holds ((debt Real) (earnings Real) (applicable Bool)) Bool (=> applicable (<= (/ debt earnings) 3.5)))'
        self.assertIsNone(evaluate_history(smt, [dict(debt='10', earnings='0', applicable=True)])['compliant'])
        self.assertTrue(evaluate_history(smt, [dict(debt='10', earnings='0', applicable=False)])['compliant'])

    def test_equivalence_accounts_for_definedness(self):
        a = '(set-logic QF_NRA)(define-fun covenant_holds ((x Real) (d Real)) Bool (<= (/ x d) 3))'
        b = '(set-logic QF_NRA)(define-fun covenant_holds ((x Real) (d Real)) Bool (<= x (* 3 d)))'
        self.assertEqual(compare(a, b)['status'], 'differs_from_reference')
        # Guarding a ratio means zero denominators at inapplicable times do not
        # create a compliance failure or require a positivity assumption.
        result = evaluate_history(a, [dict(x='1', d='-1')])
        self.assertTrue(result['compliant'])

    def test_function_required_and_no_extra_commands(self):
        self.assertFalse(validate_smt('(set-logic QF_LRA)(declare-const x Real)(assert (<= x 3))')['valid'])
        self.assertFalse(validate_smt(QUARTER + '(check-sat)')['valid'])
        self.assertFalse(validate_smt(QUARTER.replace('leverage_ratio 3.5', 'leverage_ratio true'))['valid'])


if __name__ == '__main__': unittest.main()
