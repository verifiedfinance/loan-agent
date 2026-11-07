"""Headroom experiment core (PLAN.md): spec format, reference answers, certificate text, SMT rendering of a spec,
and the Z3 evaluation used by BOTH the reference validation and the formula arm.

Spec (one clause):
  items:  [{name, description, role: earnings|debt|other}]        line items the certificate reports (Real params)
  flags:  [{name, description}]                                   condition facts (Bool params)
  limits: [{label, numerator: [[item, +1|-1], ...], denominator: [...] (empty = level limit), op: <=|<|>=|>,
            unit: ratio|pct|usd, schedule: [{from: YYYY-MM-DD|null, to: YYYY-MM-DD|null, threshold}],
            when: null | {flag, value}}]
A limit applies on a test date when a schedule step covers the date and its condition flag (if any) has the value.

A case gives values for every item (one may be missing -> None), test_date (YYYYMMDD int) and flag values.
Answers: status (comply|breach|undetermined), cushion (% decrease of the earnings item that keeps every limit
satisfied; 'none' | 'undetermined'), capacity (USD increase of the debt item; 'none' | 'undetermined').
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import re
from fractions import Fraction

import z3

OPS = {'<=': lambda a, b: a <= b, '<': lambda a, b: a < b, '>=': lambda a, b: a >= b, '>': lambda a, b: a > b}
ITERS = 80


# --------------------------------------------------------------------------- spec evaluation (exact, Fractions)
def date_int(s):
    return int(s.replace('-', '')) if s else None


def threshold_on(limit, date):
    for step in limit['schedule']:
        lo, hi = date_int(step.get('from')), date_int(step.get('to'))
        if (lo is None or date >= lo) and (hi is None or date <= hi):
            return Fraction(str(step['threshold']))
    return None


def applies(limit, date, flags):
    w = limit.get('when')
    return threshold_on(limit, date) is not None and (not w or flags.get(w['flag']) == w['value'])


def side(terms, values):
    return sum((Fraction(s) * values[i] for i, s in terms), Fraction(0))


def limit_holds(limit, values, date):
    num = side(limit['numerator'], values)
    if limit['denominator']:
        den = side(limit['denominator'], values)
        if den <= 0:
            return False  # a ratio with a non-positive denominator fails (e.g. negative EBITDA breaches leverage)
        return OPS[limit['op']](num / den, threshold_on(limit, date))
    return OPS[limit['op']](num, threshold_on(limit, date))


def spec_status(spec, values, date, flags):
    """comply / breach / undetermined; an item with value None is unknown."""
    unknown = False
    for lim in spec['limits']:
        if not applies(lim, date, flags):
            continue
        used = {i for i, _ in lim['numerator'] + lim['denominator']}
        if any(values.get(i) is None for i in used):
            unknown = True
            continue
        if not limit_holds(lim, values, date):
            return 'breach'
    return 'undetermined' if unknown else 'comply'


# --------------------------------------------------------------------------- generic headroom procedure
def domain(spec, values, date, flags, item, kind):
    """Search range for the varied item: (0, 100 x0] for the cushion, [0, 100 x0 + 1e10] for the capacity, narrowed so
    that every applicable ratio's denominator in the reference reading stays positive. Used by BOTH arms."""
    x0 = Fraction(values[item])
    lo, hi = (x0 / 10**6, x0 * 100) if kind == 'cushion' else (Fraction(0), x0 * 100 + 10**10)
    for lim in spec['limits']:
        if not lim['denominator'] or not applies(lim, date, flags):
            continue
        s = sum((Fraction(sg) for i, sg in lim['denominator'] if i == item), Fraction(0))
        if s == 0 or any(values.get(i) is None for i, _ in lim['denominator'] if i != item):
            continue
        rest = sum((Fraction(sg) * Fraction(values[i]) for i, sg in lim['denominator'] if i != item), Fraction(0))
        zero = -rest / s  # denominator = rest + s * x
        if s > 0:
            lo = max(lo, zero + abs(zero) / 10**9 + Fraction(1, 10**6))
        else:
            hi = min(hi, zero - abs(zero) / 10**9 - Fraction(1, 10**6))
    return lo, hi


def _search(feasible, x0, kind, bounds=None):
    """kind='cushion': smallest feasible x in the domain; kind='capacity': largest feasible x in the domain.
    feasible(x) -> True/False. Returns the boundary (Fraction) or 'none' / 'undetermined'."""
    x0 = Fraction(x0)
    if kind == 'cushion':
        lo, hi = bounds or (x0 / 10**6, x0 * 100)
        if feasible(lo):
            return 'none'
        if not feasible(hi):
            return 'undetermined'
        for _ in range(ITERS):  # invariant: lo infeasible, hi feasible
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if feasible(mid) else (mid, hi)
            lo, hi = Fraction(float(lo)), Fraction(float(hi))
        return hi
    lo, hi = bounds or (Fraction(0), x0 * 100 + 10**10)
    if feasible(hi):
        return 'none'
    if not feasible(lo):
        return 'undetermined'
    for _ in range(ITERS):  # invariant: lo feasible, hi infeasible
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if feasible(mid) else (lo, mid)
        lo, hi = Fraction(float(lo)), Fraction(float(hi))
    return lo


def headroom(feasible_with, status, values, earnings, debt, bounds_for=None):
    """feasible_with(item, x) -> bool (certainly compliant with item set to x). Returns (cushion, capacity)."""
    out = []
    for item, kind in ((earnings, 'cushion'), (debt, 'capacity')):
        if item is None:
            out.append(None)  # question not asked
        elif status == 'undetermined' or values.get(item) is None:
            out.append('undetermined')
        else:
            b = _search(lambda x: feasible_with(item, x), values[item], kind, bounds_for(item, kind) if bounds_for else None)
            if isinstance(b, str):
                out.append(b)
            elif kind == 'cushion':
                out.append(float((Fraction(values[item]) - b) / Fraction(values[item]) * 100))
            else:
                out.append(float(b - Fraction(values[item])))
    return tuple(out)


def roles(spec):
    pick = lambda r: next((i['name'] for i in spec['items'] if i['role'] == r), None)
    return pick('earnings'), pick('debt')


def reference_answers(spec, values, date, flags):
    status = spec_status(spec, values, date, flags)

    def feasible_with(item, x):
        v = dict(values)
        v[item] = Fraction(x)
        return spec_status(spec, v, date, flags) == 'comply'
    e, d = roles(spec)
    cushion, capacity = headroom(feasible_with, status, values, e, d, lambda i, k: domain(spec, values, date, flags, i, k))
    return dict(status=status, cushion=cushion, capacity=capacity)


# --------------------------------------------------------------------------- SMT
def num(x):
    f = Fraction(str(x)) if not isinstance(x, Fraction) else x
    s = f'(/ {abs(f.numerator)}.0 {f.denominator}.0)' if f.denominator != 1 else f'{abs(f.numerator)}.0'
    return f'(- {s})' if f < 0 else s


def params(spec):
    return [(i['name'], 'Real') for i in spec['items']] + [('test_date', 'Int')] + [(f['name'], 'Bool') for f in spec['flags']]


def spec_to_smt(spec):
    """The reference spec as an SMT covenant_holds over the case parameters (for validation)."""
    def sum_expr(terms):
        parts = [i if s > 0 else f'(- {i})' for i, s in terms]
        return parts[0] if len(parts) == 1 else f"(+ {' '.join(parts)})"
    conj = []
    for lim in spec['limits']:
        value = sum_expr(lim['numerator']) if not lim['denominator'] else f"(/ {sum_expr(lim['numerator'])} {sum_expr(lim['denominator'])})"
        steps = []
        for st in lim['schedule']:
            cond = [f'(>= test_date {date_int(st["from"])})'] if st.get('from') else []
            cond += [f'(<= test_date {date_int(st["to"])})'] if st.get('to') else []
            guard = 'true' if not cond else (cond[0] if len(cond) == 1 else f"(and {' '.join(cond)})")
            steps.append(f"(=> {guard} ({lim['op']} {value} {num(st['threshold'])}))")
        body = steps[0] if len(steps) == 1 else f"(and {' '.join(steps)})"
        w = lim.get('when')
        if w:
            body = f"(=> {w['flag'] if w['value'] else '(not ' + w['flag'] + ')'} {body})"
        conj.append(body)
    sig = ' '.join(f'({n} {t})' for n, t in params(spec))
    return f"(define-fun covenant_holds ({sig}) Bool (and {' '.join(conj)}))"


class FormulaEvaluator:
    """Z3 evaluation of any covenant_holds over the case's parameters (the formula arm, and reference validation)."""

    def __init__(self, formula, spec, lenient=False):
        body = re.sub(r'\(\s*set-logic[^)]*\)', '', formula)
        body = re.sub(r'\(\s*(check-sat|get-model|exit|reset|push|pop|set-option|set-info)[^)]*\)', '', body)
        self.names = dict(params(spec))
        decls = ''.join(f'(declare-const {n} {t})\n' for n, t in params(spec))
        sig = re.search(r'define-fun\s+covenant_holds\s*\(((?:\s*\(\s*[^()\s]+\s+[^()\s]+\s*\))*)\s*\)\s*Bool', body)
        if not sig:
            raise ValueError('no covenant_holds(...) Bool definition')
        got = re.findall(r'\(\s*([^()\s]+)\s+([^()\s]+)\s*\)', sig.group(1))
        if lenient:  # post-hoc sensitivity: any correctly typed subset of the required parameters, in any order
            if not got or len({n for n, _ in got}) != len(got) or any(self.names.get(n) != t for n, t in got):
                raise ValueError('covenant_holds parameters are not a typed subset of the required list')
        elif [n for n, _ in got] != [n for n, _ in params(spec)] or any(self.names[n] != t for n, t in got):
            raise ValueError('covenant_holds parameters differ from the required list')
        call = f"(covenant_holds {' '.join(n for n, _ in got)})"
        self.holds = z3.parse_smt2_string(body + decls + f'(assert {call})')
        self.fails = z3.parse_smt2_string(body + decls + f'(assert (not {call}))')
        self.const = {n: (z3.Real(n) if t == 'Real' else z3.Int(n) if t == 'Int' else z3.Bool(n)) for n, t in params(spec)}

    def _facts(self, values, date, flags):
        out = [self.const['test_date'] == date]
        out += [self.const[k] == z3.RealVal(str(Fraction(v))) for k, v in values.items() if v is not None]
        out += [self.const[k] == z3.BoolVal(bool(v)) for k, v in flags.items() if v is not None]
        return out

    def _sat(self, goal, facts):
        s = z3.Solver()
        s.set('timeout', 5000)
        s.add(goal)
        s.add(facts)
        r = s.check()
        if r == z3.unknown:
            raise RuntimeError('z3 unknown')
        return r == z3.sat

    def status(self, values, date, flags):
        f = self._facts(values, date, flags)
        h, nf = self._sat(self.holds, f), self._sat(self.fails, f)
        if h and not nf:
            return 'comply'
        if nf and not h:
            return 'breach'
        if h and nf:
            return 'undetermined'
        return 'inconsistent'

    def answers(self, spec, values, date, flags):
        status = self.status(values, date, flags)
        if status == 'inconsistent':
            return dict(status='error', cushion='error', capacity='error')

        def feasible_with(item, x):
            v = dict(values)
            v[item] = Fraction(x)
            return not self._sat(self.fails, self._facts(v, date, flags))  # certainly compliant
        e, d = roles(spec)
        cushion, capacity = headroom(feasible_with, status, values, e, d, lambda i, k: domain(spec, values, date, flags, i, k))
        return dict(status=status, cushion=cushion, capacity=capacity)


# --------------------------------------------------------------------------- scoring
def correct(question, pred, ref, values=None, debt=None):
    if ref is None:
        return None  # not asked
    if isinstance(ref, str) or isinstance(pred, str) or pred is None:
        return int(isinstance(pred, str) and pred == ref)
    if question == 'status':
        return int(pred == ref)
    if question == 'cushion':
        return int(abs(float(pred) - ref) <= 0.5)
    base = float(values[debt]) if values and debt and values.get(debt) is not None else abs(ref)
    return int(abs(float(pred) - ref) <= max(0.005 * base, 1.0))  # within 0.5% of the current debt


def case_score(case, ans):
    e, d = roles(case['spec_ref']) if 'spec_ref' in case else (case.get('earnings_item'), case.get('debt_item'))
    marks = [int(ans.get('status') == case['answers']['status'])]
    for q in ('cushion', 'capacity'):
        c = correct(q, ans.get(q), case['answers'][q], case['values'], d)
        if c is not None:
            marks.append(c)
    return sum(marks) / len(marks), marks
