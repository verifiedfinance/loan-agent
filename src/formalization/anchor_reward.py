"""Text-anchored, solver-verified reward for covenant autoformalization (no labels, no LLM judge).

Anchors: the deterministic extractor (downstream/reference_rules.extract) reads every numeric limit in the
clause text as (comply direction, threshold), e.g. "shall not permit the Leverage Ratio to exceed 3.50"
-> (<=, 3.5); "greater than or equal to 3.25 is prohibited" -> (<, 3.25).

For a candidate SMT formula, Z3 expands covenant_holds into a Boolean term H and the reward checks:
  recall      each anchor is realized by a comparison atom with the same constant and the same effective
              direction and strictness (polarity through not / implication antecedents is respected;
              `c * y` on one side counts as constant c, i.e. a ratio written multiplied out)
  liveness    a matched atom must matter: replacing it by true or by false changes H (Z3 equivalence check),
              so dummy constraints such as (or (<= x 3.5) true) earn nothing
  precision   live numeric atoms whose constant does not appear anywhere in the clause text (as a number,
              percentage or dollar amount) count against the formula (invented thresholds or assumptions)
  vacuity     H must be satisfiable and refutable (not always true, not always false)

anchor_score = F1(precision, recall) over live atoms; reward = 0 if invalid or vacuous, else anchor_score.
Clauses without extractable anchors return score None (the caller decides the fallback).
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import re
import sys
from fractions import Fraction
from pathlib import Path

import z3

from decide import signature  # noqa: E402
from reference_rules import extract  # noqa: E402

TIMEOUT_MS = 2000
NEGATE = {'<=': '>', '<': '>=', '>=': '<', '>': '<='}
FLIP = {'<=': '>=', '<': '>', '>=': '<=', '>': '<'}
KIND = {z3.Z3_OP_LE: '<=', z3.Z3_OP_LT: '<', z3.Z3_OP_GE: '>=', z3.Z3_OP_GT: '>'}


def text_numbers(text):
    """Every number the clause states, as Fractions (percentages also as decimals, $ amounts as dollars)."""
    out = set()
    for m in re.finditer(r'\$?\s?(\d{1,3}(?:,\d{3})+|\d*\.?\d+)\s*(%|percent)?', text):
        raw = m.group(1).replace(',', '')
        try:
            v = Fraction(raw)
        except ValueError:
            continue
        out.add(v)
        if m.group(2):
            out.add(v / 100)
    return out


def _const(e):
    if z3.is_rational_value(e) or z3.is_int_value(e):
        return Fraction(e.as_fraction()) if z3.is_rational_value(e) else Fraction(e.as_long())
    if z3.is_app_of(e, z3.Z3_OP_UMINUS) and e.num_args() == 1:
        c = _const(e.arg(0))
        return -c if c is not None else None
    if z3.is_app_of(e, z3.Z3_OP_TO_REAL) and e.num_args() == 1:
        return _const(e.arg(0))
    return None


def _scaled(e):
    """Constant c of a term c * y (or y * c), else None."""
    if z3.is_app_of(e, z3.Z3_OP_MUL) and e.num_args() == 2:
        for a, b in ((e.arg(0), e.arg(1)), (e.arg(1), e.arg(0))):
            c = _const(a)
            if c is not None and _const(b) is None:
                return c
    return None


def atoms(h):
    """[(atom, effective comply op or None if polarity is ambiguous, constant)] for numeric comparison atoms."""
    found, seen = [], set()

    def visit(e, pol):
        if not z3.is_app(e):
            return
        k = e.decl().kind()
        if k in KIND and e.num_args() == 2:
            op, lhs, rhs = KIND[k], e.arg(0), e.arg(1)
            c, other = _const(rhs), lhs
            if c is None:
                c, other, op = _const(lhs), rhs, FLIP[op]
            if c is None:  # x <op> c*y  ->  x/y <op> c  (ratio multiplied out)
                c = _scaled(rhs)
                if c is None:
                    c = _scaled(lhs)
                    op = FLIP[op] if c is not None else op
            if c is not None and (_const(other) is None):
                eff = None if pol is None else (op if pol else NEGATE[op])
                key = (e.get_id(), eff)
                if key not in seen:
                    seen.add(key)
                    found.append((e, eff, c))
            return
        if k == z3.Z3_OP_NOT:
            visit(e.arg(0), None if pol is None else not pol)
        elif k in (z3.Z3_OP_AND, z3.Z3_OP_OR):
            for a in e.children():
                visit(a, pol)
        elif k == z3.Z3_OP_IMPLIES:
            visit(e.arg(0), None if pol is None else not pol)
            visit(e.arg(1), pol)
        elif k == z3.Z3_OP_ITE and z3.is_bool(e):
            visit(e.arg(0), None)
            visit(e.arg(1), pol)
            visit(e.arg(2), pol)
        else:  # =, xor, distinct over Booleans, or anything else: polarity unknown below here
            for a in e.children():
                if z3.is_bool(a):
                    visit(a, None)

    visit(h, True)
    return found


def _check(formula):
    s = z3.Solver()
    s.set('timeout', TIMEOUT_MS)
    s.add(formula)
    return s.check()


def expand(smt2):
    params = signature(smt2)
    if not params:
        return None
    body = re.sub(r'\(\s*(set-logic|check-sat|get-model|exit|reset|push|pop)[^)]*\)', '', smt2)
    decls = ''.join(f'(declare-const {n} {s})\n' for n, s in params)
    call = f"(covenant_holds {' '.join(n for n, _ in params)})"
    return z3.parse_smt2_string(f'{body}\n{decls}(assert {call})\n')[0]


def anchor_reward(text, smt2, valid=True):
    out = dict(reward=0.0, anchor_score=None, anchors=0, recall=None, precision=None, vacuous=None,
               live_atoms=0, unknown_liveness=0, matched=[], missing=[], invented=[], error='')
    anchors = sorted({(a['op'], Fraction(str(a['threshold']))) for a in extract(text)}, key=lambda x: (x[1], x[0]))
    out['anchors'] = len(anchors)
    if not valid:
        return out
    try:
        h = expand(smt2)
    except z3.Z3Exception as exc:
        out['error'] = f'parse: {str(exc)[:120]}'
        return out
    if h is None:
        out['error'] = 'no covenant_holds'
        return out
    sat_h, sat_not = _check(h), _check(z3.Not(h))
    out['vacuous'] = sat_h == z3.unsat or sat_not == z3.unsat
    if out['vacuous']:
        return out
    live = []
    for atom, eff, c in atoms(h):
        verdicts = []
        for value in (z3.BoolVal(True), z3.BoolVal(False)):
            verdicts.append(_check(h != z3.substitute(h, (atom, value))))
        if z3.sat in verdicts:
            live.append((atom, eff, c))
        elif z3.unknown in verdicts:
            out['unknown_liveness'] += 1
            live.append((atom, eff, c))  # benefit of the doubt; counted separately
    out['live_atoms'] = len(live)
    nums = text_numbers(text)
    hits = set()
    for op, t in anchors:
        ok = any(eff == op and (c == t or c == 100 * t) for _, eff, c in live)
        (out['matched'] if ok else out['missing']).append(f'{op} {float(t):g}')
        hits.add(ok)
    invented = [f'{float(c):g}' for _, _, c in live if c not in nums and (c / 100) not in nums]
    out['invented'] = invented
    if not anchors:
        return out
    recall = len(out['matched']) / len(anchors)
    precision = (len(live) - len(invented)) / len(live) if live else 0.0
    out['recall'], out['precision'] = recall, precision
    f1 = 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
    out['anchor_score'] = out['reward'] = f1
    return out
