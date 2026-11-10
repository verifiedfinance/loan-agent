"""Execution score for exec_rl (PLAN.md): label-free, judge-free, solver-checked counterfactual scenarios.

    score(text, smt2, status_facts) -> dict(score, scenarios, passed, ...)

1. anchors: deterministic extractor over the clause text -> {(comply op, threshold)}
2. each anchor is matched to a live comparison atom of the expanded formula with the same constant and effective
   direction; the atom's left side (a term; c*y forms become x/y) is the quantity the scenarios set
3. status_facts: (assert (= p true|false)) lines for Bool parameters only, from the fixed binder (Binder below) for the
   standard state (agreement in force, test date, no waiver, other events not reported); anything else is dropped
4. scenarios per matched anchor, other matched anchors held halfway inside their limits:
     inside 2%  -> compliance must be POSSIBLE   (facts and H satisfiable)
     outside 2% -> breach must be FORCED         (facts and H unsatisfiable, facts and not H satisfiable)
     at T       -> forced breach if strict (< >), possible compliance otherwise
   every denominator inside a set term is constrained positive (a ratio's base)
   an unmatched anchor fails its three scenarios
5. score = passed / scenarios; None when the clause has no extractable anchor
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

from anchor_reward import KIND, FLIP, NEGATE, _const, _scaled, expand  # noqa: E402
from decide import facts_from_text, signature  # noqa: E402
from reference_rules import extract  # noqa: E402

STEP, TIMEOUT_MS = Fraction(1, 50), 2000


def _check(*constraints):
    s = z3.Solver()
    s.set('timeout', TIMEOUT_MS)
    s.add(*constraints)
    return s.check()


def _denominators(e, out):
    if z3.is_app(e):
        if e.decl().kind() == z3.Z3_OP_DIV and e.num_args() == 2:
            out.append(e.arg(1))
        for a in e.children():
            _denominators(a, out)
    return out


def comparison_atoms(h):
    """[(atom, effective comply op or None, constant, term)] -- like anchor_reward.atoms, also returning the term."""
    found, seen = [], set()

    def visit(e, pol):
        if not z3.is_app(e):
            return
        k = e.decl().kind()
        if k in KIND and e.num_args() == 2:
            op, lhs, rhs = KIND[k], e.arg(0), e.arg(1)
            c, term = _const(rhs), lhs
            if c is None and _const(lhs) is not None:
                c, term, op = _const(lhs), rhs, FLIP[op]
            if c is None:
                c = _scaled(rhs)
                if c is not None:
                    y = rhs.arg(1) if _const(rhs.arg(0)) is not None else rhs.arg(0)
                    try:
                        term = lhs / y
                    except z3.Z3Exception:  # ill-sorted division in a sampled formula: not a limit atom
                        return
                elif _scaled(lhs) is not None:
                    c, op = _scaled(lhs), FLIP[op]
                    y = lhs.arg(1) if _const(lhs.arg(0)) is not None else lhs.arg(0)
                    try:
                        term = rhs / y
                    except z3.Z3Exception:
                        return
            if c is not None and _const(term) is None:
                eff = None if pol is None else (op if pol else NEGATE[op])
                key = (e.get_id(), eff)
                if key not in seen:
                    seen.add(key)
                    found.append((e, eff, c, term))
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
        else:
            for a in e.children():
                if z3.is_bool(a):
                    visit(a, None)

    visit(h, True)
    return found


def status_constraints(smt2, facts_text):
    """Parse binder facts; keep only (= BoolParam true|false)."""
    params = dict(signature(smt2) or [])
    keep = []
    for a in facts_from_text(facts_text)[0]:
        m = re.match(r'\(\s*assert\s*\(\s*=\s*([^\s()]+)\s+(true|false)\s*\)\s*\)\s*$', a)
        if m and params.get(m.group(1)) == 'Bool':
            keep.append(z3.Bool(m.group(1)) == (m.group(2) == 'true'))
    return keep


def _real(v):
    return z3.RealVal(f'{v.numerator}/{v.denominator}')


def score(text, smt2, status_facts='', extra_constraints=None, anchors=None):
    out = dict(score=None, scenarios=0, passed=0, anchors=0, matched=0, error='', details=[])
    if anchors is None:  # default: limits read from the clause text by the deterministic rules
        anchors = sorted({(a['op'], Fraction(str(a['threshold']))) for a in extract(text)}, key=lambda x: (x[1], x[0]))
    out['anchors'] = len(anchors)
    if not anchors:
        return out
    try:
        h = expand(smt2)
    except z3.Z3Exception as exc:
        out['error'] = f'parse: {str(exc)[:120]}'
        out['score'] = 0.0
        return out
    if h is None:
        out['error'] = 'no covenant_holds'
        out['score'] = 0.0
        return out
    status = status_constraints(smt2, status_facts) + list(extra_constraints or [])  # extra: headroom RL v2 (test_date)
    atoms = comparison_atoms(h)
    matches = {}
    for op, t in anchors:
        for atom, eff, c, term in atoms:
            if eff == op and (c == t or c == 100 * t):
                matches[(op, t)] = (term, c)
                break
    out['matched'] = len(matches)

    def held(op, t):
        return t / 2 if op in ('<=', '<') else 2 * t

    total = passed = 0
    for op, t in anchors:
        if (op, t) not in matches:
            total += 3
            out['details'].append(f'{op} {float(t):g}: unmatched')
            continue
        term, c = matches[(op, t)]
        others = []
        for (op2, t2), (term2, c2) in matches.items():
            if (op2, t2) != (op, t) and not z3.eq(z3.simplify(term2), z3.simplify(term)):
                others.append(term2 == _real(held(op2, c2)))
        positive = [d > 0 for d in _denominators(term, [])]
        for o in others:
            positive += [d > 0 for d in _denominators(o.arg(0), [])]
        up = op in ('<=', '<')
        cases = [('inside', c * (1 - STEP) if up else c * (1 + STEP), 'possible'),
                 ('outside', c * (1 + STEP) if up else c * (1 - STEP), 'forced_breach'),
                 ('at', c, 'forced_breach' if op in ('<', '>') else 'possible')]
        for name, value, want in cases:
            facts = status + others + positive + [term == _real(value)]
            holds = _check(*facts, h)
            if want == 'possible':
                ok = holds == z3.sat
            else:
                ok = holds == z3.unsat and _check(*facts, z3.Not(h)) == z3.sat
            total += 1
            passed += ok
            if not ok:
                out['details'].append(f'{op} {float(c):g} {name}: expected {want}, got holds={holds}')
    out.update(scenarios=total, passed=passed, score=passed / total if total else None)
    return out


# ------------------------------------------------------------------ schedules and conditions (reward v3)
def _date_int(d):
    return int(str(d).replace('-', '')[:8])


def _window_dates(frm, to):
    """Test dates spread over a schedule window (YYYYMMDD ints): near its start, its middle and its end. Open ends stand
    for 1993-03-31 and 2099-12-31, which bracket the benchmark's agreements. Testing several dates stops a formula from
    passing by switching itself off (e.g. an invented `test_date >= 2012` guard) at dates the reward never checks."""
    import datetime as dt
    as_date = lambda v: dt.date(v // 10000, v // 100 % 100, v % 100)
    to_int = lambda d: d.year * 10000 + d.month * 100 + d.day
    a = as_date(_date_int(frm)) if frm else dt.date(1993, 3, 31)
    b = as_date(_date_int(to)) if to else dt.date(2099, 12, 31)
    if b <= a:
        return [to_int(a)]
    step = dt.timedelta(days=min(31, (b - a).days // 4 or 1))
    return sorted({to_int(a + step), to_int(a + (b - a) / 2), to_int(b - step)})


def scenarios(spec):
    """(test_date, {flag: value}) settings that make each schedule step and condition of the spec apply, with several
    test dates per schedule window. Each setting lists the limits (op, threshold) in force under it. A clause without
    schedules is tested at dates from 1993 to 2099, since its limits must hold the same way at any date."""
    import itertools
    flags = [f['name'] for f in spec.get('flags', [])]
    combos = [dict(zip(flags, v)) for v in itertools.product([False, True], repeat=len(flags))] if len(flags) <= 3 else \
             [{f: False for f in flags}] + [{**{f: False for f in flags}, g: True} for g in flags]
    dates = sorted({d for l in spec['limits'] for st in l['schedule'] for d in _window_dates(st.get('from'), st.get('to'))})
    out = []
    for d in dates:
        for fl in (combos or [{}]):
            anchors = set()
            for l in spec['limits']:
                w = l.get('when')
                if w and fl.get(w['flag']) != bool(w['value']):
                    continue
                for st in l['schedule']:
                    lo = _date_int(st['from']) if st.get('from') else 0
                    hi = _date_int(st['to']) if st.get('to') else 99999999
                    if lo <= d <= hi:
                        anchors.add((l['op'], Fraction(str(st['threshold']))))
            if anchors:
                out.append((d, fl, sorted(anchors, key=lambda x: (x[1], x[0]))))
    return out


def score_spec(text, smt2, spec):
    """Execution score over every schedule window and condition setting of the clause (reward v3): the same three tests
    per limit as score(), run with test_date inside the window and the condition flags set so that the limit applies.
    The windows and conditions come from the teacher's reading of the clause (spec); the thresholds are checked in the
    clause text when the RL pool is built. Unlike the v2 reward, which fixes test_date to 2099-12-31, every clause is
    tested at several dates (see _window_dates)."""
    total = passed = 0
    details = []
    for d, fl, anchors in scenarios(spec):
        extra = [z3.Int('test_date') == d] + [z3.Bool(f) == v for f, v in fl.items()]
        r = score(text, smt2, '', extra, anchors=anchors)
        if r['score'] is None and r['error']:
            return dict(score=0.0, scenarios=0, passed=0, error=r['error'], details=[r['error']])
        total += r['scenarios']
        passed += r['passed']
        details += [f'{d} {fl}: {x}' for x in r['details']]
    return dict(score=passed / total if total else None, scenarios=total, passed=passed, error='', details=details)


# ------------------------------------------------------------------ fixed binder (GPU; used during training)
STANDARD_STATE = (
    'Status at the observation date: the credit agreement is in force; Commitments, Loans and Letters of Credit are '
    'outstanding and Obligations remain unpaid (nothing has been paid in full, terminated, or cash collateralized); the '
    'date is the last day of a fiscal quarter and of a fiscal year, ending a period of four consecutive fiscal quarters, '
    'and it is a test date for every covenant; no consent, waiver or amendment has been given; no Default or Event of '
    'Default has occurred. Whether any other event or condition has occurred (for example a ratings event, an '
    'acquisition, a trigger period, an election by the borrower) is NOT reported.')
BINDER_SYSTEM = ('You state the values of Boolean inputs of a covenant formula for a given situation. '
                 'You never change the formula.')
BINDER_USER = """Covenant formula (SMT-LIB 2), with each parameter described in the comments:

{formula}

{state}

For each Bool parameter of covenant_holds whose value this status determines, write
(assert (= <parameter> true)) or (assert (= <parameter> false)).
Leave out Bool parameters the status does not determine, and leave out every Real or Int parameter.
Output only the assertions."""


class Binder:
    """Fixed, untrained binder model (greedy). bind([smt2, ...]) -> [facts text, ...]."""

    def __init__(self, model='Qwen/Qwen2.5-Coder-7B-Instruct', max_new_tokens=256, device='cuda'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lora_utils import load_tokenizer
        self.tok = load_tokenizer(model, padding_side='left')
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
        self.max_new_tokens, self.device, self.name = max_new_tokens, device, model

    def bind(self, formulas):
        import torch
        msgs = [[{'role': 'system', 'content': BINDER_SYSTEM},
                 {'role': 'user', 'content': BINDER_USER.format(formula=f.strip(), state=STANDARD_STATE)}] for f in formulas]
        inputs = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, padding=True,
                                              return_dict=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None,
                                      top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(g[inputs['input_ids'].shape[-1]:], skip_special_tokens=True) for g in gen]
