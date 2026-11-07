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
                    term = lhs / y
                elif _scaled(lhs) is not None:
                    c, op = _scaled(lhs), FLIP[op]
                    y = lhs.arg(1) if _const(lhs.arg(0)) is not None else lhs.arg(0)
                    term = rhs / y
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


def score(text, smt2, status_facts='', extra_constraints=None):
    out = dict(score=None, scenarios=0, passed=0, anchors=0, matched=0, error='', details=[])
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
        self.tok = AutoTokenizer.from_pretrained(model, padding_side='left')
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
