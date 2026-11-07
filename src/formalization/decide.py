"""Z3 decision for the downstream test: given a covenant formula and SMT facts, comply / breach / undetermined.

    formula   an SMT-LIB 2 script that defines (covenant_holds p1 ... pn) and declares nothing else
              (the system's own formalization, or the reference rendered by reference_smt)
    facts     model-written text; only top-level (assert ...) forms are kept, anything else is
              reported (`other_commands`) and ignored, so facts cannot reset or redefine the formula

Verdict with F = facts and H = (covenant_holds p1 ... pn), every parameter declared as a constant:
    comply        F ∧ ¬H unsat and F ∧ H sat
    breach        F ∧ H unsat and F ∧ ¬H sat
    undetermined  both sat (the facts leave the answer open)
    inconsistent  both unsat (the facts contradict each other or the formula's own assertions)
    error         parse error, unknown (timeout), or no covenant_holds
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import re

import z3

TIMEOUT_MS = 5000


def sexprs(text):
    """Top-level balanced s-expressions in `text` (strings and comments respected)."""
    out, depth, start, i, n = [], 0, None, 0, len(text)
    while i < n:
        ch = text[i]
        if ch == ';':
            j = text.find('\n', i)
            i = n if j < 0 else j
            continue
        if ch == '"':
            j = text.find('"', i + 1)
            i = n if j < 0 else j + 1
            continue
        if ch == '(':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ')' and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
        i += 1
    return out


def facts_from_text(text):
    forms = sexprs(text)
    asserts = [f for f in forms if re.match(r'\(\s*assert\b', f)]
    return asserts, len(forms) - len(asserts)


SORT = re.compile(r'\(define-fun\s+covenant_holds\s+\((.*?)\)\s+Bool', re.S)


def signature(formula):
    """[(name, sort)] of covenant_holds, or None."""
    m = SORT.search(formula)
    if not m:
        return None
    return re.findall(r'\(\s*([^\s()]+)\s+([^\s()]+)\s*\)', m.group(1))


def decide(formula, facts_text):
    params = signature(formula)
    if params is None:
        return dict(verdict='error', detail='no covenant_holds definition')
    asserts, other = facts_from_text(facts_text)
    body = re.sub(r'\(\s*set-logic[^)]*\)', '', formula)
    body = re.sub(r'\(\s*(check-sat|get-model|exit|reset|push|pop)[^)]*\)', '', body)
    decls = ''.join(f'(declare-const {n} {s})\n' for n, s in params)
    call = f"(covenant_holds {' '.join(n for n, _ in params)})"
    results = {}
    for name, goal in (('holds', call), ('fails', f'(not {call})')):
        script = f'{body}\n{decls}' + '\n'.join(asserts) + f'\n(assert {goal})\n'
        try:
            solver = z3.Solver()
            solver.set('timeout', TIMEOUT_MS)
            solver.add(z3.parse_smt2_string(script))
            results[name] = solver.check()
        except z3.Z3Exception as exc:
            return dict(verdict='error', detail=f'parse: {str(exc)[:200]}', asserts=len(asserts), other_commands=other)
    h, f = results['holds'], results['fails']
    if z3.unknown in (h, f):
        verdict = 'error'
    elif h == z3.sat and f == z3.unsat:
        verdict = 'comply'
    elif h == z3.unsat and f == z3.sat:
        verdict = 'breach'
    elif h == z3.sat and f == z3.sat:
        verdict = 'undetermined'
    else:
        verdict = 'inconsistent'
    return dict(verdict=verdict, detail=f'holds={h} fails={f}', asserts=len(asserts), other_commands=other)


# ---------------------------------------------------------------- reference rendering (for validation)
SMT_OP = {'<=': '<=', '<': '<', '>=': '>=', '>': '>'}


def num(x):
    return f'{x:.12f}'.rstrip('0').rstrip('.') if '.' in f'{x:.12f}' else str(x)


def reference_smt(spec):
    metrics = sorted({l['metric'] for l in spec})
    flags = sorted({l['when'][0] for l in spec if l.get('when')})
    params = ' '.join([f'({m} Real)' for m in metrics] + [f'({f} Bool)' for f in flags])
    terms = []
    for l in spec:
        cmp_ = f"({SMT_OP[l['op']]} {l['metric']} {num(float(l['threshold']))})"
        if l.get('when'):
            f, v = l['when']
            terms.append(f"(=> {f if v else f'(not {f})'} {cmp_})")
        else:
            terms.append(cmp_)
    return f"(define-fun covenant_holds ({params}) Bool (and {' '.join(terms)}))"


def reference_facts(case):
    lines = [f'(assert (= {m} {num(float(v))}))' for m, v in case['metrics'].items()]
    lines += [f"(assert (= {f} {'true' if v else 'false'}))" for f, v in case['flags'].items() if v is not None]
    return '\n'.join(lines)
