"""Formal verification of each generated formula against the reference reading (PLAN.md, added 2026-11-06 evening).

For every clause and system, Z3 checks whether the model's covenant_holds and the reference's are equivalent for ALL
inputs in the domain: every line item > 0, every reference ratio denominator > 0, test_date a calendar date between 1990-01-01 and
2030-12-31, flags free. unsat of (model != reference) = verified equivalent; sat = a counterexample exists (printed);
unknown = undecided (timeout).

    python src/headroom/verify_equivalence.py sft789 rl789_s42 qwen7b_prompted
Output: results/headroom/verification/<name>.jsonl and verification/summary.json
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import json
import sys
from pathlib import Path

import z3

import headroom as hr  # noqa: E402
from run_formula import extract_smt  # noqa: E402

TIMEOUT_MS = 20000


def holds_expr(formula, spec):
    ev = hr.FormulaEvaluator(formula, spec)  # strict parameter check, same as the formula arm
    return ev.holds[0] if len(ev.holds) == 1 else z3.And(*ev.holds), ev.const


def calendar_date(td):
    """Constraints that make the YYYYMMDD integer td a real calendar date. Without them Z3 may return counterexamples
    such as 19971232 (December 32), which fall between the windows of a date schedule and never occur in practice."""
    y, m, d = td / 10000, (td / 100) % 100, td % 100  # integer division and remainder (linear integer arithmetic)
    leap = z3.And(y % 4 == 0, z3.Or(y % 100 != 0, y % 400 == 0))
    last = z3.If(m == 2, z3.If(leap, 29, 28), z3.If(z3.Or(m == 4, m == 6, m == 9, m == 11), 30, 31))
    return [m >= 1, m <= 12, d >= 1, d <= last]


def domain(spec, const):
    d = [const[i['name']] > 0 for i in spec['items']]
    d += [const['test_date'] >= 19900101, const['test_date'] <= 20301231] + calendar_date(const['test_date'])
    for lim in spec['limits']:
        if lim['denominator']:
            d.append(z3.Sum([const[i] * s for i, s in lim['denominator']]) > 0)
    return d


def check(formula, spec):
    """'equivalent' | 'different' (+ counterexample) | 'unknown' | 'invalid'"""
    try:
        model_h, const = holds_expr(formula, spec)
        ref_h, _ = holds_expr(hr.spec_to_smt(spec), spec)
    except Exception as e:  # noqa: BLE001
        return dict(result='invalid', detail=f'{type(e).__name__}: {str(e)[:120]}')
    s = z3.Solver()
    s.set('timeout', TIMEOUT_MS)
    s.add(*domain(spec, const))
    s.add(model_h != ref_h)
    r = s.check()
    if r == z3.unsat:
        return dict(result='equivalent')
    if r == z3.sat:
        m = s.model()
        return dict(result='different', counterexample={str(d): str(m[d]) for d in m.decls()})
    return dict(result='unknown')


def main():
    specs = {json.loads(l)['id']: json.loads(l) for l in open(P.HEADROOM_DATA / 'specs.jsonl')}
    used = {json.loads(l)['clause_id'] for l in open(P.HEADROOM_DATA / 'cases.jsonl')}  # the 143 clauses with cases
    out_dir = P.HEADROOM_RESULTS / 'verification'
    out_dir.mkdir(exist_ok=True)
    summary_path = out_dir / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in sys.argv[1:]:
        formulas = {json.loads(l)['case_id']: json.loads(l) for l in open(P.HEADROOM_RESULTS / 'formulas' / f'{name}.jsonl')}
        rows = []
        for k in sorted(used):
            f = formulas.get(k)
            r = check(extract_smt(f['output']), specs[k]['spec']) if f and not f.get('error') else dict(result='invalid', detail='no formula')
            rows.append(dict(clause_id=k, **r))
        (out_dir / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        counts = {x: sum(r['result'] == x for r in rows) for x in ('equivalent', 'different', 'unknown', 'invalid')}
        summary[name] = dict(clauses=len(rows), **counts, verified_rate=counts['equivalent'] / len(rows))
        print(name, summary[name], flush=True)
    summary_path.write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
