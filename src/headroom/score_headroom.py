"""Score the headroom experiment (PLAN.md): per system and question, the primary paired test, Holm table.

    python src/headroom/score_headroom.py     # -> results.md, summary.json
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import headroom as hr  # noqa: E402

PRIMARY = ('formula_rl789_s42', 'direct_qwen7b')
B, SEED = 10_000, 0


def score_case(c, ans):
    ans = ans or {}
    marks = {'status': int(ans.get('status') == c['answers']['status'])}
    for q in ('cushion', 'capacity'):
        m = hr.correct(q, ans.get(q), c['answers'][q], c['values'], c['debt_item'])
        if m is not None:
            marks[q] = m
    return marks


def main():
    cases = {json.loads(l)['case_id']: json.loads(l) for l in open(P.HEADROOM_DATA / 'cases.jsonl')}
    systems = {}
    for p in sorted((P.HEADROOM_RESULTS / 'answers').glob('*.jsonl')):
        if p.name.endswith('.raw.jsonl'):
            continue
        got = {json.loads(l)['case_id']: json.loads(l).get('answer') for l in open(p)}
        if len(got) < len(cases):
            print(f'{p.stem}: incomplete ({len(got)}/{len(cases)}), skipped')
            continue
        systems[p.stem] = {k: score_case(c, got.get(k)) for k, c in cases.items()}
    rng = random.Random(SEED)

    def per_clause(marks):
        by = defaultdict(list)
        for k, m in marks.items():
            by[cases[k]['clause_id']].append(sum(m.values()) / len(m))
        return {c: sum(v) / len(v) for c, v in by.items()}

    def boot(vals):
        n = len(vals)
        s = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(4000))
        return s[100], s[3899]

    def flip(d):
        obs = sum(d) / len(d)
        hits = sum(abs(sum(x if rng.random() < .5 else -x for x in d) / len(d)) >= abs(obs) - 1e-12 for _ in range(B))
        return obs, (hits + 1) / (B + 1)

    summary, rows = {}, []
    for s, marks in systems.items():
        pc = per_clause(marks)
        q = {k: [m[k] for m in marks.values() if k in m] for k in ('status', 'cushion', 'capacity')}
        kinds = defaultdict(list)
        for k, m in marks.items():
            kinds[cases[k]['kind']].append(sum(m.values()) / len(m))
        summary[s] = dict(clause_mean=sum(pc.values()) / len(pc), ci=boot(list(pc.values())),
                          **{f'{k}_acc': sum(v) / len(v) for k, v in q.items() if v}, n_cases=len(marks),
                          by_kind={k: sum(v) / len(v) for k, v in kinds.items()})
    tests = {}
    formula = [s for s in systems if s.startswith('formula_')]
    direct = [s for s in systems if s.startswith('direct_')]
    for a in formula:
        for b in direct:
            pa, pb = per_clause(systems[a]), per_clause(systems[b])
            d = [pa[c] - pb[c] for c in sorted(pa)]
            obs, p = flip(d)
            tests[f'{a} - {b}'] = dict(diff=obs, ci=boot(d), p=p)
    order = sorted(tests, key=lambda k: tests[k]['p'])
    run = 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * tests[k]['p']))
        tests[k]['p_holm'] = run
    primary = tests.get(f'{PRIMARY[0]} - {PRIMARY[1]}')
    (P.HEADROOM_RESULTS / 'summary.json').write_text(json.dumps(dict(systems=summary, tests=tests, primary=primary), indent=1))
    pct = lambda x: f'{100 * x:.1f}'
    md = ['# Headroom experiment: results', '', f'{len(cases)} cases, {len({c["clause_id"] for c in cases.values()})} clauses '
          '(borrower-disjoint; constructed certificates). PLAN.md.', '',
          '| System | Score per clause [95% CI] | Status | Earnings cushion | Debt capacity |', '|---|---|---|---|---|']
    for s in sorted(summary, key=lambda s: -summary[s]['clause_mean']):
        v = summary[s]
        md.append(f"| {s} | {pct(v['clause_mean'])} [{pct(v['ci'][0])}, {pct(v['ci'][1])}] | {pct(v.get('status_acc', 0))} | "
                  f"{pct(v.get('cushion_acc', 0))} | {pct(v.get('capacity_acc', 0))} |")
    md += ['', '| Formula arm minus direct arm | Diff | 95% CI | p | p (Holm) |', '|---|---|---|---|---|']
    md += [f"| {k} | {100 * tests[k]['diff']:+.1f} | [{100 * tests[k]['ci'][0]:+.1f}, {100 * tests[k]['ci'][1]:+.1f}] | "
           f"{tests[k]['p']:.4f} | {tests[k]['p_holm']:.4f} |" for k in order]
    if primary:
        md += ['', f"**Primary: {PRIMARY[0]} minus {PRIMARY[1]} = {100 * primary['diff']:+.1f} points "
               f"[{100 * primary['ci'][0]:+.1f}, {100 * primary['ci'][1]:+.1f}], p = {primary['p']:.4f}.**"]
    (P.HEADROOM_RESULTS / 'results.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
