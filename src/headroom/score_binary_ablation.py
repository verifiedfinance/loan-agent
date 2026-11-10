"""Binary-reward ablation (scripts/06_binary_ablation.sh): RL with exec_bin vs RL v2 (exec_sig) and SFT.

Same scoring as score_headroom.py (per-clause score, bootstrap CI, paired sign-flip test), kept separate so the
paper's Holm family of formula-vs-direct tests is unchanged. Per-clause scores are averaged over seeds before the
paired tests, as in the paper.

    python src/headroom/score_binary_ablation.py   # -> results/headroom/binary_ablation.{json,md}
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import json
import random
from collections import defaultdict
from statistics import mean, stdev

from score_headroom import score_case  # noqa: E402

ARMS = {'binary': ['formula_rl789_bin_s42', 'formula_rl789_bin_s43', 'formula_rl789_bin_s44'],
        'graded (RL v2)': ['formula_rl789_v2', 'formula_rl789_v2_s43', 'formula_rl789_v2_s44'],
        'SFT': ['formula_sft789']}
COMPARE = [('binary', 'graded (RL v2)'), ('binary', 'SFT'), ('graded (RL v2)', 'SFT')]
B, SEED = 10_000, 0


def main():
    cases = {json.loads(l)['case_id']: json.loads(l) for l in open(P.HEADROOM_DATA / 'cases.jsonl')}
    rng = random.Random(SEED)

    def load(name):
        got = {json.loads(l)['case_id']: json.loads(l).get('answer') for l in open(P.HEADROOM_RESULTS / 'answers' / f'{name}.jsonl')}
        assert len(got) == len(cases), f'{name}: {len(got)}/{len(cases)} cases'
        return {k: score_case(c, got.get(k)) for k, c in cases.items()}

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

    seeds, arm_pc = {}, {}
    for arm, names in ARMS.items():
        pcs = [per_clause(load(n)) for n in names]
        seeds[arm] = [mean(pc.values()) for pc in pcs]
        arm_pc[arm] = {c: mean(pc[c] for pc in pcs) for c in pcs[0]}
    tests = {}
    for a, b in COMPARE:
        d = [arm_pc[a][c] - arm_pc[b][c] for c in sorted(arm_pc[a])]
        obs, p = flip(d)
        tests[f'{a} - {b}'] = dict(diff=obs, ci=boot(d), p=p,
                                   clauses_better=sum(x > 1e-12 for x in d), clauses_worse=sum(x < -1e-12 for x in d))
    out = dict(seeds=seeds, mean={a: mean(v) for a, v in seeds.items()},
               sd={a: (stdev(v) if len(v) > 1 else None) for a, v in seeds.items()}, tests=tests)
    (P.HEADROOM_RESULTS / 'binary_ablation.json').write_text(json.dumps(out, indent=1))
    pct = lambda x: f'{100 * x:.1f}'
    md = ['# Binary-reward ablation', '', f'{len(cases)} cases, {len(arm_pc["SFT"])} clauses; score per clause.', '',
          '| Arm | Seeds | Mean ± s.d. |', '|---|---|---|']
    for a in ARMS:
        sd = out['sd'][a]
        md.append(f"| {a} | {', '.join(pct(x) for x in seeds[a])} | {pct(out['mean'][a])}" + (f" ± {pct(sd)}" if sd is not None else '') + ' |')
    md += ['', '| Comparison | Diff | 95% CI | p | clauses better / worse |', '|---|---|---|---|---|']
    md += [f"| {k} | {100 * t['diff']:+.1f} | [{100 * t['ci'][0]:+.1f}, {100 * t['ci'][1]:+.1f}] | {t['p']:.4f} | "
           f"{t['clauses_better']} / {t['clauses_worse']} |" for k, t in tests.items()]
    (P.HEADROOM_RESULTS / 'binary_ablation.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
