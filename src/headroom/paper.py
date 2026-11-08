#!/usr/bin/env python3
"""Paper tables (LaTeX, booktabs) and figures (figures4papers house style) for the headroom experiment.

Everything is computed from the raw files in this directory (data/headroom/, results/headroom/,
models/rl_v2_s*/). Used by notebooks/headroom_paper.ipynb. Outputs: results/paper/{figures,tables}/.

Figure style follows github.com/ChenLiu-1996/figures4papers (scientific-figure-making/references): Arial/Helvetica-like
sans fonts, top/right spines off, frameless legends, semantic palette (blues = our method, reds = baselines, neutral =
reference), black bar edges + hatching for print, values written on the bars, dedicated legend panel, tight_layout(pad=2),
PNG + PDF at 300 dpi.
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DATA_DIR, RES = P.HEADROOM_DATA, P.HEADROOM_RESULTS
import headroom as hr  # noqa: E402
import score_headroom as sc  # noqa: E402

OUT = P.PAPER
FIG, TAB = OUT / 'figures', OUT / 'tables'
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------------- style (figures4papers)
PALETTE = {'blue_main': '#0F4D92', 'blue_secondary': '#3775BA', 'green_1': '#DDF3DE', 'green_2': '#AADCA9',
           'green_3': '#8BCF8B', 'red_1': '#F6CFCB', 'red_2': '#E9A6A1', 'red_strong': '#B64342', 'neutral': '#CFCECE',
           'gray_mid': '#767676', 'gray_dark': '#4D4D4D', 'ink': '#272727', 'highlight': '#FFD700', 'teal': '#42949E',
           'violet': '#9A4D8E'}
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans'],
    'font.size': 16, 'axes.linewidth': 2, 'axes.spines.right': False, 'axes.spines.top': False,
    'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'xtick.major.width': 2, 'ytick.major.width': 2, 'xtick.major.size': 5, 'ytick.major.size': 5,
    'axes.labelcolor': PALETTE['ink'], 'xtick.color': PALETTE['ink'], 'ytick.color': PALETTE['ink'], 'text.color': PALETTE['ink'],
})


def finalize(fig, name, pad=2):
    fig.tight_layout(pad=pad)
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
    return FIG / f'{name}.png'


# --------------------------------------------------------------------------------------------- systems
# key -> (label, short label, group, answers file)
DIRECT = [('qwen7b', 'Qwen2.5-Coder-7B', '7B', 'answers_long/direct_qwen7b.jsonl'),
          ('qwen32b', 'Qwen2.5-Coder-32B', '32B', 'answers_long/direct_qwen32b.jsonl'),
          ('qwen27b', 'Qwen3.8-27B', '27B', 'answers_long/direct_qwen27b.jsonl')]
FORMULA = [('prompted', 'Prompted, no training', 'answers/formula_qwen7b_prompted.jsonl'),
           ('sft', 'SFT (789 pairs)', 'answers/formula_sft789.jsonl'),
           ('rl_v1', 'SFT + RL, v1 (no interface term)', 'answers/formula_rl789_s42.jsonl')]
RL_V2 = {42: 'answers/formula_rl789_v2.jsonl', 43: 'answers/formula_rl789_v2_s43.jsonl', 44: 'answers/formula_rl789_v2_s44.jsonl'}
RL_V2_FORMULAS = {42: 'rl789_v2', 43: 'rl789_v2_s43', 44: 'rl789_v2_s44'}
FORMULA_FILES = {'prompted': 'qwen7b_prompted', 'sft': 'sft789', 'rl_v1': 'rl789_s42'}
QUESTIONS = [('status', 'Status'), ('cushion', 'Earnings cushion'), ('capacity', 'Debt capacity')]


# --------------------------------------------------------------------------------------------- data
_cache = {}


def cases():
    if 'cases' not in _cache:
        _cache['cases'] = {json.loads(l)['case_id']: json.loads(l) for l in open(DATA_DIR / 'cases.jsonl')}
    return _cache['cases']


def specs():
    if 'specs' not in _cache:
        _cache['specs'] = {json.loads(l)['id']: json.loads(l) for l in open(DATA_DIR / 'specs.jsonl')}
    return _cache['specs']


def answers(rel):
    return {json.loads(l)['case_id']: json.loads(l).get('answer') for l in open(RES / rel)}


def marks(rel):
    """case_id -> {question: 0/1} for one answers file."""
    if rel not in _cache:
        got = answers(rel)
        _cache[rel] = {k: sc.score_case(c, got.get(k)) for k, c in cases().items()}
    return _cache[rel]


def per_clause(m, keep=None):
    by = defaultdict(list)
    for k, mk in m.items():
        c = cases()[k]
        if keep is None or keep(c):
            by[c['clause_id']].append(sum(mk.values()) / len(mk))
    return {c: sum(v) / len(v) for c, v in by.items()}


def q_acc(m, q, keep=None):
    v = [mk[q] for k, mk in m.items() if q in mk and (keep is None or keep(cases()[k]))]
    return sum(v) / len(v)


def mean(d):
    return sum(d.values()) / len(d)


def boot_ci(vals, reps=10_000, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    s = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return s[int(0.025 * reps)], s[int(0.975 * reps) - 1]


def paired(a, b, reps=10_000, seed=0):
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    obs = sum(d) / len(d)
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < .5 else -x for x in d) / len(d)) >= abs(obs) - 1e-12 for _ in range(reps))
    lo, hi = boot_ci(d, reps, seed)
    return dict(diff=obs, lo=lo, hi=hi, p=(hits + 1) / (reps + 1), n=len(d))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        out[i] = run
    return out


def schedule_clauses():
    return {k for k, s in specs().items() if s['spec']['flags'] or any(len(l['schedule']) > 1 for l in s['spec']['limits'])}


def system_stats():
    """All systems: per-clause scores, overall, CI, per-question accuracy. RL v2 = mean over seeds (and per seed)."""
    if 'stats' in _cache:
        return _cache['stats']
    out = {}

    def add(key, label, group, rel, ours):
        m = marks(rel)
        pc = per_clause(m)
        out[key] = dict(label=label, group=group, ours=ours, pc=pc, score=mean(pc), ci=boot_ci(list(pc.values())),
                        **{q: q_acc(m, q) for q, _ in QUESTIONS})
    for key, label, group, rel in DIRECT:
        add(key, label, group, rel, False)
    for key, label, rel in FORMULA:
        add(key, label, '7B formalizer', rel, True)
    for s, rel in RL_V2.items():
        add(f'rl_v2_s{s}', f'SFT + RL v2 (seed {s})', '7B formalizer', rel, True)
    seeds = [out[f'rl_v2_s{s}'] for s in RL_V2]
    pc = {c: sum(s['pc'][c] for s in seeds) / len(seeds) for c in seeds[0]['pc']}
    out['rl_v2'] = dict(label='SFT + RL v2 (ours, 3 seeds)', group='7B formalizer', ours=True, pc=pc, score=mean(pc),
                        ci=boot_ci(list(pc.values())), seed_scores=[s['score'] for s in seeds],
                        **{q: float(np.mean([s[q] for s in seeds])) for q, _ in QUESTIONS},
                        **{f'{q}_sd': float(np.std([s[q] for s in seeds], ddof=1)) for q, _ in QUESTIONS},
                        score_sd=float(np.std([s['score'] for s in seeds], ddof=1)))
    _cache['stats'] = out
    return out


def verification():
    v = json.loads((RES / 'verification/summary.json').read_text())
    rows = {}
    for key, name in list(FORMULA_FILES.items()) + [(f'rl_v2_s{s}', n) for s, n in RL_V2_FORMULAS.items()]:
        if name not in v:
            continue
        per = {json.loads(l)['clause_id']: json.loads(l)['result'] for l in open(RES / f'verification/{name}.jsonl')}
        rows[key] = dict(**v[name], per=per)
    return rows


def cross_check(a_name, b_name, a_rel, cache_name):
    """Label-free self-check: two formulas for the same clause are checked for equivalence by Z3 (no reference).
    Returns coverage and the accuracy of a's answers on agreed vs other clauses. Cached in paper/cross_<name>.json."""
    path = OUT / f'cross_{cache_name}.json'
    if path.exists():
        res = json.loads(path.read_text())
    else:
        import z3
        import verify_equivalence as ve
        from run_formula import extract_smt
        fa = {json.loads(l)['case_id']: json.loads(l) for l in open(RES / f'formulas/{a_name}.jsonl')}
        fb = {json.loads(l)['case_id']: json.loads(l) for l in open(RES / f'formulas/{b_name}.jsonl')}
        res = {}
        for k in sorted({c['clause_id'] for c in cases().values()}):
            sp = specs()[k]['spec']
            try:
                ha, const = ve.holds_expr(extract_smt(fa[k]['output']), sp)
                hb, _ = ve.holds_expr(extract_smt(fb[k]['output']), sp)
                s = z3.Solver()
                s.set('timeout', 20000)
                s.add(*ve.domain(sp, const))
                s.add(ha != hb)
                r = s.check()
                res[k] = 'agree' if r == z3.unsat else ('disagree' if r == z3.sat else 'unknown')
            except Exception:  # noqa: BLE001
                res[k] = 'not comparable'
        path.write_text(json.dumps(res, indent=0))
    m = marks(a_rel)
    pc = per_clause(m)
    agreed = [k for k, r in res.items() if r == 'agree']
    other = [k for k, r in res.items() if r != 'agree']
    return dict(n=len(res), agreed=len(agreed), coverage=len(agreed) / len(res),
                acc_agreed=sum(pc[k] for k in agreed) / len(agreed) if agreed else float('nan'),
                acc_other=sum(pc[k] for k in other) / len(other) if other else float('nan'))


# --------------------------------------------------------------------------------------------- LaTeX helpers
def tex(s):
    return (str(s).replace('\\', r'\textbackslash{}').replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')
            .replace('#', r'\#').replace('~', r'\textasciitilde{}'))


def pct(x, d=1):
    return f'{100 * x:.{d}f}'


def write(name, latex, markdown):
    """Write a table as LaTeX (booktabs; needs adjustbox, which shrinks it only if wider than the line) and Markdown."""
    latex = latex.replace('\\begin{tabular}', '\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}', 1)
    latex = latex.replace('\\end{tabular}', '\\end{tabular}\n\\end{adjustbox}', 1)
    (TAB / f'{name}.tex').write_text(latex + '\n')
    (TAB / f'{name}.md').write_text(markdown + '\n')
    return latex, markdown


def fmt_p(p):
    return r'$<$0.001' if p < 0.001 else f'{p:.3f}'


def emph(vals, higher=True):
    """Column-wise: index of best (bold) and second best (underline)."""
    order = sorted(range(len(vals)), key=lambda i: -vals[i] if higher else vals[i])
    return order[0], order[1] if len(order) > 1 else None


# --------------------------------------------------------------------------------------------- tables
def table_benchmark():
    sp = json.loads((DATA_DIR / 'specs_summary.json').read_text())
    val = json.loads((DATA_DIR / 'validation.json').read_text())
    cs = cases()
    clauses = {c['clause_id'] for c in cs.values()}
    borrowers = {specs()[k]['borrower'] for k in clauses}
    contracts = {specs()[k]['contract_id'] for k in clauses}
    years = [specs()[k]['agreement_year'] for k in clauses if specs()[k].get('agreement_year')]
    sched = len(clauses & schedule_clauses())
    multi = sum(len(specs()[k]['spec']['limits']) > 1 for k in clauses)
    st = Counter(c['answers']['status'] for c in cs.values())
    asked = {q: sum(c['answers'][q] is not None for c in cs.values()) for q in ('cushion', 'capacity')}
    reasons = sp['exclusion_reasons']
    disagree = reasons.get('readers disagree', 0)
    both = sp['kept'] + disagree
    rows = [
        ('Source', 'Borrower-disjoint test clauses (no borrower appears in training)', f"{sp['clauses']}"),
        ('', 'Excluded: no numeric covenant, dependent or redacted threshold (reader A / B)',
         f"{reasons.get('reader A', 0) + reasons.get('reader B', 0)}"),
        ('', 'Excluded: ratio named without its components (status-only)', f"{reasons.get('no earnings or debt item', 0)}"),
        ('', 'Excluded: the two readers disagree', f'{disagree} (of {both} read by both)'),
        ('', 'Kept reference readings (both readers agree exactly)', f"{sp['kept']}"),
        ('Benchmark', 'Clauses with cases', f'{len(clauses)}'),
        ('', 'Agreements / borrowers', f'{len(contracts)} / {len(borrowers)}'),
        ('', 'Agreement years', f'{min(years)}--{max(years)}'),
        ('', 'Clauses with several limits / with date schedules or conditions', f'{multi} / {sched}'),
        ('', 'Cases (constructed certificates)', f'{len(cs):,}'),
        ('', 'Status: comply / breach / undetermined', f"{st['comply']} / {st['breach']} / {st['undetermined']}"),
        ('', 'Cases asking the earnings cushion / the debt capacity', f"{asked['cushion']} / {asked['capacity']}"),
        ('Validation', 'Reference answers reproduced by the formula-arm Z3 code', f"{val['validation_correct']:,} / {val['cases']:,}"),
    ]
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
           r'\caption{The headroom benchmark. Clauses come from agreements whose borrowers never appear in any training '
           r'data; each kept clause has a reference reading on which two independent readers (Claude and Qwen3.8-27B) '
           r'agree exactly. Certificates are constructed so that the binding limit sits at a chosen cushion.}',
           r'\label{tab:benchmark}', r'\begin{tabular}{@{}llr@{}}', r'\toprule', r' & \textbf{Item} & \textbf{Count} \\', r'\midrule']
    md = ['| | Item | Count |', '|---|---|---|']
    last = None
    for grp, item, val_ in rows:
        if grp and last is not None:
            lat.append(r'\midrule')
        last = grp or last
        lat.append(f'{tex(grp)} & {tex(item)} & {tex(val_)} \\\\')
        md.append(f'| {grp} | {item} | {val_} |')
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return write('tab1_benchmark', '\n'.join(lat), '\n'.join(md))


def pm(st, c):
    """'95.0 ± 1.2' when the system has seeds, else '85.8' (Markdown)."""
    sd = st.get('score_sd' if c == 'score' else f'{c}_sd')
    return pct(st[c]) + (f' ± {pct(sd)}' if sd is not None else '')


def table_main():
    S = system_stats()
    ver = verification()
    order = [('Reads the clause directly', [k for k, *_ in DIRECT]),
             ('Formalizes, then Z3 decides (Qwen2.5-Coder-7B + LoRA)', ['prompted', 'sft', 'rl_v1', 'rl_v2'])]
    size = {'qwen7b': '7B', 'qwen32b': '32B', 'qwen27b': '27B', 'prompted': '7B', 'sft': '7B', 'rl_v1': '7B', 'rl_v2': '7B'}
    keys = [k for _, ks in order for k in ks]
    cols = ['score', 'status', 'cushion', 'capacity']
    best = {c: emph([S[k][c] for k in keys]) for c in cols}

    def verified(k):
        if k in ('rl_v2',):
            vals = [ver[f'rl_v2_s{s}']['verified_rate'] for s in RL_V2 if f'rl_v2_s{s}' in ver]
            return f'{pct(np.mean(vals))}' + r'{\scriptsize$\pm$' + pct(np.std(vals, ddof=1)) + '}'
        return pct(ver[k]['verified_rate']) if k in ver else '--'

    def cell(k, c):
        v = S[k][c]
        s = pct(v)
        if k == 'rl_v2':
            sd = S[k]['score_sd'] if c == 'score' else S[k][f'{c}_sd']
            s += r'{\scriptsize$\pm$' + pct(sd) + '}'
        if keys[best[c][0]] == k:
            s = r'\textbf{' + s + '}'
        return s
    lat = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{6pt}',
           r'\caption{Headroom questions on the borrower-disjoint benchmark (143 clauses, 1,064 cases). Score: mean '
           r'per-clause share of questions answered correctly, with a 95\% clause-bootstrap interval. Status: comply / '
           r'breach / undetermined; earnings cushion: within 0.5 percentage points; debt capacity: within 0.5\% of the '
           r'current debt. Verified: formulas proven equivalent to the reference reading by Z3 for all inputs. '
           r'All models are open-weights and run in-house. \textbf{Bold}: best. RL v2: mean $\pm$ s.d. over three seeds; '
           r'the other systems are single greedy runs. Direct readers were given 4,096 answer tokens.}',
           r'\label{tab:main}', r'\begin{tabular}{@{}lcccccc@{}}', r'\toprule',
           r'\textbf{System} & \textbf{Size} & \textbf{Score} & \textbf{Status} & '
           r'\textbf{Cushion} & \textbf{Capacity} & \textbf{Verified} \\', r'\midrule']
    md = ['| System | Size | Score [95% CI] | Status | Cushion | Capacity | Verified |', '|---|---|---|---|---|---|---|']
    for gi, (grp, ks) in enumerate(order):
        if gi:
            lat.append(r'\midrule')
        lat.append(r'\multicolumn{7}{@{}l}{\textit{' + tex(grp) + r'}} \\')
        md.append(f'| *{grp}* | | | | | | |')
        for k in ks:
            lab = S[k]['label']
            if k in ('sft', 'rl_v2'):
                lab_t = r'\quad ' + tex(lab)
            else:
                lab_t = r'\quad ' + tex(lab)
            ci = S[k]['ci']
            score = cell(k, 'score') + r'\,{\scriptsize[' + pct(ci[0]) + ', ' + pct(ci[1]) + ']}'
            lat.append(f"{lab_t} & {size[k]} & {score} & {cell(k, 'status')} & "
                       f"{cell(k, 'cushion')} & {cell(k, 'capacity')} & {verified(k)} \\\\")
            md.append(f"| {lab} | {size[k]} | {pm(S[k], 'score')} [{pct(ci[0])}, {pct(ci[1])}] | "
                      f"{pm(S[k], 'status')} | {pm(S[k], 'cushion')} | {pm(S[k], 'capacity')} | "
                      f"{verified(k).replace(r'{\scriptsize$\pm$', ' ± ').rstrip('}')} |")
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return write('tab2_main', '\n'.join(lat), '\n'.join(md))


def table_paired():
    S = system_stats()
    comps = [('sft', 'SFT (same model before RL)'), ('qwen7b', 'Qwen2.5-Coder-7B, reading directly'),
             ('qwen32b', 'Qwen2.5-Coder-32B, reading directly'), ('qwen27b', 'Qwen3.8-27B, reading directly')]
    res = [paired(S['rl_v2']['pc'], S[k]['pc']) for k, _ in comps]
    ph = holm([r['p'] for r in res])
    seed_diffs = {k: [paired(S[f'rl_v2_s{s}']['pc'], S[k]['pc'])['diff'] for s in RL_V2] for k, _ in comps}
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
           r'\caption{SFT + RL v2 (three-seed mean) minus each comparator: mean per-clause difference in points, 95\% '
           r'clause-bootstrap interval, two-sided sign-flip test over the 143 clauses (10,000 permutations), Holm-corrected '
           r'over the four comparisons. Seeds: the difference for each of the three RL seeds. Lower block: a separate Holm '
           r'family of four comparisons.}',
           r'\label{tab:paired}', r'\begin{tabular}{@{}lrcrr@{}}', r'\toprule',
           r'\textbf{RL v2 minus} & \textbf{$\Delta$} & \textbf{95\% CI} & \textbf{$p_{\mathrm{Holm}}$} & \textbf{Seeds} \\', r'\midrule']
    md = ['| RL v2 minus | Δ (points) | 95% CI | p (Holm) | Per seed |', '|---|---|---|---|---|']
    for (k, lab), r, p in zip(comps, res, ph):
        sd = '/'.join(f'{100 * x:+.1f}' for x in seed_diffs[k])
        lat.append(f"{tex(lab)} & {100 * r['diff']:+.1f} & [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] & {fmt_p(p)} & "
                   r'{\scriptsize ' + sd + r'} \\')
        md.append(f"| {lab} | {100 * r['diff']:+.1f} | [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] | {p:.4f} | {sd} |")
    stages = [('sft', 'prompted', 'SFT minus prompted formalizer'), ('rl_v1', 'sft', 'RL v1 minus SFT'),
              ('sft', 'qwen7b', 'SFT formalizer minus 7B reading'), ('prompted', 'qwen7b', 'Prompted formalizer minus 7B reading')]
    res2 = [paired(S[a]['pc'], S[b]['pc']) for a, b, _ in stages]
    ph2 = holm([r['p'] for r in res2])
    lat += [r'\midrule', r'\multicolumn{5}{@{}l}{\textit{Training stages and equal-size comparisons (one run each)}} \\']
    md += ['| *Training stages and equal-size comparisons* | | | | |']
    for (_, _, lab), r, p in zip(stages, res2, ph2):
        lat.append(f"{tex(lab)} & {100 * r['diff']:+.1f} & [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] & {fmt_p(p)} & -- \\\\")
        md.append(f"| {lab} | {100 * r['diff']:+.1f} | [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] | {p:.4f} | -- |")
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return write('tab3_paired', '\n'.join(lat), '\n'.join(md))


def table_verification():
    S = system_stats()
    ver = verification()
    rows = [('prompted', 'Prompted, no training'), ('sft', 'SFT (789 pairs)'), ('rl_v1', 'SFT + RL v1')] + \
           [(f'rl_v2_s{s}', f'SFT + RL v2, seed {s}') for s in RL_V2]
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
           r'\caption{Formal verification of the generated formulas (143 clauses). Equivalent: Z3 proves the formula '
           r'equal to the reference reading for every input (line items $>0$, positive ratio denominators, any test '
           r'date in 1990--2030); different: Z3 finds a counterexample; unusable: invalid SMT or wrong parameters. '
           r'Accuracy: case-level score on the clauses in each group.}',
           r'\label{tab:verification}', r'\begin{tabular}{@{}lrrrcc@{}}', r'\toprule',
           r' & \multicolumn{3}{c}{\textbf{Clauses}} & \multicolumn{2}{c}{\textbf{Score on clauses}} \\',
           r'\cmidrule(lr){2-4}\cmidrule(l){5-6}',
           r'\textbf{Formalizer (7B)} & \textbf{Equiv.} & \textbf{Diff.} & \textbf{Unus.} & \textbf{equivalent} & \textbf{not equivalent} \\',
           r'\midrule']
    md = ['| Formalizer | Equivalent | Different | Unusable | Score, equivalent | Score, not equivalent |', '|---|---|---|---|---|---|']
    for k, lab in rows:
        v = ver[k]
        pc = S[k]['pc']
        eq = [c for c, r in v['per'].items() if r == 'equivalent']
        ne = [c for c, r in v['per'].items() if r != 'equivalent']
        a1 = sum(pc[c] for c in eq) / len(eq) if eq else float('nan')
        a2 = sum(pc[c] for c in ne) / len(ne) if ne else float('nan')
        lat.append(f"{tex(lab)} & {v['equivalent']} & {v['different']} & {v['invalid']} & {pct(a1)} & {pct(a2)} \\\\")
        md.append(f"| {lab} | {v['equivalent']} | {v['different']} | {v['invalid']} | {pct(a1)} | {pct(a2)} |")
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return write('tab4_verification', '\n'.join(lat), '\n'.join(md))


CROSS = [('rl789_v2', 'rl789_v2_s43', 'answers/formula_rl789_v2.jsonl', 'v2s42_v2s43', 'RL v2 seed 42 $\\times$ seed 43', 'RL v2 seed 42 × seed 43'),
         ('rl789_v2', 'sft789', 'answers/formula_rl789_v2.jsonl', 'v2s42_sft', 'RL v2 seed 42 $\\times$ SFT', 'RL v2 seed 42 × SFT'),
         ('sft789', 'qwen7b_prompted', 'answers/formula_sft789.jsonl', 'sft_prompted', 'SFT $\\times$ prompted 7B', 'SFT × prompted 7B')]


def table_selfcheck():
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
           r'\caption{Label-free self-check. Two formalizers write a formula for the same clause and Z3 checks whether '
           r'the two are equivalent for all inputs; no reference reading is used. Agreed clauses could be checked '
           r"automatically every quarter; the rest go to an analyst. Score: the first formalizer's "
           r'case-level score on each group.}',
           r'\label{tab:selfcheck}', r'\begin{tabular}{@{}lccc@{}}', r'\toprule',
           r'\textbf{Pair of formalizers} & \textbf{Agree (coverage)} & \textbf{Score, agreed} & \textbf{Score, to analyst} \\', r'\midrule']
    md = ['| Pair | Agree (coverage) | Score, agreed | Score, to analyst |', '|---|---|---|---|']
    out = []
    for a, b, rel, cache, lab_t, lab_m in CROSS:
        r = cross_check(a, b, rel, cache)
        out.append((lab_m, r))
        lat.append(f"{lab_t} & {r['agreed']} / {r['n']} ({pct(r['coverage'], 0)}\\%) & {pct(r['acc_agreed'])} & {pct(r['acc_other'])} \\\\")
        md.append(f"| {lab_m} | {r['agreed']} / {r['n']} ({pct(r['coverage'], 0)}%) | {pct(r['acc_agreed'])} | {pct(r['acc_other'])} |")
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    write('tab5_selfcheck', '\n'.join(lat), '\n'.join(md))
    return '\n'.join(lat), '\n'.join(md), out


def table_sensitivity():
    S = system_stats()
    sched = schedule_clauses()
    rows = []
    for k, lab, short, _ in DIRECT[:3]:
        short_m = marks(f'answers/direct_{ {"qwen7b": "qwen7b", "qwen32b": "qwen32b", "qwen27b": "qwen27b"}[k] }.jsonl')
        cut = sum(json.loads(l).get('error') == 'hit_token_limit' for l in open(RES / f'answers/direct_{k}.jsonl'))
        cut_l = sum(json.loads(l).get('error') == 'hit_token_limit' for l in open(RES / f'answers_long/direct_{k}.jsonl'))
        rows.append(('Answer-token limit (direct readers)', f'{lab}: 1,024 $\\rightarrow$ 4,096 tokens',
                     f'{lab}: 1,024 → 4,096 tokens', pct(mean(per_clause(short_m))), pct(S[k]['score']),
                     f'cut off: {cut} $\\rightarrow$ {cut_l}', f'cut off: {cut} → {cut_l}'))
    for k, lab, rel in FORMULA[1:]:
        len_m = marks(rel.replace('answers/', 'answers_lenient/'))
        rows.append(('Formula interface (formalizers)', f'{lab}: strict $\\rightarrow$ typed subset of parameters',
                     f'{lab}: strict → typed subset', pct(S[k]['score']), pct(mean(per_clause(len_m))), '', ''))
    for k, lab in (('sft', 'SFT'), ('rl_v2', 'SFT + RL v2 (3 seeds)')):
        runs = [S[f'rl_v2_s{s}']['pc'] for s in RL_V2] if k == 'rl_v2' else [S[k]['pc']]
        a = [np.mean([pc[c] for c in pc if c not in sched]) for pc in runs]
        b = [np.mean([pc[c] for c in pc if c in sched]) for pc in runs]
        fa = lambda v, t: pct(np.mean(v)) + ((r'{\scriptsize$\pm$' + pct(np.std(v, ddof=1)) + '}') if t and len(v) > 1 else
                                            (f' ± {pct(np.std(v, ddof=1))}' if len(v) > 1 else ''))
        n_a, n_b = sum(c not in sched for c in runs[0]), sum(c in sched for c in runs[0])
        rows.append(('Clause type', f'{lab}: no schedule/condition $\\rightarrow$ with', f'{lab}: no schedule/condition → with',
                     (fa(a, 1), fa(a, 0)), (fa(b, 1), fa(b, 0)), f'$n$ = {n_a} / {n_b}', f'n = {n_a} / {n_b}'))
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
           r'\caption{Sensitivity analyses (per-clause score). Direct readers were first run with a 1,024-token answer '
           r'limit; long-reasoning models were cut off, so the main table uses 4,096. The strict interface rule (the '
           r'pre-registered one) is used throughout the paper; the lenient rule only diagnoses RL v1. RL v2 was trained '
           r'only on clauses without date schedules or conditions; its rows are mean $\pm$ s.d. over three seeds.}',
           r'\label{tab:sensitivity}', r'\begin{tabular}{@{}llrrl@{}}', r'\toprule',
           r'\textbf{Analysis} & \textbf{Setting: A $\rightarrow$ B} & \textbf{A} & \textbf{B} & \textbf{Note} \\', r'\midrule']
    md = ['| Analysis | Setting: A → B | A | B | Note |', '|---|---|---|---|---|']
    last = None
    for grp, s_t, s_m, a, b, n_t, n_m in rows:
        if grp != last and last is not None:
            lat.append(r'\midrule')
        (a_t, a_m), (b_t, b_m) = [x if isinstance(x, tuple) else (x, x) for x in (a, b)]
        lat.append(f"{tex(grp) if grp != last else ''} & {s_t} & {a_t} & {b_t} & {n_t} \\\\")
        md.append(f"| {grp if grp != last else ''} | {s_m} | {a_m} | {b_m} | {n_m} |")
        last = grp
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return write('tab6_sensitivity', '\n'.join(lat), '\n'.join(md))


# --------------------------------------------------------------------------------------------- figures
COLORS = {'qwen7b': PALETTE['red_1'], 'qwen32b': PALETTE['red_2'], 'qwen27b': PALETTE['red_strong'],
          'prompted': PALETTE['green_1'], 'sft': PALETTE['blue_secondary'], 'rl_v1': PALETTE['green_2'], 'rl_v2': PALETTE['blue_main']}
HATCH = {'qwen7b': '//', 'qwen32b': '//', 'qwen27b': '//'}


def fig_main():
    """Fig. 1: overall score and the three questions; open readers (hatched reds), our formalizers (blues)."""
    S = system_stats()
    keys = ['qwen7b', 'qwen32b', 'qwen27b', 'sft', 'rl_v2']
    labels = {'qwen7b': 'Qwen2.5-Coder-7B, reads directly', 'qwen32b': 'Qwen2.5-Coder-32B, reads directly',
              'qwen27b': 'Qwen3.8-27B, reads directly',
              'sft': 'Ours: 7B SFT formalizer + Z3', 'rl_v2': 'Ours: 7B SFT + RL formalizer + Z3 (3 seeds)'}
    metrics = [('score', 'Overall (per clause)'), ('status', 'Status'), ('cushion', 'Earnings cushion'), ('capacity', 'Debt capacity')]
    fig, axes = plt.subplots(1, 5, figsize=(26, 6.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.95]))
    for ax, (m, title) in zip(axes, metrics):
        x = np.arange(len(keys))
        vals = [100 * S[k][m] for k in keys]
        bars = ax.bar(x, vals, width=0.78, color=[COLORS[k] for k in keys], edgecolor='black', linewidth=1.8,
                      hatch=None, zorder=2)
        for b, k in zip(bars, keys):
            if k in HATCH:
                b.set_hatch(HATCH[k])
        sd = S['rl_v2']['score_sd'] if m == 'score' else S['rl_v2'][f'{m}_sd']
        i = keys.index('rl_v2')
        ax.errorbar(x[i], vals[i], yerr=100 * sd, color='black', capsize=6, capthick=2, linewidth=2, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (100 * sd if xi == i else 0) + 1.5, f'{v:.1f}', ha='center', va='bottom', fontsize=14,
                    fontweight='bold' if keys[xi] == 'rl_v2' else 'normal')
        ax.set_title(title, fontsize=18, pad=12)
        ax.set_xticks([])
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylabel('Correct (%)' if m == 'score' else '')
    axes[-1].set_axis_off()
    handles = [Patch(facecolor=COLORS[k], edgecolor='black', linewidth=1.5, hatch=HATCH.get(k), label=labels[k]) for k in keys]
    axes[-1].legend(handles=handles, loc='center left', fontsize=14, handlelength=2.2, handleheight=1.4, labelspacing=1.0)
    return finalize(fig, 'fig1_main')


def fig_size():
    """Fig. 2: accuracy against model size; direct readers (line) vs our 7B formalizers (blue)."""
    S = system_stats()
    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    xs = {'qwen7b': 7, 'qwen27b': 27, 'qwen32b': 32}
    pts = sorted(xs, key=lambda k: xs[k])
    ax.plot([xs[k] for k in pts], [100 * S[k]['score'] for k in pts], color=PALETTE['red_strong'], linewidth=1.8,
            linestyle=(0, (4, 3)), marker='o', markersize=11, markerfacecolor=PALETTE['red_2'], markeredgecolor='black', markeredgewidth=1.5,
            label='Open models reading the clause directly', zorder=3)
    for k in pts:
        lo, hi = S[k]['ci']
        ax.vlines(xs[k], 100 * lo, 100 * hi, color=PALETTE['red_strong'], linewidth=2, alpha=0.6, zorder=2)
        ax.annotate(S[k]['label'], (xs[k], 100 * S[k]['score']), xytext=(10, -22 if k != 'qwen27b' else 14),
                    textcoords='offset points', fontsize=13)
    for k, mk, off, lab, xo in (('sft', 's', (14, -6), 'Ours: SFT', 6.2), ('rl_v2', '*', (16, -6), 'Ours: SFT + RL (3 seeds)', 7.8)):
        lo, hi = S[k]['ci']
        ax.vlines(xo, 100 * lo, 100 * hi, color=COLORS[k], linewidth=3, zorder=4)
        ax.plot(xo, 100 * S[k]['score'], marker=mk, markersize=20 if mk == '*' else 13, color=COLORS[k],
                markeredgecolor='black', markeredgewidth=1.5, linestyle='', zorder=5)
        ax.annotate(f"{lab}  {100 * S[k]['score']:.1f}", (xo, 100 * S[k]['score']), xytext=off, textcoords='offset points',
                    fontsize=14, fontweight='bold' if k == 'rl_v2' else 'normal', color=COLORS[k])
    ax.set_xlabel('Parameters of the model answering (billions)')
    ax.set_ylabel('Correct per clause (%)')
    ax.set_xlim(3, 41)
    ax.set_xticks([7, 27, 32])
    ax.set_ylim(55, 102)
    ax.legend(loc='lower right', fontsize=13)
    return finalize(fig, 'fig2_size')


def _rolling(y, w):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(w) / w, mode='valid') if len(y) >= w else y


def fig_training():
    """Fig. 3: learning curves. (a) SFT loss; RL v2, three seeds: (b) reward of the sampled formulas, (c) share of GRPO
    steps whose four samples differ in reward (the only steps with a gradient), (d) share of samples keeping the interface."""
    sft = [h for h in json.loads((P.MODELS / 'sft789/trainer_state.json').read_text())['log_history'] if 'loss' in h]
    seeds = (42, 43, 44)
    curves = defaultdict(list)
    for sd in seeds:
        rows = [json.loads(l) for l in open(P.MODELS / f'rl_v2_s{sd}/exec_rewards.jsonl')]
        per_step = np.array([r['reward'] for r in rows]).reshape(-1, 4)
        ok = np.array([float(r['valid_signature']) for r in rows]).reshape(-1, 4)
        curves['reward'].append(_rolling(per_step.mean(1), 50))
        curves['signal'].append(_rolling((per_step.std(1) > 0).astype(float) * 100, 50))
        curves['sig'].append(_rolling(ok.mean(1) * 100, 50))
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.4))
    ax = axes[0]
    st = [h['step'] for h in sft]
    ax.plot(st, [h['loss'] for h in sft], color=PALETTE['blue_secondary'], linewidth=1.2, alpha=0.45, label='per step')
    ax.plot(st[len(st) - len(_rolling([h['loss'] for h in sft], 10)):], _rolling([h['loss'] for h in sft], 10),
            color=PALETTE['blue_main'], linewidth=3, label='10-step mean')
    ax.set_title('(a) SFT: training loss', loc='left', fontsize=19)
    ax.set_xlabel('SFT step (1 epoch, 789 pairs)')
    ax.set_ylabel('Cross-entropy loss')
    ax.set_xlim(0, st[-1])
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=15)
    panels = [('reward', '(b) GRPO: reward', 'Mean reward per step', (0.85, 1.005)),
              ('signal', '(c) GRPO: steps with a learning signal', 'Steps whose samples differ (%)', (0, None)),
              ('sig', '(d) GRPO: interface kept', 'Samples with exact parameters (%)', (90, 100.5))]
    for ax, (key, title, ylab, ylim) in zip(axes[1:], panels):
        n = min(len(c) for c in curves[key])
        arr = np.array([c[:n] for c in curves[key]])
        steps = np.arange(n) + 50
        ax.fill_between(steps, arr.min(0), arr.max(0), color=PALETTE['blue_secondary'], alpha=0.25, linewidth=0,
                        label='range over 3 seeds')
        ax.plot(steps, arr.mean(0), color=PALETTE['blue_main'], linewidth=3, label='mean over 3 seeds')
        ax.set_title(title, loc='left', fontsize=19)
        ax.set_xlabel('GRPO step (50-step mean)')
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 1000)
        ax.set_ylim(*ylim)
    axes[1].legend(loc='lower right', fontsize=15)
    for ax in axes:
        ax.tick_params(labelsize=15)
        ax.xaxis.label.set_size(17)
        ax.yaxis.label.set_size(17)
    return finalize(fig, 'fig3_training')


def fig_verification():
    """Fig. 4: (a) formal verification of each formalizer's formulas; (b) label-free self-check: coverage and score."""
    ver = verification()
    S = system_stats()
    rows = [('prompted', 'Prompted 7B'), ('sft', 'SFT'), ('rl_v1', 'SFT + RL v1')] + [(f'rl_v2_s{s}', f'SFT + RL v2 ({s})') for s in RL_V2]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.4), gridspec_kw=dict(width_ratios=[1.35, 1, 0.55]))
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    segs = [('equivalent', 'Proven equivalent', PALETTE['blue_main'], None), ('different', 'Counterexample found', PALETTE['red_2'], '//'),
            ('invalid', 'Unusable (invalid / wrong interface)', PALETTE['neutral'], '..')]
    left = np.zeros(len(rows))
    for key, lab, col, h in segs:
        vals = np.array([ver[k][key] for k, _ in rows])
        ax.barh(y, vals, left=left, color=col, edgecolor='black', linewidth=1.5, hatch=h, height=0.7, label=lab)
        for yi, l, v in zip(y, left, vals):
            if v >= 8:
                ax.text(l + v / 2, yi, str(int(v)), ha='center', va='center', fontsize=13,
                        color='white' if key == 'equivalent' else PALETTE['ink'])
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in rows])
    ax.set_xlabel('Clauses (of 143)')
    ax.set_xlim(0, 143)
    ax.set_title('(a) Formal verification against the reference', loc='left', fontsize=17)
    ax2 = axes[1]
    cross = [(lab, cross_check(a, b, rel, cache)) for a, b, rel, cache, _, lab in CROSS]
    x = np.arange(len(cross))
    w = 0.36
    b1 = ax2.bar(x - w / 2, [100 * r['acc_agreed'] for _, r in cross], width=w, color=PALETTE['blue_main'], edgecolor='black',
                 linewidth=1.5, label='Formulas agree: automatic')
    b2 = ax2.bar(x + w / 2, [100 * r['acc_other'] for _, r in cross], width=w, color=PALETTE['red_1'], edgecolor='black',
                 linewidth=1.5, hatch='//', label='Do not agree: to an analyst')
    for bars in (b1, b2):
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=13)
    for xi, (_, r) in zip(x, cross):
        ax2.text(xi - w / 2, 50, f"{100 * r['coverage']:.0f}%\nof\nclauses", ha='center', va='center', fontsize=12,
                 color='white', fontweight='bold', linespacing=1.1)
    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace(' × ', '\n× ') for l, _ in cross], fontsize=13)
    ax2.set_ylabel('Correct per clause (%)')
    ax2.set_ylim(0, 128)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.legend(loc='upper center', ncol=2, fontsize=12, columnspacing=1.2, handleheight=1.2, bbox_to_anchor=(0.5, 1.02))
    ax2.set_title('(b) Label-free self-check: two formalizers agree?', loc='left', fontsize=17)
    axes[2].set_axis_off()
    h1, l1 = ax.get_legend_handles_labels()
    axes[2].legend(h1, l1, loc='center left', fontsize=13, labelspacing=1.0, handleheight=1.3, title='(a)', title_fontsize=14)
    return finalize(fig, 'fig4_verification')


def fig_cases():
    """Fig. 5: score by case type and clause type: 7B reading directly vs our formalizers."""
    S = system_stats()
    sched = schedule_clauses()
    groups = [('Cushion\n$-$5% / $-$1%', lambda c: c['kind'] == 'headroom' and re.search(r'target [-]\d', c['note'])),
              ('Cushion\n+1% / +5%', lambda c: c['kind'] == 'headroom' and re.search(r'target \+[15]%', c['note'])),
              ('Cushion\n+20%', lambda c: c['kind'] == 'headroom' and '+20%' in c['note']),
              ('Missing\nfigure', lambda c: c['kind'] == 'missing'),
              ('No schedule /\ncondition', lambda c: c['clause_id'] not in sched),
              ('With schedule /\ncondition', lambda c: c['clause_id'] in sched)]
    keys = [('qwen7b', 'answers_long/direct_qwen7b.jsonl'), ('qwen27b', 'answers_long/direct_qwen27b.jsonl'),
            ('sft', 'answers/formula_sft789.jsonl'), ('rl_v2', None)]
    labels = {'qwen7b': 'Qwen2.5-Coder-7B, reads directly', 'qwen27b': 'Qwen3.8-27B, reads directly', 'sft': 'Ours: SFT', 'rl_v2': 'Ours: SFT + RL (3 seeds)'}

    def acc(rel, keep):
        if rel is None:
            return float(np.mean([mean(per_clause(marks(r), keep)) for r in RL_V2.values()]))
        return mean(per_clause(marks(rel), keep))
    fig, axes = plt.subplots(1, 2, figsize=(22, 6.2), gridspec_kw=dict(width_ratios=[1, 0.32]))
    ax = axes[0]
    x = np.arange(len(groups))
    w = 0.2
    for j, (k, rel) in enumerate(keys):
        vals = [100 * acc(rel, keep) for _, keep in groups]
        bars = ax.bar(x + (j - 1.5) * w, vals, width=w, color=COLORS[k], edgecolor='black', linewidth=1.5, hatch=HATCH.get(k), label=labels[k])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f'{b.get_height():.0f}', ha='center', va='bottom', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([g for g, _ in groups], fontsize=14)
    ax.axvline(3.5, color=PALETTE['gray_mid'], linewidth=1.5, linestyle=(0, (4, 4)))
    ax.text(1.5, 106, 'by case type', ha='center', fontsize=14, color=PALETTE['gray_dark'])
    ax.text(4.5, 106, 'by clause type', ha='center', fontsize=14, color=PALETTE['gray_dark'])
    ax.set_ylabel('Correct per clause (%)')
    ax.set_ylim(0, 112)
    axes[1].set_axis_off()
    h, l = ax.get_legend_handles_labels()
    axes[1].legend(h, l, loc='center left', fontsize=14, labelspacing=1.0, handleheight=1.3)
    return finalize(fig, 'fig5_cases')





# --------------------------------------------------------------------------------------------- overview figure

EXAMPLE = ('fin_320522', '-1%')  # LINC Logistics (2011): four covenants; the case placed at an earnings cushion of -1%


TIKZ_OVERVIEW = r"""% Fig. 0 (overview): the analyst's question, reading directly vs. formalize-then-solve, and headroom accuracy.
% Generated by src/headroom/paper.py (fig_overview_tex); every number comes from the result files.
% Compile standalone (pdflatex) or \input{} the tikzpicture into the paper (needs the packages and colours below).
\documentclass[tikz,border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[scaled=0.92]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pifont}
\usepackage{pgfplots}
\usepackage[eulergreek]{sansmath}
\renewcommand{\ttdefault}{lmtt}
\pgfplotsset{compat=1.17}
\usetikzlibrary{arrows.meta,backgrounds,fit,patterns,positioning}
\definecolor{bluemain}{HTML}{0F4D92}
\definecolor{bluebg}{HTML}{E8EFF8}
\definecolor{redone}{HTML}{F6CFCB}
\definecolor{redtwo}{HTML}{E9A6A1}
\definecolor{redstrong}{HTML}{B64342}
\definecolor{neutral}{HTML}{CFCECE}
\definecolor{graydark}{HTML}{4D4D4D}
\definecolor{graymid}{HTML}{767676}
\definecolor{ink}{HTML}{272727}
\definecolor{paperbg}{HTML}{F7F7F5}
\begin{document}
\sansmath\hyphenpenalty=10000\exhyphenpenalty=10000
\begin{tikzpicture}[
    font=\small, >={Stealth[length=2.6mm,width=2mm]},
    box/.style={draw=ink, rounded corners=3pt, line width=0.8pt, inner sep=6pt, align=left, anchor=north west},
    head/.style={font=\bfseries\large, anchor=west, inner sep=0pt},
    sub/.style={font=\bfseries},
    note/.style={color=graydark},
    flow/.style={->, line width=1.1pt, color=graymid},
  ]
  % ---------------------------------------------------------------- (a) the analyst's quarterly question
  \node[head] at (0, 0) {(a) The analyst's quarterly question};
  \node[box, fill=paperbg, text width=6.1cm] (clause) at (0, -0.45) {
    {\color{graydark}Credit agreement: <<COMPANY>> (<<YEAR>>)}\\[4pt]
    \textit{7.9 Financial Covenants. (a) Maintain at all times a Senior Debt to EBITDA Ratio of not more than
    2.5 to 1.0. (b) \ldots{} a Total Debt to EBITDA Ratio of not more than 3.0 to 1.0. (c), (d) Fixed Charge
    ratios \ldots}\\[8pt]
    {\color{graydark}Compliance certificate, <<DATE>>}\\[2pt]
    \begin{tabular}{@{}l@{\hspace{2em}}r@{}}
      EBITDA & \$<<EBITDA>>M\\
      Senior Debt & \$<<SENIOR>>M\\
      Total Debt & \$<<TOTAL>>M\\
    \end{tabular}\\
    {\footnotesize\color{graydark}(+ two fixed-charge items)}
  };
  \node[box, fill=white, text width=6.1cm, below=0.35cm of clause.south west, anchor=north west] (questions) {
    {\bfseries 1.\ Status}\\ {\color{graydark}comply, breach or undetermined?}\\[4pt]
    {\bfseries 2.\ Earnings cushion}\\ {\color{graydark}how far can EBITDA fall before a breach?}\\[4pt]
    {\bfseries 3.\ Debt capacity}\\ {\color{graydark}how much more debt is allowed?}\\[7pt]
    {\itshape\color{bluemain}Confidential figures, every quarter, every loan:\\ in-house and auditable.}
  };

  % ---------------------------------------------------------------- (b) two ways to answer
  \node[head] at (7.2, 0) {(b) Two ways to answer};
  \node[box, fill=redone, text width=5.4cm] (direct) at (7.2, -0.45) {
    {\bfseries\normalsize Read the clause directly}\\[4pt]
    An LLM reads clause and certificate and answers in text (Qwen2.5-Coder-7B):\\[6pt]
    {\ttfamily\bfseries\color{redstrong} status <<D7STATUS>>\quad cushion <<D7CUSHION>>}\\[4pt]
    {\itshape\color{graydark}multi-step arithmetic in free text}
  };
  % formalize, then solve (ours): three stacked steps on a blue panel
  \node[box, fill=white, draw=bluemain, text width=5.1cm, below=1.05cm of direct.south west, xshift=0.15cm,
        anchor=north west] (formalizer) {
    {\bfseries Open 7B formalizer (in-house)}\\[2pt]
    {\footnotesize\color{graydark} SFT on 789 teacher pairs, then RL on unlabeled clauses with a solver-checked reward}
  };
  \node[box, fill=paperbg, draw=graymid, text width=5.1cm, below=0.45cm of formalizer.south west, anchor=north west]
        (formula) {
    {\ttfamily\scriptsize
     (define-fun covenant\_holds\\
     \hspace*{0.6em}((ebitda Real) (senior\_debt Real)\\
     \hspace*{0.9em}(total\_debt Real) \ldots) Bool\\
     \hspace*{0.6em}(and (<= (/ senior\_debt ebitda) 2.5)\\
     \hspace*{2.6em}(<= (/ total\_debt ebitda) 3.0)\\
     \hspace*{2.6em}\ldots))}
  };
  \node[inner sep=0pt, align=left, below=0.5cm of formula.south west, anchor=north west, xshift=6pt] (result) {
    {\bfseries Z3 + certificate facts:}\\[4pt]
    {\ttfamily\bfseries\color{bluemain} status <<OSTATUS>>\quad cushion <<OCUSHION>>}\\[3pt]
    {\ttfamily\color{bluemain} capacity <<OCAPACITY>>M USD}
  };
  \node[anchor=north east, color=bluemain, font=\LARGE, inner sep=0pt] (tick) at ([xshift=-4pt]formula.east |- result.north) {\ding{51}};
  \draw[->, line width=1.1pt, color=bluemain] (formalizer.south) -- (formula.north);
  \draw[->, line width=1.1pt, color=bluemain] (formula.south) -- (formula.south |- result.north);
  \begin{scope}[on background layer]
    \node[draw=bluemain, line width=1.3pt, rounded corners=3pt, fill=bluebg, inner xsep=0.15cm, inner ysep=0.2cm,
          fit={(formalizer) (formula) (result) ([yshift=0.7cm]formalizer.north)}] (ours) {};
  \end{scope}
  \node[anchor=north west, font=\bfseries\normalsize, color=bluemain] at ([xshift=0.15cm, yshift=-0.12cm]ours.north west)
       {Formalize, then solve (ours)};

  % ---------------------------------------------------------------- verdicts
  \node[box, fill=white, draw=graymid, text width=5.3cm] (reference) at (13.9, -0.45) {
    {\bfseries\normalsize Reference answer}\\[4pt]
    {\ttfamily status <<RSTATUS>>\quad cushion <<RCUSHION>>\\ capacity <<RCAPACITY>>M USD}\\[6pt]
    {\color{graydark} EBITDA must not fall below Total Debt / 3.0: the borrower is already 1\% short.}
  };
  \node[box, fill=bluebg, draw=bluemain, line width=1.1pt, text width=5.3cm] (verify) at (13.9, -4.55) {
    {\bfseries\normalsize\color{bluemain} Verify once, reuse every quarter}\\[6pt]
    {\bfseries Prove.} Z3 checks the formula once, for every possible certificate: against a reference reading, or
    against a second, independent formalizer.\\[5pt]
    {\bfseries Run.} Verified covenants are checked automatically every quarter: exact, repeatable, milliseconds.\\[5pt]
    {\bfseries Escalate.} Covenants whose formulas disagree go to an analyst.
  };

  % ---------------------------------------------------------------- arrows
  \draw[flow] (clause.east |- direct.west) -- (direct.west);
  \draw[flow] (questions.east |- ours.west) -- (ours.west);
  \draw[<->, line width=1.1pt, color=redstrong] (direct.east) -- (reference.west |- direct.east)
       node[midway, above=2pt, font=\LARGE] {\ding{55}};
  \draw[->, line width=1.1pt, color=bluemain] (ours.east |- verify.west) -- (verify.west);

  % ---------------------------------------------------------------- (c) analysis
  \node[head] at (20.3, 0) {(c) Headroom accuracy};
  \node[anchor=west, note] at (20.3, -0.5) {143 clauses from borrowers never seen in training};
  \begin{axis}[
      at={(21.2cm, -8.6cm)}, anchor=south west, width=6.9cm, height=8.4cm,
      ybar, bar width=0.55cm, ymin=0, ymax=108, ytick={0,20,40,60,80,100},
      ylabel={Earnings cushion correct (\%)}, ylabel style={font=\small},
      symbolic x coords={q7,q32,q27,ours}, xtick={q7,q32,q27,ours},
      xticklabels={{Qwen\\7B}, {Qwen\\32B}, {Qwen3.8\\27B}, {\textbf{Ours}\\\textbf{7B}}},
      xticklabel style={align=center, font=\footnotesize}, yticklabel style={font=\footnotesize},
      axis x line*=bottom, axis y line*=left, enlarge x limits=0.13, tick align=outside,
      nodes near coords, nodes near coords style={font=\footnotesize, color=ink},
    ]
    \addplot[bar shift=0pt, fill=redone, draw=black, postaction={pattern=north east lines}] coordinates {(q7,<<C7>>)};
    \addplot[bar shift=0pt, fill=redtwo, draw=black, postaction={pattern=north east lines}] coordinates {(q32,<<C32>>)};
    \addplot[bar shift=0pt, fill=redstrong, draw=black, postaction={pattern=north east lines}] coordinates {(q27,<<C27>>)};
    \addplot[bar shift=0pt, fill=bluemain, draw=black, nodes near coords style={font=\footnotesize\bfseries}]
      coordinates {(ours,<<COURS>>)};
  \end{axis}
  \node[anchor=west, note, font=\footnotesize] at (20.3, -10.05)
       {Hatched: reads the clause directly. Solid: formalizes, then Z3.};
  \node[anchor=west, note, font=\footnotesize] at (20.3, -10.5)
       {Overall score: ours <<SOURS>>, Qwen3.8 27B <<S27>>, Qwen 32B <<S32>>, Qwen 7B <<S7>>};
\end{tikzpicture}
\end{document}
"""


def fig_overview_tex():
    """Fig. 0 as TikZ / pgfplots LaTeX (results/paper/figures/fig0_overview.tex), numbers filled in from the files."""
    S = system_stats()
    spec = specs()[EXAMPLE[0]]
    case = next(c for c in cases().values() if c['clause_id'] == EXAMPLE[0] and f'target {EXAMPLE[1]}' in c['note'])
    ref = case['answers']
    d7 = answers('answers_long/direct_qwen7b.jsonl')[case['case_id']]
    ours = answers('answers/formula_rl789_v2.jsonl')[case['case_id']]
    vals = {k: float(v) for k, v in case['values'].items() if v is not None}
    pc = lambda x: (f'{x:+.1f}' if isinstance(x, (int, float)) else str(x)) + r'\%'
    import datetime as dt
    d = case['test_date']
    fill = {
        'COMPANY': tex(spec['company'].split('  ')[0].title()), 'YEAR': str(spec['agreement_year']),
        'DATE': dt.date(d // 10000, d // 100 % 100, d % 100).strftime('%b %-d, %Y'),
        'EBITDA': f"{vals['ebitda'] / 1e6:,.1f}", 'SENIOR': f"{vals['senior_debt'] / 1e6:,.1f}",
        'TOTAL': f"{vals['total_debt'] / 1e6:,.1f}",
        'D7STATUS': d7['status'], 'D7CUSHION': pc(d7['cushion']),
        'OSTATUS': ours['status'], 'OCUSHION': pc(ours['cushion']), 'OCAPACITY': f"{ours['capacity'] / 1e6:+,.2f}",
        'RSTATUS': ref['status'], 'RCUSHION': pc(ref['cushion']), 'RCAPACITY': f"{ref['capacity'] / 1e6:+,.2f}",
        'C7': f"{100 * S['qwen7b']['cushion']:.1f}", 'C32': f"{100 * S['qwen32b']['cushion']:.1f}",
        'C27': f"{100 * S['qwen27b']['cushion']:.1f}", 'COURS': f"{100 * S['rl_v2']['cushion']:.1f}",
        'SOURS': f"{100 * S['rl_v2']['score']:.1f}",
        'S27': f"{100 * S['qwen27b']['score']:.1f}", 'S32': f"{100 * S['qwen32b']['score']:.1f}",
        'S7': f"{100 * S['qwen7b']['score']:.1f}"}
    out = TIKZ_OVERVIEW
    for k, v in fill.items():
        out = out.replace(f'<<{k}>>', v)
    assert '<<' not in out, 'unfilled placeholder'
    path = FIG / 'fig0_overview.tex'
    path.write_text(out)
    return compile_tex(path)


TIKZ_METHOD = r"""% Fig. M (method). Generated by src/headroom/paper.py (fig_method_tex).
% Method figure. Stage I: distill a teacher into SFT; Stage II: GRPO with a Z3-checked reward; Stage III: verify once,
% run every quarter in-house. Compile with pdflatex; needs the fontawesome5 package (CTAN).
\documentclass[tikz,border=8pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[scaled=0.92]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\renewcommand{\ttdefault}{lmtt}
\usepackage[letterspace=90]{microtype}
\usepackage{amsmath,amssymb}
\usepackage{pifont}
\usepackage{fontawesome5}
\usepackage[eulergreek]{sansmath}
\usetikzlibrary{arrows.meta,backgrounds,calc}
\pgfdeclarelayer{page}\pgfsetlayers{page,background,main}
\definecolor{ink}{HTML}{23302D}
\definecolor{teal}{HTML}{6AB3A6}
\definecolor{panel}{HTML}{F4F5F5}
\definecolor{cream}{HTML}{FCEFD4}
\definecolor{mint}{HTML}{DCEFD9}
\definecolor{sky}{HTML}{DFEAF7}
\definecolor{skydark}{HTML}{1F5AA0}
\definecolor{good}{HTML}{2E8B57}
\definecolor{mintdark}{HTML}{3D7A47}
\definecolor{bad}{HTML}{C0453F}
\definecolor{amber}{HTML}{B7791F}
\definecolor{muted}{HTML}{65706D}
\newcommand{\yes}{{\color{good}\ding{51}}}
\newcommand{\ico}[2][ink]{{\color{#1}\fontsize{13}{13}\selectfont #2}}

\begin{document}
\sansmath
\begin{tikzpicture}[
    font=\fontsize{8}{10}\selectfont, >={Stealth[length=2.2mm,width=1.7mm]},
    panelbox/.style={draw=teal, dash pattern=on 2.4pt off 1.8pt, line width=0.9pt, rounded corners=7pt, fill=panel},
    head/.style={font=\fontsize{11}{13}\selectfont\lsstyle, color=ink, anchor=south},
    block/.style={draw=ink, line width=0.9pt, rounded corners=6pt, minimum width=4.4cm, minimum height=0.95cm,
                  anchor=north west, font=\fontsize{12.5}{14}\selectfont\bfseries},
    inset/.style={draw=ink, line width=0.75pt, rounded corners=5pt, fill=white, anchor=north west, minimum width=3.85cm,
                  inner sep=0pt},
    itext/.style={anchor=north west, text width=3.6cm, align=left, inner sep=0pt},
    chip/.style={draw=#1, text=#1, fill=white, rounded corners=1.8pt, line width=0.6pt, inner xsep=4pt, inner ysep=1.6pt,
                 font=\fontsize{7.3}{8}\selectfont},
    chipfill/.style={draw=#1, fill=#1!12, text=#1, rounded corners=1.8pt, line width=0.6pt, inner xsep=4pt, inner ysep=1.6pt,
                 font=\fontsize{7.3}{8}\selectfont\bfseries},
    lab/.style={font=\fontsize{8}{9.5}\selectfont, color=black, align=left},
    flow/.style={->, line width=0.8pt, color=ink, rounded corners=4pt},
    tie/.style={line width=0.75pt, color=ink, rounded corners=4pt},
  ]
  \def\P{5.0}  % width of Stages I and III
  \def\W{6.9}  % width of Stage II
  \def\G{1.35} % gap between panels
  \def\B{-8.75}% panel bottom
  \pgfmathsetmacro\Lb{\P+\G}
  \pgfmathsetmacro\Lc{\P+\W+2*\G}
  \begin{scope}[on background layer]
    \draw[panelbox] (0,0) rectangle (\P,\B);
    \draw[panelbox] (\Lb,0) rectangle (\Lb+\W,\B);
    \draw[panelbox] (\Lc,0) rectangle (\Lc+\P,\B);
  \end{scope}
  \node[head] at ({\P/2}, 0.1) {Stage I: DISTILLATION};
  \node[head] at ({\Lb+\W/2}, 0.1) {Stage II: GRPO, SOLVER REWARD};
  \node[head] at ({\Lc+\P/2}, 0.1) {Stage III: DEPLOYMENT};

  % ================================================================ Stage I
  \def\L{0}
  \node[block, fill=white] (teacher) at (\L+0.3,-0.35) {\phantom{x}Teacher model};
  \node at ($(teacher.west)+(0.5,0)$) {\ico{\faRobot}};
  \node[inset, minimum height=2.2cm] (t1) at (\L+0.85,-1.7) {};
  \node[itext] at (\L+1.0,-1.95) {%
    {\bfseries teacher $\varphi$ for $x\in\mathcal{C}$, kept if}\\[6pt]
    \yes\ Z3 parses it\\[1pt]
    \yes\ semantic review agrees\\[1pt]
    \yes\ parameters grounded};
  \draw[tie] (\L+0.62,-1.3) |- (t1.west);
  \node[block, fill=cream] (sft) at (\L+0.3,-4.7) {\phantom{x}SFT};
  \node at ($(sft.west)+(0.5,0)$) {\ico{\faGraduationCap}};
  \draw[flow] (\L+0.4,-1.3) -- (\L+0.4,-4.7);
  \node[lab, anchor=west] at (\L+0.55,-4.32) {filtered pairs $\mathcal{D}$};
  \node[inset, minimum height=2.35cm] (t2) at (\L+0.85,-6.05) {};
  \node[itext] at (\L+1.0,-6.2) {%
    {\bfseries\boldmath $(x,\varphi)\in\mathcal{D}$}\\[3pt]
    $x$:\ {\itshape ``Total Debt to EBITDA \ldots{} not more than $t$ to 1''}\\[2pt]
    $\varphi$:\ {\ttfamily\fontsize{7.6}{9}\selectfont (<= (/ debt ebitda) $t$)}};
  \node[anchor=west, font=\fontsize{7.4}{9}\selectfont] at (\L+0.98,-8.02)
       {$\mathcal{L}_{\mathrm{SFT}}=-\,\mathbb{E}_{\mathcal{D}}\log\pi_\theta(\varphi\mid x)$};
  \draw[tie] (\L+0.62,-5.65) |- (t2.west);

  % ================================================================ Stage II
  \def\L{\Lb}
  \node[block, fill=mint, minimum width={(\W-0.6)*1cm}] (grpo) at (\L+0.3,-0.35) {\phantom{x}Policy $\pi_\theta$};
  \node at ($(grpo.west)+(0.5,0)$) {\ico{\faDiceFour}};
  \node[inset, minimum width={(\W-1.15)*1cm}, minimum height=2.2cm] (g1) at (\L+0.85,-1.7) {};
  \node[itext, text width={(\W-1.45)*1cm}] at (\L+1.0,-1.85) {{\bfseries $G$ samples per clause in $\mathcal{C}'$}};
  \draw[tie] (\L+0.62,-1.3) |- (g1.west);
  \foreach \k [count=\j] in {1, 2, 3, G} {
    \pgfmathsetmacro\x{\L+1.75+(\j-1)*1.25+(\j==4)*0.5}
    \node[color=mintdark] at (\x,-2.55) {\fontsize{11}{11}\selectfont\faFileCode};
    \node[color=mintdark, font=\fontsize{7.4}{8}\selectfont] at (\x,-2.98) {$\varphi_\k$};
    \node[chipfill=skydark] at (\x,-3.42) {$r_\k$};
  }
  \node[color=ink] at (\L+5.0,-2.98) {$\cdots$};

  \node[block, fill=sky, minimum width={(\W-0.6)*1cm}] (z3) at (\L+0.3,-4.7) {\phantom{x}Reward $r(\varphi)$};
  \node at ($(z3.west)+(0.5,0)$) {\ico[skydark]{\faCogs}};
  \draw[flow] (\L+0.4,-1.3) -- (\L+0.4,-4.7);
  \draw[flow] (\L+0.22,-4.7) -- (\L+0.22,-1.3);
  \node[lab, anchor=west] at (\L+0.55,-4.32) {$\varphi_i\downarrow$\ \ $r_i\uparrow$};
  \node[inset, minimum width={(\W-1.15)*1cm}, minimum height=2.35cm] (g2) at (\L+0.85,-6.05) {};
  \draw[tie] (\L+0.62,-5.65) |- (g2.west);
  \node[itext, text width={(\W-1.45)*1cm}] at (\L+1.0,-6.2) {{\bfseries clause limit}\ \ $\tau\le t$, tested at};
  \foreach \n/\q/\a/\c/\m [count=\j] in {compliant/(1-\delta)\,t/sat/good/51, boundary/t/sat/good/51, breach/(1+\delta)\,t/unsat/bad/55} {
    \pgfmathsetmacro\x{\L+1.95+(\j-1)*1.75}
    \node[font=\fontsize{7.4}{8}\selectfont\bfseries] at (\x,-6.72) {\n};
    \node[font=\fontsize{7.2}{8}\selectfont] at (\x,-7.07) {$\textcolor{mintdark}{\varphi}\wedge\tau=\q$};
    \node[chipfill=\c] at (\x,-7.48) {\a\ \ding{\m}};
  }
  \node[anchor=west, font=\fontsize{7.4}{9}\selectfont] at (\L+0.98,-8.04)
       {$\textcolor{skydark}{r(\textcolor{mintdark}{\varphi})}=\mathbb{1}[\textcolor{mintdark}{\varphi}\text{ valid}]\cdot\big(\alpha+(1-\alpha)\,\tfrac{\#\,\text{passed}}{\#\,\text{tests}}\big)$};

  % ================================================================ Stage III
  \def\L{\Lc}
  \node[block, fill=white] (form) at (\L+0.3,-0.35) {\hspace*{0.75cm}SmallFormalizer};
  \node at ($(form.west)+(0.5,0)$) {\ico{\faServer}};
  \node[inset, minimum height=2.2cm] (f1) at (\L+0.85,-1.7) {};
  \node[itext] at (\L+1.0,-1.85) {{\bfseries Equivalence check}};
  \draw[tie] (\L+0.62,-1.3) |- (f1.west);
  \node[color=mintdark] (fa) at (\L+1.75,-2.85) {\fontsize{11}{11}\selectfont\faFileCode};
  \node[color=ink] (fb) at (\L+3.85,-2.85) {\fontsize{11}{11}\selectfont\faFileCode};
  \draw[<->, line width=0.7pt, color=skydark] ($(fa.east)+(0.05,0)$) -- node[above=0pt, font=\fontsize{7.3}{8}\selectfont\bfseries] {$\varphi\equiv\varphi'$\,?} ($(fb.west)+(-0.05,0)$);
  \node[chipfill=good, anchor=west] at (\L+0.95,-3.55) {$\equiv$: deploy};
  \node[chipfill=bad, anchor=east] at (\L+\P-0.42,-3.55) {$\not\equiv$: analyst};
  \node[block, fill=sky] (run) at (\L+0.3,-4.7) {\hspace*{0.6cm}Compliance test};
  \node at ($(run.west)+(0.5,0)$) {\ico[skydark]{\faCalendarCheck}};
  \draw[flow] (\L+0.4,-1.3) -- (\L+0.4,-4.7);
  \node[lab, anchor=west] at (\L+0.55,-4.32) {verified formulas};
  \node[inset, minimum height=2.35cm] (f2) at (\L+0.85,-6.05) {};
  \node[itext] at (\L+1.0,-6.2) {{\bfseries\boldmath $\varphi$ on certificate figures $v$}};
  \draw[tie] (\L+0.62,-5.65) |- (f2.west);
  \node[chipfill=good, anchor=west, inner xsep=2.2pt, font=\fontsize{6.6}{8}\selectfont\bfseries] (s1) at (\L+0.95,-6.92) {comply};
  \node[chipfill=bad, anchor=west, inner xsep=2.2pt, font=\fontsize{6.6}{8}\selectfont\bfseries] (s2) at ($(s1.east)+(0.07,0)$) {breach};
  \node[chipfill=amber, anchor=west, inner xsep=2.2pt, font=\fontsize{6.6}{8}\selectfont\bfseries] at ($(s2.east)+(0.07,0)$) {undetermined};
  \node[anchor=west, font=\fontsize{7.0}{9}\selectfont] at (\L+0.95,-7.5)
       {{\bfseries cushion $\boldsymbol{c}$}: earnings drop to breach};
  \node[anchor=west, font=\fontsize{7.0}{9}\selectfont] at (\L+0.95,-7.98)
       {{\bfseries capacity $\boldsymbol{k}$}: extra debt to breach};

  % ================================================================ between stages
  \draw[flow] (sft.east) -- ++(0.55,0) |- node[lab, pos=0.25, right=1pt, align=left] {init\\$\pi_{\mathrm{SFT}}$} (grpo.west);
  \draw[flow] (grpo.east) -- (form.west);

  \begin{pgfonlayer}{page}   % opaque white page under everything (also for the SVG)
    \fill[white] ([shift={(-6pt,-6pt)}]current bounding box.south west) rectangle ([shift={(6pt,6pt)}]current bounding box.north east);
  \end{pgfonlayer}
\end{tikzpicture}
\end{document}
"""


def fig_method_tex():
    """Method figure as TikZ (results/paper/figures/fig_method.tex); symbols only, values are in the paper caption.
    Needs the fontawesome5 LaTeX package."""
    path = FIG / 'fig_method.tex'
    path.write_text(TIKZ_METHOD)
    return compile_tex(path)


def compile_tex(path):
    """Compile a standalone .tex next to itself (PDF, SVG, and a 300 dpi PNG for the notebook) if pdflatex is installed.
    Returns the PNG when compiled, else the .tex."""
    import shutil
    import subprocess
    import tempfile
    if not shutil.which('pdflatex'):
        print('pdflatex not found: wrote', path, '(compile it yourself)')
        return path
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', '-output-directory', tmp, str(path)],
                           capture_output=True, text=True)
        pdf = Path(tmp) / (path.stem + '.pdf')
        if r.returncode or not pdf.exists():
            raise RuntimeError(f'pdflatex failed on {path}:\n' + r.stdout[-3000:])
        shutil.copy(pdf, path.with_suffix('.pdf'))
    if shutil.which('pdftocairo'):  # vector SVG (text as outlines) for slides and editing tools
        subprocess.run(['pdftocairo', '-svg', str(path.with_suffix('.pdf')), str(path.with_suffix('.svg'))], check=True)
    if shutil.which('pdftoppm'):
        subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile', str(path.with_suffix('.pdf')),
                        str(path.with_suffix(''))], check=True)
        return path.with_suffix('.png')
    return path.with_suffix('.pdf')


def text_stats():
    """Every number the paper text cites that is not in a table, with s.d. over RL seeds and paired tests
    (results/paper/stats.json and stats.md). Tests: two-sided sign-flip over clauses, 95% clause-bootstrap CI."""
    S, sched, ver = system_stats(), schedule_clauses(), verification()
    seeds = list(RL_V2)
    msd = lambda v: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), seeds=[float(x) for x in v])
    test = lambda a, b: {k: float(v) for k, v in paired(a, b).items()}
    out = {}

    # case types: one score per clause within each case type; RL v2 as mean +- s.d. over seeds
    types = {'breach (cushion -5% / -1%)': lambda c: c['kind'] == 'headroom' and re.search(r'target [-]\d', c['note']),
             'thin (cushion +1% / +5%)': lambda c: c['kind'] == 'headroom' and re.search(r'target \+[15]%', c['note']),
             'comfortable (cushion +20%)': lambda c: c['kind'] == 'headroom' and '+20%' in c['note'],
             'missing figure': lambda c: c['kind'] == 'missing'}
    files = dict([(k, rel) for k, _, _, rel in DIRECT] + [(k, rel) for k, _, rel in FORMULA])
    ct = {}
    for k, rel in files.items():
        ct[k] = {t: 100 * mean(per_clause(marks(rel), keep)) for t, keep in types.items()}
    ct['rl_v2'] = {t: msd([100 * mean(per_clause(marks(RL_V2[sd]), keep)) for sd in seeds]) for t, keep in types.items()}
    out['case_types'] = ct
    # does a system lose accuracy on breach cases? per clause: breach-case score minus comfortable-case score
    br, cf = types['breach (cushion -5% / -1%)'], types['comfortable (cushion +20%)']
    gap = {}
    for k, rel in list(files.items()) + [(f'rl_v2_s{sd}', RL_V2[sd]) for sd in seeds]:
        gap[k] = test(per_clause(marks(rel), br), per_clause(marks(rel), cf))
    ks = list(gap)
    for k, ph in zip(ks, holm([gap[k]['p'] for k in ks])):
        gap[k]['p_holm'] = ph
    out['breach_minus_comfortable'] = gap

    # RL v2 minus SFT: per question and per clause type (seed-mean per-clause scores, as in Table 3)
    def pc_q(rel, q):
        by = defaultdict(list)
        for cid, mk in marks(rel).items():
            if q in mk:
                by[cases()[cid]['clause_id']].append(mk[q])
        return {c: sum(v) / len(v) for c, v in by.items()}
    sft_rel = dict((k, r) for k, _, r in FORMULA)['sft']
    rq = {}
    for q, _ in QUESTIONS:
        runs = [pc_q(RL_V2[sd], q) for sd in seeds]
        avg = {c: np.mean([r[c] for r in runs]) for c in runs[0]}
        rq[q] = test(avg, pc_q(sft_rel, q))
    for q, ph in zip(list(rq), holm([rq[q]['p'] for q in rq])):
        rq[q]['p_holm'] = ph
    out['rl_v2_minus_sft_by_question'] = rq
    rv, sf = S['rl_v2']['pc'], S['sft']['pc']
    out['rl_v2_minus_sft_by_clause_type'] = {
        'no schedule/condition': test({c: rv[c] for c in rv if c not in sched}, {c: sf[c] for c in sf if c not in sched}),
        'with schedule/condition': test({c: rv[c] for c in rv if c in sched}, {c: sf[c] for c in sf if c in sched})}

    # verification counts per RL seed
    out['verification_rl_v2'] = {k: msd([ver[f'rl_v2_s{sd}'][k] for sd in seeds])
                                 for k in ('equivalent', 'different', 'unusable') if k in ver[f'rl_v2_s{seeds[0]}']}

    # training: GRPO groups of 4 completions per step
    tr = dict(no_signal=[], valid_signature=[], mean_reward=[], first100=[], last100=[], no_anchor=[],
              informative_steps=[], informative_by_interface=[], informative_by_execution=[],
              informative_pct_first200=[], informative_pct_last700=[], sft_loss_first10=[], sft_loss_last10=[],
              sft_token_acc_first10=[], sft_token_acc_last10=[])
    for sd in seeds:
        rows = [json.loads(l) for l in open(P.MODELS / f'rl_v2_s{sd}/exec_rewards.jsonl')]
        groups = [rows[i:i + 4] for i in range(0, len(rows), 4)]
        tr['no_signal'].append(100 * np.mean([len({r['reward'] for r in g}) == 1 for g in groups]))
        inf = [g for g in groups if len({r['reward'] for r in g}) > 1]  # reward differs inside the group: a gradient
        by_sig = sum(len({r['valid_signature'] for r in g}) > 1 for g in inf)
        tr['informative_steps'].append(len(inf))
        tr['informative_by_interface'].append(by_sig)
        tr['informative_by_execution'].append(len(inf) - by_sig)
        flags = [len({r['reward'] for r in g}) > 1 for g in groups]
        tr['informative_pct_first200'].append(100 * np.mean(flags[:200]))
        tr['informative_pct_last700'].append(100 * np.mean(flags[300:]))
        tr['valid_signature'].append(100 * np.mean([r['valid_signature'] for r in rows]))
        tr['mean_reward'].append(np.mean([r['reward'] for r in rows]))
        tr['first100'].append(np.mean([r['reward'] for r in rows[:400]]))
        tr['last100'].append(np.mean([r['reward'] for r in rows[-400:]]))
        valid = [r for r in rows if r['valid_signature']]
        tr['no_anchor'].append(100 * np.mean([r['exec'] is None for r in valid]))
    sft = [h for h in json.loads((P.MODELS / 'sft789/trainer_state.json').read_text())['log_history'] if 'loss' in h]
    for k, key in (('sft_loss', 'loss'), ('sft_token_acc', 'mean_token_accuracy')):  # one SFT run: s.d. over the 10 steps
        tr[f'{k}_first10'], tr[f'{k}_last10'] = [h[key] for h in sft[:10]], [h[key] for h in sft[-10:]]
    out['training_rl_v2'] = {k: msd(v) for k, v in tr.items()}
    out['training_rl_v2']['steps'] = len(groups)

    (OUT / 'stats.json').write_text(json.dumps(out, indent=1))
    f = lambda d: f"{d['mean']:.1f} ± {d['sd']:.1f}" if isinstance(d, dict) else f'{d:.1f}'
    t = lambda r: f"{100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}], p = {r['p']:.4f}" + (
        f", p_Holm = {r['p_holm']:.4f}" if 'p_holm' in r else '') + f", n = {r['n']:.0f}"
    md = ['# Numbers cited in the paper text (generated by paper.text_stats)', '',
          'RL v2: mean ± s.d. over seeds 42/43/44. Tests: two-sided sign-flip over clauses (10,000), 95% clause bootstrap.', '',
          '## Per-clause score by case type', '', '| System | ' + ' | '.join(types) + ' |', '|---' * (len(types) + 1) + '|']
    md += [f"| {k} | " + ' | '.join(f(ct[k][t_]) for t_ in types) + ' |' for k in ct]
    md += ['', '## Breach-case minus comfortable-case score, per clause (Holm over all rows)', '']
    md += [f'- {k}: {t(r)}' for k, r in gap.items()]
    md += ['', '## RL v2 (seed mean) minus SFT, by question (Holm over three)', '']
    md += [f'- {q}: {t(r)}' for q, r in rq.items()]
    md += ['', '## RL v2 (seed mean) minus SFT, by clause type', '']
    md += [f'- {k}: {t(r)}' for k, r in out['rl_v2_minus_sft_by_clause_type'].items()]
    md += ['', '## Verification, RL v2 (clauses of 143)', '']
    md += [f'- {k}: {f(v)}' for k, v in out['verification_rl_v2'].items()]
    md += ['', f"## Training: RL v2 ({out['training_rl_v2']['steps']} steps, 4 completions each; mean ± s.d. over seeds) and SFT (sft_*: mean ± s.d. over 10 logged steps)", '']
    for k, v in out['training_rl_v2'].items():
        if k != 'steps':
            md.append(f"- {k}: {f(v)}" if k in ('no_signal', 'valid_signature', 'no_anchor') or k.startswith('informative') else
                      f"- {k}: {v['mean']:.3f} ± {v['sd']:.3f}")
    (OUT / 'stats.md').write_text('\n'.join(md) + '\n')
    return OUT / 'stats.md'


def build_all():
    out = dict(tables=[table_benchmark(), table_main(), table_paired(), table_verification(), table_selfcheck()[:2], table_sensitivity()],
               stats=text_stats(),
               figures=[fig_overview_tex(), fig_method_tex(), fig_main(), fig_size(), fig_training(), fig_verification(), fig_cases()])
    return out


if __name__ == '__main__':
    r = build_all()
    for latex, md in r['tables']:
        print(md, '\n')
    print([str(p) for p in r['figures']])

