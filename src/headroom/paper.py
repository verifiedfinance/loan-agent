#!/usr/bin/env python3
"""Paper tables (LaTeX, booktabs) and figures (figures4papers house style) for the headroom experiment.

Everything is computed from the raw files in this directory (data/headroom/, results/headroom/,
models/rl_v2_s*/). Used by notebooks/headroom_paper.ipynb. Outputs: results/paper/{figures,tables}/.

Figure style follows github.com/ChenLiu-1996/figures4papers (scientific-figure-making/references): Arial/Helvetica-like
sans fonts, top/right spines off, frameless legends, semantic palette (blues = our method, reds = baselines, neutral =
reference), black bar edges (no hatching), values written on the bars, dedicated legend panel, tight_layout(pad=2),
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
# Figures are drawn at their printed size: TW is the LNCS text width (12.2 cm), and the paper includes every figure at
# width=\textwidth, so font sizes below are the sizes in print (body text 10 pt, captions 9 pt; nothing below 6 pt).
TW = 4.8
FS_VAL = 6       # values written on bars
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.linewidth': 0.6, 'axes.spines.right': False, 'axes.spines.top': False,
    'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.pad': 2, 'ytick.major.pad': 2, 'axes.labelpad': 2, 'axes.titlepad': 4, 'lines.linewidth': 1.2,
    'hatch.linewidth': 0.5, 'legend.handlelength': 1.5, 'legend.handleheight': 0.8, 'legend.handletextpad': 0.5,
    'legend.columnspacing': 1.2, 'legend.labelspacing': 0.35, 'legend.borderaxespad': 0.2,
    'axes.labelcolor': PALETTE['ink'], 'xtick.color': PALETTE['ink'], 'ytick.color': PALETTE['ink'], 'text.color': PALETTE['ink'],
})
BAR = dict(edgecolor='black', linewidth=0.4)          # bar outline
ERR = dict(fmt='none', ecolor='black', capsize=1.2, capthick=0.5, linewidth=0.5)


def top_legend(fig, handles, labels, ncol, **kw):
    """Legend centred above all panels; returns the figure fraction left for the axes (pass to finalize as top)."""
    rows = math.ceil(len(labels) / ncol)
    fig.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.5, 1.0), **kw)
    return 1 - (rows * 0.135 + 0.06) / fig.get_figheight()


def finalize(fig, name, pad=0.3, top=1.0):
    fig.tight_layout(pad=pad, rect=(0, 0, 1, top))
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
    return FIG / f'{name}.png'


# --------------------------------------------------------------------------------------------- systems
# key -> (label, short label, group, answers file)
DIRECT = [('qwen7b', 'Qwen2.5-Coder-7B', '7B', 'answers_long/direct_qwen7b.jsonl'),
          ('qwen32b', 'Qwen2.5-Coder-32B', '32B', 'answers_long/direct_qwen32b.jsonl'),
          ('qwen27b', 'Qwen3.8-27B', '27B', 'answers_long/direct_qwen27b.jsonl'),
          ('llama70b', 'Llama-3.3-70B (4-bit)', '70B', 'answers_long/direct_llama70b.jsonl')]
FORMULA = [('prompted', 'Prompted, no training', 'answers/formula_qwen7b_prompted.jsonl'),
           ('sft', 'SFT (789 pairs)', 'answers/formula_sft789.jsonl'),
           ('rl_v1', 'SFT + RL, no validity gate', 'answers/formula_rl789_s42.jsonl')]
# FormaFin = the adopted Stage II reward (schedule- and condition-aware, multi-date; exec_sched, scripts/09_schedule_reward.sh).
# Keys keep the name rl_v2 so every table/figure picks it up; the earlier single-date reward runs are rl789_v2*.
RL_V2 = {s: f'answers/formula_rl789_sched_s{s}.jsonl' for s in (42, 43, 44)}
RL_V2_FORMULAS = {s: f'rl789_sched_s{s}' for s in (42, 43, 44)}
RL_MODEL_DIR = 'rl_sched_s{}'
FAMILY_FF = [('ff_llama', 'FormaFin: Llama-3.1-8B (3 seeds)', 'answers/formula_llama_rl_sched_s{}.jsonl'),
             ('ff_mistral', 'FormaFin: Mistral-7B (3 seeds)', 'answers/formula_mistral_rl_v2_s{}.jsonl')]
# Same recipe on two more model families (both with the adopted reward; Mistral's runs are named mistral_rl_v2).
FAMILIES = [('Qwen2.5-Coder-7B', 'qwen7b', 'qwen7b_prompted', 'sft789', 'rl789_sched_s{}'),
            ('Llama-3.1-8B', 'llama8b', 'llama8b_prompted', 'llama_sft789', 'llama_rl_sched_s{}'),
            ('Mistral-7B', 'mistral7b', 'mistral7b_prompted', 'mistral_sft789', 'mistral_rl_v2_s{}')]
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
    """Accuracy on one question, averaged like the overall score: first over the cases of each clause that ask it, then
    over those clauses (143 for the status, 94 for the cushion, 97 for the capacity)."""
    by = defaultdict(list)
    for k, mk in m.items():
        if q in mk and (keep is None or keep(cases()[k])):
            by[cases()[k]['clause_id']].append(mk[q])
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


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
        add(f'rl_v2_s{s}', f'FormaFin (seed {s})', '7B formalizer', rel, True)
    seeds = [out[f'rl_v2_s{s}'] for s in RL_V2]
    pc = {c: sum(s['pc'][c] for s in seeds) / len(seeds) for c in seeds[0]['pc']}
    out['rl_v2'] = dict(label='FormaFin: SFT + RL (ours, 3 seeds)', group='7B formalizer', ours=True, pc=pc, score=mean(pc),
                        ci=boot_ci(list(pc.values())), seed_scores=[s['score'] for s in seeds],
                        **{q: float(np.mean([s[q] for s in seeds])) for q, _ in QUESTIONS},
                        **{f'{q}_sd': float(np.std([s[q] for s in seeds], ddof=1)) for q, _ in QUESTIONS},
                        score_sd=float(np.std([s['score'] for s in seeds], ddof=1)))
    for key, label, fmt in FAMILY_FF:
        runs = []
        for sd in RL_V2:
            m = marks(fmt.format(sd))
            runs.append(dict(pc=per_clause(m), **{q: q_acc(m, q) for q, _ in QUESTIONS}))
        pcf = {c: sum(r['pc'][c] for r in runs) / len(runs) for c in runs[0]['pc']}
        out[key] = dict(label=label, group='formalizer', ours=True, pc=pcf, score=mean(pcf), ci=boot_ci(list(pcf.values())),
                        seed_scores=[mean(r['pc']) for r in runs], score_sd=float(np.std([mean(r['pc']) for r in runs], ddof=1)),
                        **{q: float(np.mean([r[q] for r in runs])) for q, _ in QUESTIONS},
                        **{f'{q}_sd': float(np.std([r[q] for r in runs], ddof=1)) for q, _ in QUESTIONS})
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
    ver = {json.loads(l)['clause_id']: json.loads(l)['result'] for l in open(RES / f'verification/{a_name}.jsonl')}
    proven = sum(ver.get(k) == 'equivalent' for k in agreed)
    return dict(n=len(res), agreed=len(agreed), coverage=len(agreed) / len(res), proven_agreed=proven,
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
           r'\caption{CovenantBench. Clauses come from agreements whose borrowers never appear in any training '
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
    """Table 1, two-level layout: Overall score | by question | proven correct | gain over the best reader. Readers by
    size; ablations on Qwen2.5-Coder-7B; FormaFin (ours, shaded) on three families with +- s.d. over three training runs
    in its own column so numbers stay aligned. 95% intervals are drawn in Fig. 3. Bold: best per column.
    LaTeX needs xcolor with the table option and a colour named oursbg."""
    S = system_stats()
    ver = json.loads((RES / 'verification/summary.json').read_text())
    groups = [('Open LLMs reading the clause', [('qwen7b', '7B'), ('qwen27b', '27B'), ('qwen32b', '32B'), ('llama70b', '70B')], False),
              ('Ablations on Qwen2.5-Coder-7B (formalize, then Z3)', [('prompted', '7B'), ('sft', '7B'), ('rl_v1', '7B')], False),
              ('FormaFin (ours), three training runs', [('rl_v2', '7B'), ('ff_llama', '8B'), ('ff_mistral', '7B')], True)]
    names = {'qwen7b': 'Qwen2.5-Coder-7B', 'qwen27b': 'Qwen3.8-27B', 'qwen32b': 'Qwen2.5-Coder-32B', 'llama70b': 'Llama-3.3-70B (4-bit)',
             'prompted': 'Prompted, no training', 'sft': 'Stage I only (SFT)', 'rl_v1': 'SFT + RL without validity gate',
             'rl_v2': 'Qwen2.5-Coder-7B', 'ff_llama': 'Llama-3.1-8B', 'ff_mistral': 'Mistral-7B'}
    ver_runs = {'prompted': ['qwen7b_prompted'], 'sft': ['sft789'], 'rl_v1': ['rl789_s42'],
                'rl_v2': [f'rl789_sched_s{s}' for s in RL_V2], 'ff_llama': [f'llama_rl_sched_s{s}' for s in RL_V2],
                'ff_mistral': [f'mistral_rl_v2_s{s}' for s in RL_V2]}
    keys = [k for _, ks, _ in groups for k, _ in ks]
    proven = {k: float(np.mean([100 * ver[r]['verified_rate'] for r in ver_runs[k]])) for k in ver_runs}
    best_reader = max((k for k, _ in groups[0][1]), key=lambda k: S[k]['score'])
    cols = ['score', 'status', 'cushion', 'capacity']
    best = {c: max(keys, key=lambda k: S[k][c]) for c in cols}
    best['proven'] = max(proven, key=proven.get)
    bold = lambda txt, b: r'\textbf{' + txt + '}' if b else txt
    lat = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
           r'\caption{Results on \bench{}. Score is the share of questions answered correctly, averaged first over the cases '
           r'of each covenant and then over covenants (95\% intervals in \cref{fig:main}). Status, Cushion and Capacity '
           r'are averaged the same way, over the 143, 94 and 97 covenants that ask each question. For FormaFin (shaded), $\pm$ is the standard deviation over three '
           r'training runs. Proven: share of formulas verified correct for every possible input (readers write no '
           r'formulas). Gain: score minus that of the best reader, ' + S[best_reader]['label'] + r', computed before rounding. Best results in bold.}',
           r'\label{tab:main}', r'\begin{adjustbox}{max width=\linewidth}',
           r'\begin{tabular}{lcrccccc}', r'\toprule',
           r' & & & \multicolumn{3}{c}{By question} & & \\',
           r'\cmidrule(lr){4-6}',
           r'System & Size & Score & Status & Cushion & Capacity & Proven & Gain \\', r'\midrule']
    md = ['| System | Size | Overall | Status | Cushion | Capacity | Proven correct | Gain over best reader |', '|---|---|---|---|---|---|---|---|']
    for gi, (title, ks, is_ours) in enumerate(groups):
        shade = r'\rowcolor{oursbg}' if is_ours else ''
        if gi:
            lat.append(r'\midrule')
        lat.append(shade + r'\multicolumn{8}{l}{\textsc{' + title + r'}} \\')
        md.append(f'| **{title}** | | | | | | | |')
        for k, size in ks:
            sd = r'{\scriptsize\,$\pm$' + pct(S[k]['score_sd'] if is_ours else 0.01) + '}'
            sd = sd if is_ours else r'\phantom{' + sd + '}'   # keeps every score aligned on the decimal point
            q = ' & '.join(bold(pct(S[k][c]), best[c] == k) for c in cols[1:])
            pv = bold(f'{proven[k]:.1f}', best['proven'] == k) if k in proven else r'\textcolor{gray}{--}'
            gain = f"{100 * (S[k]['score'] - S[best_reader]['score']):+.1f}" if is_ours else ''
            lat.append(shade + f"\\quad {names[k]} & {size} & {bold(pct(S[k]['score']), best['score'] == k)}{sd} & {q} & {pv} & {gain} \\\\")
            md.append(f"| {names[k]} | {size} | {pct(S[k]['score'])}{(' ± ' + pct(S[k]['score_sd'])) if is_ours else ''} | "
                      + ' | '.join(pct(S[k][c]) for c in cols[1:]) + f" | {pv} | {gain} |")
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table*}']
    return write('tab2_main', '\n'.join(lat), '\n'.join(md))


def table_paired():
    S = system_stats()
    comps = [('sft', 'SFT (same model before RL)'), ('qwen7b', 'Qwen2.5-Coder-7B, reading directly'),
             ('qwen32b', 'Qwen2.5-Coder-32B, reading directly'), ('qwen27b', 'Qwen3.8-27B, reading directly'),
             ('llama70b', 'Llama-3.3-70B (4-bit), reading directly')]
    res = [paired(S['rl_v2']['pc'], S[k]['pc']) for k, _ in comps]
    ph = holm([r['p'] for r in res])
    seed_diffs = {k: [paired(S[f'rl_v2_s{s}']['pc'], S[k]['pc'])['diff'] for s in RL_V2] for k, _ in comps}
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
           r'\caption{FormaFin (three-seed mean) minus each comparator: mean per-clause difference in points, 95\% '
           r'clause-bootstrap interval, two-sided sign-flip test over the 143 clauses (10,000 permutations), Holm-corrected '
           r'over the five comparisons. Seeds: the difference for each of the three RL seeds. Lower block: a separate Holm '
           r'family of four comparisons.}',
           r'\label{tab:paired}', r'\begin{tabular}{@{}lrcrr@{}}', r'\toprule',
           r'\textbf{FormaFin minus} & \textbf{$\Delta$} & \textbf{95\% CI} & \textbf{$p_{\mathrm{Holm}}$} & \textbf{Seeds} \\', r'\midrule']
    md = ['| FormaFin minus | Δ (points) | 95% CI | p (Holm) | Per seed |', '|---|---|---|---|---|']
    for (k, lab), r, p in zip(comps, res, ph):
        sd = '/'.join(f'{100 * x:+.1f}' for x in seed_diffs[k])
        lat.append(f"{tex(lab)} & {100 * r['diff']:+.1f} & [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] & {fmt_p(p)} & "
                   r'{\scriptsize ' + sd + r'} \\')
        md.append(f"| {lab} | {100 * r['diff']:+.1f} | [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] | {p:.4f} | {sd} |")
    stages = [('sft', 'prompted', 'SFT minus prompted formalizer'), ('rl_v1', 'sft', 'RL without gate minus SFT'),
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
    rows = [('prompted', 'Prompted, no training'), ('sft', 'SFT (789 pairs)'), ('rl_v1', 'SFT + RL, no gate')] + \
           [(f'rl_v2_s{s}', f'FormaFin, seed {s}') for s in RL_V2]
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


CROSS = [('rl789_sched_s42', 'rl789_sched_s43', 'answers/formula_rl789_sched_s42.jsonl', 'sched42_sched43', 'FormaFin seed 42 $\\times$ seed 43', 'FormaFin seed 42 × seed 43'),
         ('rl789_sched_s42', 'sft789', 'answers/formula_rl789_sched_s42.jsonl', 'sched42_sft', 'FormaFin seed 42 $\\times$ SFT', 'FormaFin seed 42 × SFT'),
         ('sft789', 'qwen7b_prompted', 'answers/formula_sft789.jsonl', 'sft_prompted', 'SFT $\\times$ prompted 7B', 'SFT × prompted 7B'),
         ('rl789_sched_s42', 'llama_rl_sched_s42', 'answers/formula_rl789_sched_s42.jsonl', 'sched42_llama42', 'FormaFin Qwen $\\times$ Llama', 'FormaFin Qwen × Llama'),
         ('rl789_sched_s42', 'mistral_rl_v2_s42', 'answers/formula_rl789_sched_s42.jsonl', 'sched42_mistral42', 'FormaFin Qwen $\\times$ Mistral', 'FormaFin Qwen × Mistral')]


def table_selfcheck():
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{5pt}',
           r'\caption{Label-free self-check. Two formalizers write a formula for the same clause and Z3 checks whether '
           r'the two are equivalent for all inputs; no reference reading is used. Agreed clauses could be checked '
           r"automatically every quarter; the rest go to an analyst. Score: the first formalizer's "
           r'case-level score on each group. Proven: agreed clauses on which Z3 also proves the first formalizer equivalent '
           r'to the reference reading. The three FormaFin seeds share one Stage I adapter; the cross-family pairs (seed 42 '
           r'of each family) share only the teacher data.}',
           r'\label{tab:selfcheck}', r'\begin{tabular}{@{}lcccc@{}}', r'\toprule',
           r'\textbf{Pair of formalizers} & \textbf{Agree (coverage)} & \textbf{Proven} & \textbf{Score, agreed} & \textbf{Score, to analyst} \\', r'\midrule']
    md = ['| Pair | Agree (coverage) | Proven among agreed | Score, agreed | Score, to analyst |', '|---|---|---|---|---|']
    out = []
    for a, b, rel, cache, lab_t, lab_m in CROSS:
        r = cross_check(a, b, rel, cache)
        out.append((lab_m, r))
        lat.append(f"{lab_t} & {r['agreed']} / {r['n']} ({pct(r['coverage'], 0)}\\%) & {r['proven_agreed']} / {r['agreed']} & {pct(r['acc_agreed'])} & {pct(r['acc_other'])} \\\\")
        md.append(f"| {lab_m} | {r['agreed']} / {r['n']} ({pct(r['coverage'], 0)}%) | {r['proven_agreed']} / {r['agreed']} | {pct(r['acc_agreed'])} | {pct(r['acc_other'])} |")
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
    for k, lab in (('sft', 'SFT'), ('rl_v2', 'FormaFin (3 seeds)')):
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
           r'pre-registered one) is used throughout the paper; the lenient rule only diagnoses the run without the gate. FormaFin rows: '
           r'mean $\pm$ s.d. over three seeds, trained with the schedule- and condition-aware reward.}',
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
# figures4papers semantics: baselines (open readers) in neutral grays, darker = larger model; our method in blues, the
# other families in teal/green; SFT = the same blue at lower strength (alpha-style ablation, pre-blended so it prints
# opaque); red is reserved for contrasts (counterexamples, wrongly flagged answers). No hatching: hue + black edges suffice.
COLORS = {'qwen7b': PALETTE['neutral'], 'qwen27b': '#9E9D9D', 'qwen32b': PALETTE['gray_mid'], 'llama70b': PALETTE['gray_dark'],
          'ff_llama': PALETTE['teal'], 'ff_mistral': PALETTE['green_3'],
          'prompted': PALETTE['green_1'], 'sft': '#9BBADC', 'rl_v1': PALETTE['green_2'], 'rl_v2': PALETTE['blue_main']}
HATCH = {}
OURS_SD = ('rl_v2', 'ff_llama', 'ff_mistral')  # systems drawn with +-1 s.d. over three seeds


def fig_main():
    """Fig. 1: one panel, grouped by question (overall, status, cushion, capacity); in each group the open readers
    (neutral grays, darker = larger) next to our formalizers (SFT and FormaFin on three families, blues/greens). Bars equal Table 1
    (overall: mean per-clause score; questions: share of cases); every bar has a 95% interval from resampling clauses
    (FormaFin: pooled over its three seeds), so all systems carry the same kind of uncertainty."""
    S = system_stats()
    keys = ['qwen7b', 'qwen27b', 'qwen32b', 'llama70b', 'sft', 'rl_v2', 'ff_llama', 'ff_mistral']  # readers by size
    labels = {'qwen7b': 'Qwen2.5-Coder-7B, reads directly', 'qwen32b': 'Qwen2.5-Coder-32B, reads directly',
              'qwen27b': 'Qwen3.8-27B, reads directly', 'llama70b': 'Llama-3.3-70B (4-bit), reads directly',
              'sft': 'Ours: SFT, Qwen2.5-Coder-7B', 'rl_v2': 'Ours: FormaFin, Qwen2.5-Coder-7B',
              'ff_llama': 'Ours: FormaFin, Llama-3.1-8B', 'ff_mistral': 'Ours: FormaFin, Mistral-7B'}
    metrics = [('score', 'Overall (per clause)'), ('status', 'Status'), ('cushion', 'Earnings cushion'), ('capacity', 'Debt capacity')]
    fig, ax = plt.subplots(figsize=(TW, 2.75))
    n = len(keys)
    w = 0.86 / n
    gx = np.arange(len(metrics))
    def counts(rel, q):  # per clause: (score sum, n) -- per-clause mean for 'score' and for each question
        by = defaultdict(lambda: [0.0, 0])
        for cid, mk in marks(rel).items():
            c = cases()[cid]['clause_id']
            if q == 'score':
                by[c][0] += sum(mk.values()) / len(mk); by[c][1] += 1
            elif q in mk:
                by[c][0] += mk[q]; by[c][1] += 1
        return by
    rng = np.random.default_rng(0)
    files = dict([(k, rel) for k, _, _, rel in DIRECT] + [(k, rel) for k, _, rel in FORMULA])
    seeds = {'rl_v2': list(RL_V2.values()), **{k: [fmt.format(sd) for sd in RL_V2] for k, _, fmt in FAMILY_FF}}
    for j, k in enumerate(keys):
        xs = gx + (j - (n - 1) / 2) * w
        vals, lo, hi = [], [], []
        for m, _ in metrics:
            runs = [counts(r, m) for r in (seeds[k] if k in seeds else [files[k]])]
            cl = sorted(runs[0])
            if m == 'score':   # Table 1 score: mean over clauses of the per-clause mean (seed-mean for FormaFin)
                per = np.array([np.mean([r[c][0] / r[c][1] for r in runs]) for c in cl])
                est = per.mean()
                boot = per[rng.integers(0, len(cl), (10_000, len(cl)))].mean(1)
            else:              # Table 1 columns: per-clause accuracy on this question, averaged over clauses (seed-mean)
                per = np.array([np.mean([r[c][0] / r[c][1] for r in runs]) for c in cl])
                est = per.mean()
                boot = per[rng.integers(0, len(cl), (10_000, len(cl)))].mean(1)
            v = 100 * est
            vals.append(v); lo.append(v - 100 * np.percentile(boot, 2.5)); hi.append(100 * np.percentile(boot, 97.5) - v)
        ax.bar(xs, vals, width=w, color=COLORS[k], hatch=HATCH.get(k), label=labels[k], zorder=2, **BAR)
        ax.errorbar(xs, vals, yerr=[lo, hi], zorder=3, **ERR)
        for xi, v, h in zip(xs, vals, hi):
            ax.text(xi, v + h + 1.0, f'{v:.1f}', ha='center', va='bottom', fontsize=FS_VAL, rotation=90,
                    fontweight='bold' if k in OURS_SD else 'normal')
    for g in gx[:-1]:
        ax.axvline(g + 0.5, color=PALETTE['neutral'], linewidth=0.6, zorder=1)
    ax.set_xticks(gx)
    ax.set_xticklabels([t for _, t in metrics])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(0, 117)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('Correct (%)')
    h, l = ax.get_legend_handles_labels()       # 2 columns: readers left, ours right
    top = top_legend(fig, h, l, ncol=2)
    return finalize(fig, 'fig1_main', top=top)


def fig_size():
    """Fig. 2: accuracy against model size; direct readers (line) vs our 7-8B formalizers."""
    S = system_stats()
    fig, ax = plt.subplots(figsize=(0.6 * TW, 2.35))   # included at 0.6\textwidth (main_long)
    xs = {'qwen7b': 7, 'qwen27b': 27, 'qwen32b': 32, 'llama70b': 70}
    pts = sorted(xs, key=lambda k: xs[k])
    ax.plot([xs[k] for k in pts], [100 * S[k]['score'] for k in pts], color=PALETTE['gray_mid'], linewidth=0.9,
            linestyle=(0, (4, 3)), marker='o', markersize=4.5, markerfacecolor=PALETTE['neutral'], markeredgecolor='black', markeredgewidth=0.6,
            label='Open models reading directly', zorder=3)
    offs = {'qwen7b': (5, -9), 'qwen27b': (-24, 6), 'qwen32b': (4, 4), 'llama70b': (-62, -10)}
    names = {'qwen7b': 'Qwen-Coder-7B', 'qwen27b': 'Qwen3.8-27B', 'qwen32b': 'Qwen-Coder-32B', 'llama70b': 'Llama-3.3-70B'}
    for k in pts:
        lo, hi = S[k]['ci']
        ax.vlines(xs[k], 100 * lo, 100 * hi, color=PALETTE['gray_mid'], linewidth=0.8, alpha=0.7, zorder=2)
        ax.annotate(names[k], (xs[k], 100 * S[k]['score']), xytext=offs[k], textcoords='offset points', fontsize=6.5)
    ours = (('sft', 's', 6.2, 'SFT, Qwen-7B'), ('ff_mistral', 'D', 7.0, 'FormaFin, Mistral-7B'),
            ('rl_v2', '*', 7.8, 'FormaFin, Qwen-7B'), ('ff_llama', '^', 8.8, 'FormaFin, Llama-8B'))
    for k, mk, xo, lab in ours:
        lo, hi = S[k]['ci']
        ax.vlines(xo, 100 * lo, 100 * hi, color=COLORS[k], linewidth=1.0, alpha=0.8, zorder=4)
        ax.plot(xo, 100 * S[k]['score'], marker=mk, markersize=7 if mk == '*' else 4.5, color=COLORS[k],
                markeredgecolor='black', markeredgewidth=0.6, linestyle='', zorder=5)
        ax.plot([], [], marker=mk, markersize=6 if mk == '*' else 4, color=COLORS[k], markeredgecolor='black',
                markeredgewidth=0.5, linestyle='', label=f"Ours: {lab} ({100 * S[k]['score']:.1f})")
    ax.set_xlabel('Parameters of the answering model (B)')
    ax.set_ylabel('Correct per clause (%)')
    ax.set_xlim(0, 76)
    ax.set_xticks([7, 27, 32, 70])
    ax.set_ylim(55, 102)
    h, l = ax.get_legend_handles_labels()
    order = [l.index(x) for x in sorted(l, key=lambda t: (not t.startswith('Open'), t))]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc='lower right', fontsize=6.5, handlelength=1.2)
    return finalize(fig, 'fig2_size')


def _rolling(y, w):
    y = np.asarray(y, float)
    return np.convolve(y, np.ones(w) / w, mode='valid') if len(y) >= w else y


_VIR = plt.get_cmap('viridis')   # one colour per model family, shared by the training figures (Fig. 4 and Fig. 5)
TRAIN_FAMILIES = [('Qwen2.5-Coder-7B', 'sft789', RL_MODEL_DIR, _VIR(0.02), '-'),
                  ('Llama-3.1-8B', 'llama_sft789', 'llama_rl_sched_s{}', _VIR(0.45), (0, (5, 2))),
                  ('Mistral-7B', 'mistral_sft789', 'mistral_rl_v2_s{}', _VIR(0.74), (0, (1.5, 1.5)))]


def fig_training():
    """Fig. 3: learning curves for all three model families (adopted reward). (a) SFT loss; GRPO, three seeds each:
    (b) reward of the sampled formulas, (c) share of steps whose four samples differ in reward (the only steps with a
    gradient), (d) share of samples keeping the interface. Lines: mean over seeds; bands: range over seeds."""
    fig, axes = plt.subplots(1, 3, figsize=(TW, 1.95))
    panels = [('signal', '(b) GRPO: learning signal', 'Steps whose samples differ (%)'),
              ('sig', '(c) GRPO: interface kept', 'Samples, exact parameters (%)')]   # the reward is in Fig. 5b
    for fam, sft_dir, rl_dir, col, ls in TRAIN_FAMILIES:
        sft = [h for h in json.loads((P.MODELS / sft_dir / 'trainer_state.json').read_text())['log_history'] if 'loss' in h]
        st = [h['step'] for h in sft]
        loss = [h['loss'] for h in sft]
        axes[0].plot(st, loss, color=col, linewidth=0.4, alpha=0.3)
        axes[0].plot(st[len(st) - len(_rolling(loss, 10)):], _rolling(loss, 10), color=col, linewidth=1.2, linestyle=ls, label=fam)
        curves = defaultdict(list)
        for sd in (42, 43, 44):
            rows = [json.loads(l) for l in open(P.MODELS / rl_dir.format(sd) / 'exec_rewards.jsonl')]
            per_step = np.array([r['reward'] for r in rows]).reshape(-1, 4)
            ok = np.array([float(r['valid_signature']) for r in rows]).reshape(-1, 4)
            curves['signal'].append(_rolling((per_step.std(1) > 0).astype(float) * 100, 50))
            curves['sig'].append(_rolling(ok.mean(1) * 100, 50))
        for ax, (key, _, _) in zip(axes[1:], panels):
            n = min(len(c) for c in curves[key])
            arr = np.array([c[:n] for c in curves[key]])
            steps = np.arange(n) + 50
            ax.fill_between(steps, arr.min(0), arr.max(0), color=col, alpha=0.18, linewidth=0)
            ax.plot(steps, arr.mean(0), color=col, linewidth=1.2, linestyle=ls, label=fam)
    ax = axes[0]
    ax.set_title('(a) SFT: training loss', loc='left')
    ax.set_xlabel('SFT step (1 epoch)')
    ax.set_ylabel('Loss (10-step mean)')
    ax.set_ylim(0, None)
    for ax, (key, title, ylab) in zip(axes[1:], panels):
        ax.set_title(title, loc='left')
        ax.set_xlabel('GRPO step (50-step mean)')
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 1000)
        ax.set_xticks([0, 500, 1000])
    h, l = axes[0].get_legend_handles_labels()   # mean and range over seeds are explained in the caption
    top = top_legend(fig, h, l, ncol=3, handlelength=2.6)
    return finalize(fig, 'fig3_training', top=top)


def fig_verification():
    """Fig. 4: (a) formal verification of each formalizer's formulas; (b) label-free self-check: coverage and score."""
    ver = verification()
    S = system_stats()
    rows = [('prompted', 'Prompted 7B'), ('sft', 'SFT'), ('rl_v1', 'RL, no gate')] + [(f'rl_v2_s{s}', f'FormaFin ({s})') for s in RL_V2]
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.3), gridspec_kw=dict(width_ratios=[1.1, 1.2]))
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    segs = [('equivalent', 'Proven equivalent', PALETTE['blue_main'], None), ('different', 'Counterexample', PALETTE['red_2'], None),
            ('invalid', 'Unusable', PALETTE['neutral'], None)]
    left = np.zeros(len(rows))
    for key, lab, col, h in segs:
        vals = np.array([ver[k][key] for k, _ in rows])
        ax.barh(y, vals, left=left, color=col, hatch=h, height=0.7, label=lab, **BAR)
        for yi, l, v in zip(y, left, vals):
            if v >= 10:
                ax.text(l + v / 2, yi, str(int(v)), ha='center', va='center', fontsize=FS_VAL + 0.5,
                        color='white' if key == 'equivalent' else PALETTE['ink'])
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in rows])
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('Clauses (of 143)')
    ax.set_xlim(0, 143)
    ax.set_xticks([0, 25, 50, 75, 100, 125])
    ax.set_title('(a) Formal verification vs. reference', loc='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2), ncol=3, columnspacing=0.9)
    ax2 = axes[1]
    cross = [(lab, cross_check(a, b, rel, cache)) for a, b, rel, cache, _, lab in CROSS]
    short = {'FormaFin seed 42 × seed 43': 'seeds\n42×43', 'FormaFin seed 42 × SFT': 'seed 42\n×SFT',
             'SFT × prompted 7B': 'SFT×\nprompt', 'FormaFin Qwen × Llama': 'Qwen×\nLlama',
             'FormaFin Qwen × Mistral': 'Qwen×\nMistral'}
    x = np.arange(len(cross))
    w = 0.38
    b1 = ax2.bar(x - w / 2, [100 * r['acc_agreed'] for _, r in cross], width=w, color=PALETTE['blue_main'],
                 label='Agree: automatic', **BAR)
    b2 = ax2.bar(x + w / 2, [100 * r['acc_other'] for _, r in cross], width=w, color=PALETTE['red_2'],
                 label='Disagree: to an analyst', **BAR)
    for bars in (b1, b2):
        for b in bars:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f'{b.get_height():.1f}', ha='center', va='bottom',
                     fontsize=FS_VAL, rotation=90)
    for xi, (_, r) in zip(x, cross):
        ax2.text(xi - w / 2, 4, f"{100 * r['coverage']:.0f}%", ha='center', va='bottom', fontsize=FS_VAL,
                 color='white', fontweight='bold', rotation=90)
    ax2.set_xticks(x)
    ax2.set_xticklabels([short.get(l, l) for l, _ in cross])
    ax2.tick_params(axis='x', length=0, labelsize=FS_VAL + 0.5)
    ax2.set_ylabel('Correct per clause (%)')
    ax2.set_ylim(0, 118)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.legend(loc='upper center', ncol=1, bbox_to_anchor=(0.5, -0.24))
    ax2.set_title('(b) Two formalizers agree?', loc='left')
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
            ('qwen32b', 'answers_long/direct_qwen32b.jsonl'), ('llama70b', 'answers_long/direct_llama70b.jsonl'),
            ('sft', 'answers/formula_sft789.jsonl'), ('rl_v2', None)]
    labels = {'qwen7b': 'Reads: Qwen2.5-Coder-7B', 'qwen32b': 'Reads: Qwen2.5-Coder-32B',
              'qwen27b': 'Reads: Qwen3.8-27B', 'llama70b': 'Reads: Llama-3.3-70B (4-bit)',
              'sft': 'Ours: SFT (Qwen-Coder-7B)', 'rl_v2': 'Ours: FormaFin (Qwen-Coder-7B)'}

    def acc(rel, keep):
        if rel is None:
            return float(np.mean([mean(per_clause(marks(r), keep)) for r in RL_V2.values()]))
        return mean(per_clause(marks(rel), keep))
    fig, ax = plt.subplots(figsize=(TW, 2.55))
    x = np.arange(len(groups))
    w = 0.84 / len(keys)
    for j, (k, rel) in enumerate(keys):
        vals = [100 * acc(rel, keep) for _, keep in groups]
        bars = ax.bar(x + (j - (len(keys) - 1) / 2) * w, vals, width=w, color=COLORS[k], hatch=HATCH.get(k), label=labels[k], **BAR)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f'{b.get_height():.0f}', ha='center', va='bottom',
                    fontsize=FS_VAL, rotation=90, fontweight='bold' if k == 'rl_v2' else 'normal')
    ax.set_xticks(x)
    ax.set_xticklabels([g for g, _ in groups])
    ax.tick_params(axis='x', length=0)
    ax.axvline(3.5, color=PALETTE['gray_mid'], linewidth=0.6, linestyle=(0, (4, 3)))
    ax.text(1.5, 113, 'by case type', ha='center', va='center', fontsize=7, color=PALETTE['gray_dark'], style='italic')
    ax.text(4.5, 113, 'by clause type', ha='center', va='center', fontsize=7, color=PALETTE['gray_dark'], style='italic')
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel('Correct per clause (%)')
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    h, l = ax.get_legend_handles_labels()       # 3 columns: two reader columns, then ours
    top = top_legend(fig, h, l, ncol=3, columnspacing=1.0)
    return finalize(fig, 'fig5_cases', top=top)





CKPT_FAMILIES = [('Qwen2.5-Coder-7B', 'qwen', 'answers/formula_sft789.jsonl', 'rl_v2', 'answers/formula_rl789_sched_s{}.jsonl'),
                 ('Llama-3.1-8B', 'llama', 'answers/formula_llama_sft789.jsonl', 'ff_llama', 'answers/formula_llama_rl_sched_s{}.jsonl'),
                 ('Mistral-7B', 'mistral', 'answers/formula_mistral_sft789.jsonl', 'ff_mistral', 'answers/formula_mistral_rl_v2_s{}.jsonl')]


def fig_rl_progress():
    """Fig.: benchmark score along Stage II. Step 0 = the SFT model, 250/500/750 = GRPO checkpoints
    (scripts/13_checkpoint_eval.sh), 1000 = final; one viridis curve per model family (mean over three seeds, light band =
    range); gray dashed/dotted: open readers. Open frame, faint horizontal grid, legend below."""
    S = system_stats()
    steps = [0, 250, 500, 750, 1000]
    vir = plt.get_cmap('viridis')
    cols = {'qwen': vir(0.02), 'llama': vir(0.45), 'mistral': vir(0.74)}
    fig, ax = plt.subplots(figsize=(TW, 2.4))
    ax.tick_params(color=PALETTE['gray_dark'])
    ax.yaxis.grid(True, color='#ECECEC', linewidth=0.5)
    ax.set_axisbelow(True)
    for k, lab, dash in (('qwen27b', 'Best open reader (Qwen3.8-27B)', (0, (5, 2.5))), ('llama70b', 'Llama-3.3-70B reader (4-bit)', (0, (1, 1.8)))):
        ax.axhline(100 * S[k]['score'], color=PALETTE['gray_mid'], linewidth=0.9, linestyle=dash, zorder=1, label=lab)
    out = {}
    for name, short, sft, ff, final in CKPT_FAMILIES:
        col = cols[short]
        per_seed = []
        for sd in RL_V2:
            rels = [sft] + [f'answers/formula_ckpt_{short}_s{sd}_{st}.jsonl' for st in steps[1:-1]] + [final.format(sd)]
            per_seed.append([100 * mean(per_clause(marks(r))) for r in rels])
        arr = np.array(per_seed)
        m = arr.mean(0)
        out[name] = dict(steps=steps, mean=m.tolist(), seeds=arr.tolist())
        ax.fill_between(steps, arr.min(0), arr.max(0), color=col, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(steps, m, color=col, linewidth=1.2, marker='o', markersize=3.2, markeredgecolor='white', markeredgewidth=0.5,
                zorder=3, label=f'FormaFin, {name}')
    (OUT / 'rl_progress.json').write_text(json.dumps(out, indent=1))
    ax.set_xlabel('GRPO training step (0 = SFT model)')
    ax.set_ylabel('Correct per clause (%)')
    ax.set_xticks(steps)
    ax.set_xlim(-25, 1025)
    ax.set_ylim(30, 100)
    ax.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
    h, l = ax.get_legend_handles_labels()
    order = [2, 3, 4, 0, 1]   # our three models, then the two readers
    # the lower right of the plot is empty (all curves are above 88% after step 250): the legend goes there
    ax.legend([h[i] for i in order], [l[i] for i in order], loc='lower right', bbox_to_anchor=(1.0, 0.03), ncol=1,
              frameon=False, handlelength=2.2)
    return finalize(fig, 'fig7_rl_progress')


def fig_reward_vs_score(window=100):
    """Fig.: does the label-free training reward track benchmark accuracy? Bar version: one panel per model family; at
    each checkpoint (SFT model, steps 250/500/750/1000) a light bar = mean GRPO training reward over the `window` steps
    before it (x100; the first `window` steps for the SFT model) and a solid bar = benchmark score; error bars = s.d.
    over three seeds. Family colours as in fig7_rl_progress. Writes the correlations to reward_vs_score.json."""
    from scipy.stats import pearsonr, spearmanr
    import matplotlib.colors as mcolors
    path = OUT / 'rl_progress.json'
    if not path.exists():
        fig_rl_progress()
    prog = json.loads(path.read_text())
    dirs = {'Qwen2.5-Coder-7B': (RL_MODEL_DIR, 'qwen'), 'Llama-3.1-8B': ('llama_rl_sched_s{}', 'llama'),
            'Mistral-7B': ('mistral_rl_v2_s{}', 'mistral')}
    steps = [0, 250, 500, 750, 1000]
    ticks = ['SFT', '250', '500', '750', '1000']
    vir = plt.get_cmap('viridis')
    cols = {'qwen': vir(0.02), 'llama': vir(0.45), 'mistral': vir(0.74)}
    light = lambda c: tuple(1 - 0.55 * (1 - np.array(mcolors.to_rgb(c))))   # same hue, lighter (pre-blended, prints opaque)
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.2), sharey=True)
    allx, ally, rl_x, rl_y = [], [], [], []
    x = np.arange(len(steps)); w = 0.38
    for ax, (fam, (d, short)) in zip(axes, dirs.items()):
        col = cols[short]
        xs = np.zeros((3, len(steps))); ys = np.array(prog[fam]['seeds'])
        for si, sd in enumerate(RL_V2):
            r = np.array([json.loads(l)['reward'] for l in open(P.MODELS / d.format(sd) / 'exec_rewards.jsonl')]).reshape(-1, 4).mean(1)
            xs[si] = [r[:window].mean() if st == 0 else r[st - window:st].mean() for st in steps]
        allx += list(xs.ravel()); ally += list(ys.ravel())
        rl_x += list(xs[:, 1:].ravel()); rl_y += list(ys[:, 1:].ravel())
        rw, sc = 100 * xs.mean(0), ys.mean(0)
        b1 = ax.bar(x - w / 2, rw, w, color=light(col), label='Training reward (x100)', **BAR, zorder=2)
        b2 = ax.bar(x + w / 2, sc, w, color=col, label='Benchmark score (%)', **BAR, zorder=2)
        ax.errorbar(x - w / 2, rw, yerr=100 * xs.std(0, ddof=1), zorder=3, **ERR)
        ax.errorbar(x + w / 2, sc, yerr=ys.std(0, ddof=1), zorder=3, **ERR)
        for bars, err in ((b1, 100 * xs.std(0, ddof=1)), (b2, ys.std(0, ddof=1))):
            for b, e in zip(bars, err):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e + 1.5, f'{b.get_height():.0f}', ha='center', va='bottom', fontsize=FS_VAL, rotation=90)
        ax.set_title(fam, color=col)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=6.5)
        ax.set_xlabel('Checkpoint (GRPO step)')
        ax.yaxis.grid(True, color='#ECECEC', linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Reward (x100) / correct (%)')
    axes[0].set_ylim(0, 118)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    stats = dict(window=window, all=dict(n=len(allx), pearson=pearsonr(allx, ally)[0], spearman=spearmanr(allx, ally)[0]),
                 rl_only=dict(n=len(rl_x), pearson=pearsonr(rl_x, rl_y)[0], p=pearsonr(rl_x, rl_y)[1], spearman=spearmanr(rl_x, rl_y)[0]))
    (OUT / 'reward_vs_score.json').write_text(json.dumps(stats, indent=1, default=float))
    handles = [Patch(facecolor=light(PALETTE['gray_dark']), label='Mean training reward (x100), 100 steps before', **BAR),
               Patch(facecolor=PALETTE['gray_dark'], label='Benchmark score (% correct per clause)', **BAR)]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'fig8_reward_vs_score.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
    return FIG / 'fig8_reward_vs_score.png'


def fig_rl_combined(window=100):
    """Fig. 5, two aligned line panels over GRPO steps, same colours and style. (a) benchmark score (test) at the SFT model
    and checkpoints 250/500/750/1000, mean over three seeds with range band, dashed gray = open readers; (b) training
    reward (train), logged at every step, 50-step moving average, mean over seeds with range band, markers at the
    checkpoints. The reward-score correlations are written to reward_vs_score.json (fig_reward_vs_score)."""
    fig_rl_progress()                     # refreshes rl_progress.json
    fig_reward_vs_score(window)           # refreshes reward_vs_score.json
    S = system_stats()
    prog = json.loads((OUT / 'rl_progress.json').read_text())
    steps = [0, 250, 500, 750, 1000]
    vir = plt.get_cmap('viridis')
    cols = {'qwen': vir(0.02), 'llama': vir(0.45), 'mistral': vir(0.74)}
    fams = [('Qwen2.5-Coder-7B', 'qwen', RL_MODEL_DIR), ('Llama-3.1-8B', 'llama', 'llama_rl_sched_s{}'),
            ('Mistral-7B', 'mistral', 'mistral_rl_v2_s{}')]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(TW, 2.25))
    for a in (ax, bx):
        a.tick_params(color=PALETTE['gray_dark'])
        a.yaxis.grid(True, color='#ECECEC', linewidth=0.5); a.set_axisbelow(True)
        a.set_xticks(steps); a.set_xlim(-25, 1025)
        a.set_xlabel('GRPO step (0 = SFT model)')
    # (a) benchmark score
    for k, lab, dash in (('qwen27b', 'Best open reader (Qwen3.8-27B)', (0, (5, 2.5))), ('llama70b', 'Llama-3.3-70B reader (4-bit)', (0, (1, 1.8)))):
        ax.axhline(100 * S[k]['score'], color=PALETTE['gray_mid'], linewidth=0.9, linestyle=dash, zorder=1, label=lab)
    for fam, short, _ in fams:
        arr = np.array(prog[fam]['seeds']); m = arr.mean(0); col = cols[short]
        ax.fill_between(steps, arr.min(0), arr.max(0), color=col, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(steps, m, color=col, linewidth=1.2, marker='o', markersize=3.2, markeredgecolor='white', markeredgewidth=0.5,
                zorder=3, label=f'FormaFin, {fam}')
    ax.set_ylabel('Correct per clause (%)')
    ax.set_ylim(30, 100); ax.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
    h, l = ax.get_legend_handles_labels(); order = [2, 3, 4, 0, 1]
    top = top_legend(fig, [h[i] for i in order], [l[i] for i in order], ncol=2, handlelength=2.2)   # ours | readers
    ax.set_title('(a) Benchmark score (test)', loc='left')
    # (b) training reward at every step, `window`-step moving average; markers = the same `window`-step means that the
    # correlation in the text uses (step 0 = first `window` steps, i.e. the SFT model's own samples)
    for fam, short, d in fams:
        col = cols[short]
        curves, marks_ = [], []
        for sd in RL_V2:
            r = np.array([json.loads(l_)['reward'] for l_ in open(P.MODELS / d.format(sd) / 'exec_rewards.jsonl')]).reshape(-1, 4).mean(1)
            cs = np.concatenate([[0.0], np.cumsum(r)])        # trailing mean over up to `window` steps: starts at step 10
            curves.append(np.array([(cs[t] - cs[max(0, t - window)]) / min(t, window) for t in range(10, len(r) + 1)]))
            marks_.append([r[:window].mean() if st == 0 else r[st - window:st].mean() for st in steps])
        n = min(len(c) for c in curves)
        arr = np.array([c[:n] for c in curves]); m = arr.mean(0)
        xs = np.arange(n) + 10
        bx.fill_between(xs, arr.min(0), arr.max(0), color=col, alpha=0.12, linewidth=0, zorder=2)
        bx.plot(xs, m, color=col, linewidth=0.9, alpha=0.9, zorder=3)
        bx.plot(steps[1:], np.mean(marks_, 0)[1:], linestyle='', marker='o', markersize=3.2, color=col,
                markeredgecolor='white', markeredgewidth=0.5, zorder=4)
    bx.set_ylabel(f'Reward ({window}-step mean)')
    bx.set_ylim(0.5, 1.0); bx.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bx.set_title('(b) Training reward (no labels)', loc='left')
    return finalize(fig, 'fig7_rl_combined', top=top)


def fig_llm_verification():
    """Fig. 6: FormaFin formulas as verifiers of LLM analyses (bar charts; gray = LLM alone, blue = ours, green = caught, red = flagged). (a) accuracy of each LLM
    alone vs of the answers the verifier accepts; (b) share of the LLM's errors caught and of its correct answers
    flagged (all clauses / clauses where two formalizers agree)."""
    path = OUT / 'llm_verification.json'
    if not path.exists():
        table_llm_verification()
    V = json.loads(path.read_text())
    names = list(V)
    two_line = {'Qwen2.5-Coder-7B': 'Qwen2.5-\nCoder-7B', 'Qwen2.5-Coder-32B': 'Qwen2.5-\nCoder-32B',
                'Llama-3.3-70B (4-bit)': 'Llama-3.3-70B\n(4-bit)', 'Claude Sonnet 5.5': 'Claude Sonnet\n5.5 (propr.)'}
    short = []
    for n in names:
        n = n.replace('$^\\dagger$', '')
        short.append(two_line.get(n, n.replace('-', '-\n', 1) if len(n) > 11 else n))
    # stacked panels sharing the x axis, so that every bar has room for its value at print size
    fig, axes = plt.subplots(2, 1, figsize=(TW, 3.55), sharex=True)
    x = np.arange(len(names))
    w = 0.38
    ax = axes[0]
    a1 = [100 * V[n]['all']['llm_acc'] for n in names]
    a2 = [100 * V[n]['all']['acc_accepted'] for n in names]
    b1 = ax.bar(x - w / 2, a1, w, color=PALETTE['neutral'], label='LLM analysis alone', **BAR)
    b2 = ax.bar(x + w / 2, a2, w, color=PALETTE['blue_main'], label='Answers accepted by FormaFin', **BAR)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f'{b.get_height():.1f}', ha='center', va='bottom',
                    fontsize=FS_VAL, rotation=90)
    ax.set_title('(a) Accuracy of the answers (%)', loc='left')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.98), ncol=2)
    w = 0.205
    series = [('all', 'caught', PALETTE['green_3'], 'Errors caught, all clauses'),
              ('agreed', 'caught', PALETTE['green_2'], 'Errors caught, agreed clauses'),
              ('all', 'false_flag', PALETTE['red_strong'], 'Correct flagged, all clauses'),
              ('agreed', 'false_flag', PALETTE['red_2'], 'Correct flagged, agreed clauses')]
    bx = axes[1]
    for j, (grp, key, col, lab) in enumerate(series):
        vals = [100 * V[n][grp][key] for n in names]
        bars = bx.bar(x + (j - 1.5) * w, vals, w, color=col, label=lab, **BAR)
        for b in bars:
            bx.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f'{b.get_height():.1f}', ha='center', va='bottom',
                    fontsize=FS_VAL, rotation=90)
    bx.set_title('(b) What the verifier flags (%)', loc='left', pad=19)   # above the two-row legend
    bx.legend(loc='lower right', bbox_to_anchor=(1.0, 0.97), ncol=2, columnspacing=1.0)
    for a in axes:
        a.set_ylim(0, 122)
        a.set_yticks([0, 20, 40, 60, 80, 100])
        a.tick_params(axis='x', length=0)
    bx.set_xticks(x)
    bx.set_xticklabels(short, fontsize=6.5, linespacing=1.0)
    bx.set_xlim(-0.55, len(names) - 0.45)
    return finalize(fig, 'fig6_llm_verification', pad=0.3)


# --------------------------------------------------------------------------------------------- overview figure

EXAMPLE = ('fin_320522', '-1%')  # LINC Logistics (2011): four covenants; the case placed at an earnings cushion of -1%


TIKZ_OVERVIEW = r"""
\begin{document}
\begin{tikzpicture}[
    % Drawn about 16 cm wide and included at \textwidth (12.2 cm): a size here prints at ~0.77x.
    font=\fontsize{9}{10.5}\selectfont, >={Stealth[length=2.2mm,width=1.7mm]},
    panelbox/.style={draw=teal, dash pattern=on 2.4pt off 1.8pt, line width=0.9pt, rounded corners=7pt, fill=panel},
    head/.style={font=\fontsize{11}{13}\selectfont, color=ink, anchor=south},
    block/.style={draw=ink, line width=0.8pt, rounded corners=6pt, minimum width=4.35cm, minimum height=0.9cm,
                  anchor=north west, font=\fontsize{11}{13}\selectfont\bfseries},
    inset/.style={draw=ink, line width=0.6pt, rounded corners=5pt, fill=white, anchor=north west, inner sep=0pt},
    listing/.style={fill=ink!4, rounded corners=2pt, inner xsep=4pt, inner ysep=3.5pt, anchor=north west,
                    path picture={\fill[teal] (path picture bounding box.north west) rectangle
                                  ([xshift=1.4pt]path picture bounding box.south west);}},
    chipfill/.style={draw=#1, fill=#1!12, text=ink, rounded corners=1.8pt, line width=0.6pt, inner xsep=2.5pt, inner ysep=1.8pt,
                 font=\fontsize{8.1}{9.5}\selectfont},
    tag/.style={draw=ink!45, fill=ink!7, text=ink, rounded corners=1.8pt, line width=0.5pt, inner xsep=3pt,
                inner ysep=1.5pt, text height=1.6ex, text depth=0.25ex, font=\fontsize{8.6}{9.5}\selectfont},
    mstep/.style={draw=ink!45, fill=ink!7, text=ink, rounded corners=1.8pt, line width=0.5pt, inner xsep=3pt,
                 inner ysep=2pt, font=\fontsize{8.2}{9.5}\selectfont, anchor=west},
    lab/.style={font=\fontsize{9}{10.5}\selectfont, color=black, align=left},
    flow/.style={->, line width=0.7pt, color=ink, rounded corners=4pt},
    tie/.style={line width=0.6pt, color=ink, rounded corners=4pt},
  ]
  % One palette hue per approach, light tint as background, full hue as border:
  % (a) Paul Tol purple, (b) Paul Tol sand, (c) Okabe-Ito blue (all colorblind-safe)
  \definecolor{oiorange}{HTML}{AA4499}  % (a) by hand
  \definecolor{oipurple}{HTML}{DDCC77}  % (b) reading directly
  \definecolor{oiblue}{HTML}{0072B2}    % (c) formalize, then solve
  \newcommand{\lno}[1]{{\color{teal}\fontsize{7}{8}\selectfont\ttfamily #1}}
  \def\P{5.3}     % width of the left panel
  \def\G{1.35}    % gap between columns
  \def\W{9.95}    % width of panels (a)-(c)
  \def\Bt{-3.3}   % top of panel (b)
  \def\Ct{-6.6}   % top of panel (c)
  \def\B{-11.3}   % bottom of all panels
  \pgfmathsetmacro\R{\P+\G}
  \begin{scope}[on background layer]
    \draw[panelbox] (0,0) rectangle (\P,\B);
    \draw[panelbox, draw=oiorange, fill=oiorange!9] (\R,0) rectangle (\R+\W,-2.45);
    \draw[panelbox, draw=oipurple, fill=oipurple!10] (\R,\Bt) rectangle (\R+\W,\Bt-2.45);
    \draw[panelbox, draw=oiblue, fill=oiblue!7] (\R,\Ct) rectangle (\R+\W,\B);
  \end{scope}
  \node[head] at ({\P/2}, 0.1) {THE QUARTERLY CHECK};
  \node[head] at ({\R+\W/2}, 0.1) {(a) BY HAND};
  \node[head] at ({\R+\W/2}, \Bt+0.1) {(b) READING DIRECTLY};
  \node[head] at ({\R+\W/2}, \Ct+0.1) {(c) FORMALIZE, THEN SOLVE};

  % ================================================================ clause, certificate, questions
  \node[block, fill=white] (cl) at (0.2,-0.3) {\hspace*{0.6cm}Covenant clause};
  \node at ($(cl.west)+(0.45,0)$) {\ico{\faFileContract}};
  \node[inset, minimum width={(\P-0.75)*1cm}, minimum height=2.6cm] (c1) at (0.55,-1.4) {};
  \draw[tie] (0.42,-1.2) |- (c1.west);
  \node[listing, text width={(\P-1.2)*1cm}] at (0.65,-1.5) {%
    \fontsize{7.6}{9.6}\selectfont\itshape
    \begin{tabular}{@{}r@{\hspace{4pt}}l@{}}
    \lno{1} & <<COMPANY>> (<<YEAR>>), \S\,7.9\\
    \lno{2} & (a) a Senior Debt to EBITDA\\
    \lno{3} & \ Ratio of not more than 2.5 to 1.0.\\
    \lno{4} & (b) a Total Debt to EBITDA\\
    \lno{5} & \ Ratio of not more than 3.0 to 1.0.\\
    \lno{6} & (c), (d) two fixed-charge ratios \ldots
    \end{tabular}};

  \node[block, fill=white] (ce) at (0.2,-4.55) {\hspace*{0.6cm}Compliance certificate};
  \node at ($(ce.west)+(0.45,0)$) {\ico{\faFileInvoiceDollar}};
  \node[inset, minimum width={(\P-0.75)*1cm}, minimum height=2.3cm] (c2) at (0.55,-5.65) {};
  \draw[tie] (0.42,-5.45) |- (c2.west);
  \node[listing, text width={(\P-1.2)*1cm}] at (0.65,-5.75) {%
    \fontsize{8.4}{10}\selectfont
    \begin{tabular}{@{}r@{\hspace{4pt}}l@{\hspace{8pt}}r@{}}
    \lno{1} & Test date & <<DATE>>\\
    \lno{2} & EBITDA & \$<<EBITDA>>M\\
    \lno{3} & Senior debt & \$<<SENIOR>>M\\
    \lno{4} & Total debt & \$<<TOTAL>>M\\
    \lno{5} & \multicolumn{2}{l}{+ two fixed-charge items}
    \end{tabular}};

  \node[block, fill=white] (qs) at (0.2,-8.6) {\hspace*{0.6cm}Three questions};
  \node at ($(qs.west)+(0.45,0)$) {\ico{\faQuestionCircle}};
  \node[inset, minimum width={(\P-0.75)*1cm}, minimum height=0.8cm] (c3) at (0.55,-9.75) {};
  \draw[tie] (0.42,-9.55) |- (c3.west);
  % three equal-width tags spread evenly across the card
  \foreach \q [count=\j] in {status, cushion, capacity}
    \node[tag, minimum width=1.25cm, align=center] at ({0.55 + 0.75 + (\j-1)*1.42}, -10.15) {\q};

  % ================================================================ (a) by hand
  \node[block, fill=white, minimum width=4.0cm] (an) at (\R+0.2,-0.3) {\hspace*{0.6cm}Credit analyst};
  \node at ($(an.west)+(0.45,0)$) {\ico{\faUserTie}};
  \node[lab, anchor=west] at (\R+4.45,-0.75) {every covenant by hand,\\every loan, every quarter};
  \node[inset, minimum width={(\W-0.75)*1cm}, minimum height=0.85cm] (h1) at (\R+0.55,-1.4) {};
  \draw[tie] (\R+0.42,-1.2) |- (h1.west);
  \node[mstep] (m1) at (\R+0.7,-1.825) {1\ \ read the clause};
  \node[mstep] (m2) at ($(m1.east)+(0.32,0)$) {2\ \ compute each ratio};
  \node[mstep] (m3) at ($(m2.east)+(0.32,0)$) {3\ \ solve for the answers};
  \draw[flow] (m1.east) -- (m2.west);
  \draw[flow] (m2.east) -- (m3.west);

  % ================================================================ (b) reading directly
  \node[block, fill=white, minimum width=4.0cm] (rd) at (\R+0.2,\Bt-0.3) {\hspace*{0.6cm}7B model reads};
  \node at ($(rd.west)+(0.45,0)$) {\ico{\faRobot}};
  \node[lab, anchor=west] at (\R+4.45,\Bt-0.75) {reasons over the text,\\again for every certificate};
  \node[inset, minimum width={(\W-0.75)*1cm}, minimum height=0.85cm] (b1) at (\R+0.55,\Bt-1.4) {};
  \draw[tie] (\R+0.42,\Bt-1.2) |- (b1.west);
  \node[chipfill=good, anchor=west] (s1) at (\R+0.7,\Bt-1.825) {status: <<D7STATUS>>\,\ok};
  \node[chipfill=bad, anchor=west] (s2) at ($(s1.east)+(0.12,0)$) {cushion: <<D7CUSHION>>\,\no\ (true <<RCUSHION>>)};
  \node[chipfill=good, anchor=west] at ($(s2.east)+(0.12,0)$) {capacity: <<D7CAPACITY>>M\,\ok};

  % ================================================================ (c) formalize, then solve
  \node[block, fill=white, minimum width=4.45cm] (af) at (\R+0.2,\Ct-0.3) {\hspace*{0.6cm}AutoFormalizer (7B)};
  \node at ($(af.west)+(0.45,0)$) {\ico{\faServer}};
  \node[inset, minimum width=5.95cm, minimum height=3.0cm] (f1) at (\R+0.55,\Ct-1.4) {};
  \draw[tie] (\R+0.42,\Ct-1.2) |- (f1.west);
  \node[listing, text width=5.45cm] at (\R+0.65,\Ct-1.5) {%
    \fontsize{7.2}{9.2}\selectfont\ttfamily
    \begin{tabular}{@{}r@{\hspace{4pt}}l@{}}
    \lno{1} & (define-fun covenant\_holds\\
    \lno{2} & \ ((ebitda Real) (senior\_debt Real)\\
    \lno{3} & \ \ (total\_debt Real) ... (test\_date Int))\\
    \lno{4} & \ Bool\\
    \lno{5} & \ (and (<= (/ senior\_debt ebitda) 2.5)\\
    \lno{6} & \ \ \ \ \ \ (<= (/ total\_debt ebitda) 3.0)\\
    \lno{7} & \ \ \ \ \ \ ...))
    \end{tabular}};

  \node[block, fill=sky, minimum width=2.95cm] (z3) at (\R+6.8,\Ct-0.3) {\hspace*{0.6cm}Z3 solver};
  \node at ($(z3.west)+(0.45,0)$) {\ico{\faCogs}};
  \draw[flow] (af.east) -- node[lab, above=1pt, font=\fontsize{8.4}{9.5}\selectfont] {formula, once} (z3.west);
  \node[inset, minimum width=2.7cm, minimum height=3.0cm] (z1) at (\R+7.05,\Ct-1.4) {};
  \draw[tie] (\R+6.92,\Ct-1.2) |- (z1.west);
  \node[anchor=north west, font=\fontsize{9}{10.5}\selectfont\bfseries, align=left] at (\R+7.08,\Ct-1.5)
       {every quarter:};
  \node[chipfill=good, anchor=west] at (\R+7.08,\Ct-2.35) {status: <<OSTATUS>>\,\ok};
  \node[chipfill=good, anchor=west] at (\R+7.08,\Ct-3.0) {cushion: <<OCUSHION>>\,\ok};
  \node[chipfill=good, anchor=west] at (\R+7.08,\Ct-3.65) {capacity: <<OCAPACITY>>M\,\ok};

  % ================================================================ inputs to (a), (b) and (c)
  \draw[flow] (\P,-0.75) -- node[lab, above=1pt, align=center, font=\fontsize{8.6}{9.5}\selectfont] {clause\\+ figures} (an.west);
  \draw[flow] (\P,\Bt-0.75) -- node[lab, above=1pt, align=center, font=\fontsize{8.6}{9.5}\selectfont] {clause\\+ figures} (rd.west);
  \draw[flow] (\P,\Ct-0.75) -- node[lab, above=1pt, font=\fontsize{8.6}{9.5}\selectfont] {clause} (af.west);

  \begin{pgfonlayer}{page}
    \fill[white] ([shift={(-1pt,-1pt)}]current bounding box.south west) rectangle ([shift={(1pt,1pt)}]current bounding box.north east);
  \end{pgfonlayer}
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
        'D7STATUS': d7['status'], 'D7CUSHION': pc(d7['cushion']), 'D7CAPACITY': f"{d7['capacity'] / 1e6:+,.2f}",
        'OSTATUS': ours['status'], 'OCUSHION': pc(ours['cushion']), 'OCAPACITY': f"{ours['capacity'] / 1e6:+,.2f}",
        'RSTATUS': ref['status'], 'RCUSHION': pc(ref['cushion']), 'RCAPACITY': f"{ref['capacity'] / 1e6:+,.2f}",
        'C7': f"{100 * S['qwen7b']['cushion']:.1f}", 'C32': f"{100 * S['qwen32b']['cushion']:.1f}",
        'C27': f"{100 * S['qwen27b']['cushion']:.1f}", 'COURS': f"{100 * S['rl_v2']['cushion']:.1f}",
        'SOURS': f"{100 * S['rl_v2']['score']:.1f}",
        'S27': f"{100 * S['qwen27b']['score']:.1f}", 'S32': f"{100 * S['qwen32b']['score']:.1f}",
        'S7': f"{100 * S['qwen7b']['score']:.1f}"}
    out = ('% Fig. 0 (overview). Generated by src/headroom/paper.py (fig_overview_tex); numbers from the result files.\n'
           + TIKZ_METHOD[TIKZ_METHOD.index('\\documentclass'):TIKZ_METHOD.index('\\begin{document}')]
               .replace('<<ICON_CHECK>>', str(Path(__file__).resolve().parent / 'assets' / 'check_mark_button.pdf'))
               .replace('<<ICON_CROSS>>', str(Path(__file__).resolve().parent / 'assets' / 'cross_mark.pdf'))
           + TIKZ_OVERVIEW)
    for k, v in fill.items():
        out = out.replace(f'<<{k}>>', v)
    assert '<<' not in out, 'unfilled placeholder'
    path = FIG / 'fig0_overview.tex'
    path.write_text(out)
    return compile_tex(path)


TIKZ_METHOD = r"""% Fig. M (method). Generated by src/headroom/paper.py (fig_method_tex).
% Method figure. Stage I: distill a teacher into SFT; Stage II: GRPO with a Z3-checked reward; Stage III: verify once,
% run every quarter in-house. Compile with pdflatex; needs the fontawesome5 package (CTAN).
\documentclass[tikz,border=1pt]{standalone}
\usepackage[LGR,T1]{fontenc}
\DeclareFontSubstitution{LGR}{SourceSansThree-LF}{m}{n}
\usepackage[semibold]{sourcesans}   % one humanist sans for text and math
\renewcommand{\familydefault}{\sfdefault}
\renewcommand{\ttdefault}{lmtt}
\usepackage[letterspace=90]{microtype}
\usepackage{amsmath,amssymb}
\usepackage[italic,LGRgreek]{mathastext}
\renewcommand{\theta}{\text{\fontencoding{LGR}\selectfont\upshape j}}  % italic theta looks like a 9
\usepackage{pifont}
\usepackage{fontawesome5}
\usepackage{graphicx}
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
\newcommand{\ok}{\raisebox{-0.12ex}{{\color{good!75!black}\fontsize{9.5}{9.5}\selectfont\faCheckCircle}}}
\newcommand{\no}{\raisebox{-0.12ex}{{\color{bad!75!black}\fontsize{9.5}{9.5}\selectfont\faTimesCircle}}}
\newcommand{\okc}{\raisebox{-0.22ex}{\includegraphics[height=1.15em]{<<ICON_CHECK>>}}}
\newcommand{\noc}{\raisebox{-0.22ex}{\includegraphics[height=1.15em]{<<ICON_CROSS>>}}}
\newcommand{\yes}{\okc}
\newcommand{\ico}[2][ink]{{\color{#1}\fontsize{12}{12}\selectfont #2}}

\begin{document}
\begin{tikzpicture}[
    % Drawn about 16 cm wide and included at \textwidth (12.2 cm), i.e. scaled by ~0.77: a size here prints at 0.77x.
    font=\fontsize{8.8}{10.5}\selectfont, >={Stealth[length=2.2mm,width=1.7mm]},
    panelbox/.style={draw=teal, dash pattern=on 2.4pt off 1.8pt, line width=0.9pt, rounded corners=7pt, fill=panel},
    head/.style={font=\fontsize{11}{13}\selectfont, color=ink, anchor=south},
    block/.style={draw=ink, line width=0.8pt, rounded corners=6pt, minimum width=3.95cm, minimum height=0.9cm,
                  anchor=north west, font=\fontsize{11.5}{13}\selectfont\bfseries},
    inset/.style={draw=ink, line width=0.6pt, rounded corners=5pt, fill=white, anchor=north west, minimum width=3.55cm,
                  inner sep=0pt},
    itext/.style={anchor=north west, text width=3.35cm, align=left, inner sep=0pt},
    chip/.style={draw=#1, text=#1, fill=white, rounded corners=1.8pt, line width=0.6pt, inner xsep=3pt, inner ysep=1.6pt,
                 font=\fontsize{8.4}{9}\selectfont},
    chipfill/.style={draw=#1, fill=#1!12, text=#1, rounded corners=1.8pt, line width=0.6pt, inner xsep=3pt, inner ysep=1.6pt,
                 font=\fontsize{8.4}{9}\selectfont},
    lab/.style={font=\fontsize{8.8}{10}\selectfont, color=black, align=left},
    flow/.style={->, line width=0.7pt, color=ink, rounded corners=4pt},
    tie/.style={line width=0.6pt, color=ink, rounded corners=4pt},
  ]
  \def\P{4.35}  % width of Stages I and III
  \def\W{6.1}   % width of Stage II
  \def\G{0.75}  % gap between panels
  \def\B{-9.75} % panel bottom
  \pgfmathsetmacro\Lb{\P+\G}
  \pgfmathsetmacro\Lc{\P+\W+2*\G}
  \begin{scope}[on background layer]
    \draw[panelbox] (0,0) rectangle (\P,\B);
    \draw[panelbox] (\Lb,0) rectangle (\Lb+\W,\B);
    \draw[panelbox] (\Lc,0) rectangle (\Lc+\P,\B);
  \end{scope}
  \node[head] at ({\P/2}, 0.1) {Stage I: DISTILLATION};
  \node[head] at ({\Lb+\W/2}, 0.1) {Stage II: REINFORCEMENT LEARNING};
  \node[head] at ({\Lc+\P/2}, 0.1) {Stage III: DEPLOYMENT};

  % ================================================================ Stage I
  \def\L{0}
  \node[block, fill=white] (teacher) at (\L+0.2,-0.3) {\hspace*{0.6cm}Teacher model};
  \node at ($(teacher.west)+(0.45,0)$) {\ico{\faRobot}};
  \node[inset, minimum height=2.3cm] (t1) at (\L+0.6,-1.55) {};
  \node[itext] at (\L+0.72,-1.7) {%
    {\bfseries teacher's $\varphi$ kept if}\\[5pt]
    \yes\ Z3 parses it\\[1pt]
    \yes\ semantics reviewed\\[1pt]
    \yes\ parameters grounded};
  \draw[tie] (\L+0.42,-1.2) |- (t1.west);
  \node[block, fill=mint] (sft) at (\L+0.2,-4.6) {\hspace*{0.3cm}SFT};
  \node at ($(sft.west)+(0.45,0)$) {\ico{\faGraduationCap}};
  \draw[flow] (\L+0.3,-1.2) -- (\L+0.3,-4.6);
  \node[lab, anchor=west] at (\L+0.45,-4.2) {filtered pairs $\mathcal{D}$};
  \node[inset, minimum height=3.4cm] (t2) at (\L+0.6,-5.9) {};
  \node[itext] at (\L+0.72,-6.05) {{\bfseries\boldmath $(x,\varphi)\in\mathcal{D}$}};
  \node[anchor=north west, inner sep=0pt, font=\fontsize{8.8}{10.5}\selectfont] at (\L+0.72,-6.55) {$x$:};
  \node[anchor=north west, text width=3.0cm, align=left, inner sep=0pt, font=\fontsize{8}{10}\selectfont\itshape]
       at (\L+1.05,-6.55) {``Total Debt to EBITDA \ldots\ not more than $t$ to 1''};
  \node[anchor=north west, inner sep=0pt, font=\fontsize{8.8}{10.5}\selectfont] at (\L+0.72,-7.6) {$\varphi$:};
  \node[anchor=north west, inner sep=0pt, font=\fontfamily{lmvtt}\fontsize{8.2}{10}\selectfont]
       at (\L+1.05,-7.6) {(<= (/ debt ebitda) $t$)};
  \node[anchor=north west, inner sep=0pt, font=\fontsize{8.6}{10}\selectfont] at (\L+0.72,-8.4)
       {$\mathcal{L}_{\mathrm{SFT}}=-\,\mathbb{E}_{\mathcal{D}}\log\pi_\theta(\varphi\mid x)$};
  \draw[tie] (\L+0.42,-5.5) |- (t2.west);

  % ================================================================ Stage II
  \def\L{\Lb}
  \node[block, fill=mint, minimum width={(\W-0.4)*1cm}] (grpo) at (\L+0.2,-0.3) {\hspace*{0.3cm}Policy $\pi_\theta$};
  \node at ($(grpo.west)+(0.45,0)$) {\ico{\faDiceFour}};
  \node[inset, minimum width={(\W-0.8)*1cm}, minimum height=2.3cm] (g1) at (\L+0.6,-1.55) {};
  \node[itext, text width={(\W-1.05)*1cm}] at (\L+0.72,-1.7) {{\bfseries $G$ samples per clause in $\mathcal{C}'$}};
  \draw[tie] (\L+0.42,-1.2) |- (g1.west);
  \foreach \k [count=\j] in {1, 2, 3, G} {
    \pgfmathsetmacro\x{\L+1.35+(\j-1)*1.12+(\j==4)*0.6}
    \node[color=mintdark] at (\x,-2.45) {\fontsize{12}{12}\selectfont\faFileCode[regular]};
    \node[color=mintdark, font=\fontsize{8.8}{9}\selectfont] at (\x,-2.93) {$\varphi_\k$};
    \node[chipfill=skydark] at (\x,-3.4) {$r_\k$};
  }
  \node[color=ink] at (\L+4.6,-2.93) {$\cdots$};

  \node[block, fill=sky, minimum width={(\W-0.4)*1cm}] (z3) at (\L+0.2,-4.6) {\hspace*{0.3cm}Reward $r(\varphi)$};
  \node at ($(z3.west)+(0.45,0)$) {\ico{\faCogs}};
  \draw[flow] (\L+0.3,-1.2) -- (\L+0.3,-4.6);
  \draw[flow] (\L+0.14,-4.6) -- (\L+0.14,-1.2);
  \node[lab, anchor=west] at (\L+0.45,-4.2) {$\varphi_i\downarrow$\ \ $r_i\uparrow$};
  \node[inset, minimum width={(\W-0.8)*1cm}, minimum height=3.4cm] (g2) at (\L+0.6,-5.9) {};
  \draw[tie] (\L+0.42,-5.5) |- (g2.west);
  \node[itext, text width={(\W-1.05)*1cm}] at (\L+0.72,-6.05) {{\bfseries each limit} $\tau\le t$,\\ at every date and condition};
  % each scenario fixes the constrained quantity tau; the solver checks phi there (and, at the breach side, also not-phi,
  % so that a formula that is always false cannot pass)
  \foreach \n/\q/\a/\c/\m [count=\j] in {compliant/(1-\delta)\,t/SAT/good/\okc, boundary/t/SAT/good/\okc, breach/(1+\delta)\,t/UNSAT/bad/\noc} {
    \pgfmathsetmacro\x{\L+1.45+(\j-1)*1.68}
    \node[font=\fontsize{8.8}{9}\selectfont\bfseries] at (\x,-6.97) {\n};
    \node[font=\fontsize{8.6}{9}\selectfont] at (\x,-7.33) {$\tau=\q$};
    \node[chipfill=\c, text=ink] at (\x,-7.77) {$\textcolor{mintdark}{\varphi}$:\,\a\,\m};
  }
  \node[chipfill=good, text=ink] at (\L+1.45+2*1.68,-8.25) {$\neg\textcolor{mintdark}{\varphi}$:\,SAT\,\okc};
  \node[anchor=west, font=\fontsize{8.6}{10}\selectfont] at (\L+0.68,-8.85)
       {$\textcolor{skydark}{r(\textcolor{mintdark}{\varphi})}=\mathbf{1}[\textcolor{mintdark}{\varphi}\text{ valid}]\cdot\big(\alpha+(1-\alpha)\,\tfrac{\#\,\text{passed}}{\#\,\text{tests}}\big)$};

  % ================================================================ Stage III
  \def\L{\Lc}
  \node[block, fill=mint] (form) at (\L+0.2,-0.3) {\hspace*{0.6cm}AutoFormalizer};
  \node at ($(form.west)+(0.45,0)$) {\ico{\faServer}};
  \node[inset, minimum height=2.3cm] (f1) at (\L+0.6,-1.55) {};
  \node[itext] at (\L+0.72,-1.7) {{\bfseries Equivalence check}};
  \draw[tie] (\L+0.42,-1.2) |- (f1.west);
  \node[color=mintdark] (fa) at (\L+1.25,-2.7) {\fontsize{12}{12}\selectfont\faFileCode[regular]};
  \node[color=ink] (fb) at (\L+3.5,-2.7) {\fontsize{12}{12}\selectfont\faFileCode[regular]};
  \draw[<->, line width=0.7pt, color=skydark] ($(fa.east)+(0.05,0)$) -- node[above=0pt, font=\fontsize{8.6}{9}\selectfont\bfseries] {$\varphi\equiv\varphi'$\,?} ($(fb.west)+(-0.05,0)$);
  \node[chipfill=good, anchor=west] at (\L+0.72,-3.4) {$\equiv$: deploy};
  \node[chipfill=bad, anchor=east] at (\L+\P-0.3,-3.4) {$\not\equiv$: analyst};
  \node[block, fill=sky] (run) at (\L+0.2,-4.6) {\hspace*{0.6cm}Compliance test};
  \node at ($(run.west)+(0.45,0)$) {\ico{\faCalendarCheck[regular]}};
  \draw[flow] (\L+0.3,-1.2) -- (\L+0.3,-4.6);
  \node[lab, anchor=west] at (\L+0.45,-4.2) {verified formulas};
  \node[inset, minimum height=3.4cm] (f2) at (\L+0.6,-5.9) {};
  \node[itext] at (\L+0.72,-6.05) {{\bfseries\boldmath $\varphi$ on certificate $v$}};
  \draw[tie] (\L+0.42,-5.5) |- (f2.west);
  \node[chipfill=good, anchor=west] (s1) at (\L+0.72,-6.9) {comply};
  \node[chipfill=bad, anchor=west] (s2) at ($(s1.east)+(0.1,0)$) {breach};
  \node[chipfill=amber, anchor=west] at (\L+0.72,-7.42) {undetermined};
  \node[anchor=north west, inner sep=0pt, align=left, font=\fontsize{8.6}{11}\selectfont] at (\L+0.72,-7.95)
       {{\bfseries cushion} $\boldsymbol{c}$, {\bfseries capacity} $\boldsymbol{k}$\\
        {\bfseries verify} LLM answers};

  % ================================================================ between stages
  \draw[flow] (sft.east) -| (\P+0.22,-1.0) |- (grpo.west);
  \node[lab, rotate=90, font=\fontsize{8.6}{10}\selectfont] at (\P+0.5,-2.9) {init $\pi_{\mathrm{SFT}}$};
  \draw[flow] (grpo.east) -- (form.west);

  \begin{pgfonlayer}{page}   % opaque white page under everything (also for the SVG)
    \fill[white] ([shift={(-1pt,-1pt)}]current bounding box.south west) rectangle ([shift={(1pt,1pt)}]current bounding box.north east);
  \end{pgfonlayer}
\end{tikzpicture}
\end{document}
"""


def fig_method_tex():
    """Method figure as TikZ (results/paper/figures/fig_method.tex); symbols only, values are in the paper caption.
    Needs the fontawesome5 LaTeX package."""
    path = FIG / 'fig_method.tex'
    assets = Path(__file__).resolve().parent / 'assets'  # Microsoft Fluent Emoji, MIT (assets/ICONS_LICENSE.txt)
    path.write_text(TIKZ_METHOD.replace('<<ICON_CHECK>>', str(assets / 'check_mark_button.pdf'))
                               .replace('<<ICON_CROSS>>', str(assets / 'cross_mark.pdf')))
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
        rows = [json.loads(l) for l in open(P.MODELS / RL_MODEL_DIR.format(sd) / 'exec_rewards.jsonl')]
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


def table_families():
    """Same recipe on three model families (adopted reward): reading directly, prompted, SFT, FormaFin (3 seeds)."""
    v = json.loads((RES / 'verification/summary.json').read_text())
    ref = per_clause(marks(dict((k, r) for k, _, _, r in DIRECT)['qwen27b']))
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3.5pt}',
           r'\caption{The same recipe on three model families (per-clause score). FormaFin: mean $\pm$ s.d. over three '
           r'seeds; proven: formulas Z3 proves equivalent to the reference reading (of 143). $\Delta_{27B}$: FormaFin minus '
           r'the Qwen3.8-27B reader (88.8), paired sign-flip test over clauses. Differences are computed before rounding.}',
           r'\label{tab:families}', r'\begin{tabular}{@{}lccccc@{}}', r'\toprule',
           r'\textbf{Base model} & \textbf{Reads} & \textbf{SFT} & \textbf{FormaFin} & \textbf{Proven} & '
           r'$\boldsymbol{\Delta_{27B}}$ \\', r'\midrule']
    md = ['| Base model | Reads directly | Prompted | SFT | FormaFin (3 seeds) | Proven | minus 27B reader |', '|---|---|---|---|---|---|---|']
    out = {}
    for lab, direct, prompted, sft, rl in FAMILIES:
        d = mean(per_clause(marks(f'answers_long/direct_{direct}.jsonl')))
        pr = mean(per_clause(marks(f'answers/formula_{prompted}.jsonl')))
        sf = mean(per_clause(marks(f'answers/formula_{sft}.jsonl')))
        runs = [per_clause(marks(f'answers/formula_{rl.format(sd)}.jsonl')) for sd in (42, 43, 44)]
        sc_ = [mean(r) for r in runs]
        avg = {c: np.mean([r[c] for r in runs]) for c in runs[0]}
        prov = [v[rl.format(sd)]['equivalent'] for sd in (42, 43, 44)]
        t = paired(avg, ref)
        out[lab] = dict(direct=d, prompted=pr, sft=sf, formafin=sc_, proven=prov, vs27b=t)
        cell = f"{pct(np.mean(sc_))}" + r'{\scriptsize$\pm$' + pct(np.std(sc_, ddof=1)) + '}'
        lat.append(f"{lab} & {pct(d)} & {pct(sf)} & {cell} & {np.mean(prov):.0f} & {100 * t['diff']:+.1f} ({fmt_p(t['p'])}) \\\\")
        md.append(f"| {lab} | {pct(d)} | {pct(pr)} | {pct(sf)} | {pct(np.mean(sc_))} ± {pct(np.std(sc_, ddof=1))} | "
                  f"{np.mean(prov):.0f} | {100 * t['diff']:+.1f} (p = {t['p']:.4f}) |")
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (OUT / 'families.json').write_text(json.dumps(out, indent=1, default=float))
    return write('tab7_families', '\n'.join(lat), '\n'.join(md))


VERIFY_READERS = [('Qwen2.5-Coder-7B', 'answers_long/direct_qwen7b.jsonl'), ('Llama-3.1-8B', 'answers_long/direct_llama8b.jsonl'),
                  ('Mistral-7B', 'answers_long/direct_mistral7b.jsonl'), ('Qwen3.8-27B', 'answers_long/direct_qwen27b.jsonl'),
                  ('Qwen2.5-Coder-32B', 'answers_long/direct_qwen32b.jsonl'), ('Llama-3.3-70B (4-bit)', 'answers_long/direct_llama70b.jsonl'),
                  ('Claude Sonnet 5.5$^\\dagger$', 'answers/direct_claude_sonnet55.jsonl')]


def verify_reader(reader_rel, formula_rel, keep=None):
    """FormaFin as a verifier of an LLM's analysis: each answer is accepted iff it matches the formula's answer
    (same tolerance as the benchmark); a question the formula cannot answer is flagged. Pooled over questions."""
    A, F = answers(reader_rel), answers(formula_rel)
    n = err = caught = good = flagged_good = acc_n = acc_ok = 0
    for k, c in cases().items():
        if keep is not None and not keep(c):
            continue
        truth = sc.score_case(c, A.get(k))
        fa = F.get(k)
        verdict = sc.score_case(dict(c, answers={q: (fa or {}).get(q) for q in ('status', 'cushion', 'capacity')}), A.get(k)) if fa else {}
        for q, t in truth.items():
            ok = verdict.get(q, 0) == 1
            n += 1
            if t:
                good += 1
                flagged_good += not ok
            else:
                err += 1
                caught += not ok
            if ok:
                acc_n += 1
                acc_ok += t
    return dict(n=n, llm_acc=(n - err) / n, caught=caught / err if err else float('nan'),
                false_flag=flagged_good / good if good else float('nan'), acc_accepted=acc_ok / acc_n if acc_n else float('nan'),
                coverage=n)


def table_llm_verification():
    """Does a FormaFin formula catch the errors of LLM analyses? All three seeds (mean +- s.d.); then only the clauses
    on which two formalizers provably agree (seed 42 x seed 43), the ones that would run without review."""
    cross_cache = OUT / 'cross_sched42_sched43.json'
    if not cross_cache.exists():
        cross_check(*CROSS[0][:4])
    agree = json.loads(cross_cache.read_text())
    keep = lambda c: agree.get(c['clause_id']) == 'agree'
    total = sum(len(sc.score_case(c, None)) for c in cases().values())
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3pt}',
           r'\caption{FormaFin as a verifier of LLM analyses (2,527 questions). An LLM answer is accepted if it matches '
           r'the answer of the FormaFin formula (benchmark tolerance) and flagged otherwise. Caught: share of wrong LLM '
           r'answers flagged; false flags: share of correct answers flagged; accepted: accuracy of accepted answers. Acc.: '
           r'share of the 2,527 questions answered correctly, pooled over questions, so it differs from the per-covenant '
           r'score of \cref{tab:main}. '
           r'All: mean over three FormaFin seeds (s.d. over seeds $\le$ 0.8 points for caught, $\le$ 3.3 for false flags). Agreed: only clauses on which seeds 42 and 43 '
           r'provably agree, which run without review. $^\dagger$Proprietary; also reference reader A, so its accuracy '
           r'is biased upward.}',
           r'\label{tab:llmverify}', r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
           r' & & \multicolumn{3}{c}{\textbf{All clauses}} & \multicolumn{2}{c}{\textbf{Agreed}} \\',
           r'\cmidrule(lr){3-5}\cmidrule(l){6-7}',
           r'\textbf{LLM analysis} & \textbf{Acc.} & \textbf{Caught} & \textbf{False fl.} & \textbf{Accepted} & '
           r'\textbf{Caught} & \textbf{False fl.} \\', r'\midrule']
    md = ['| LLM analysis | LLM accuracy | Caught | False flags | Accepted acc. | Caught (agreed) | False flags (agreed) |',
          '|---|---|---|---|---|---|---|']
    out = {}
    for lab, rel in VERIFY_READERS:
        if not (RES / rel).exists():
            continue
        runs = [verify_reader(rel, RL_V2[sd]) for sd in RL_V2]
        ag = verify_reader(rel, RL_V2[42], keep)
        m = {k: float(np.mean([r[k] for r in runs])) for k in ('llm_acc', 'caught', 'false_flag', 'acc_accepted')}
        sd = {k: float(np.std([r[k] for r in runs], ddof=1)) for k in ('caught', 'false_flag', 'acc_accepted')}
        out[lab] = dict(all=m, sd=sd, agreed=ag, agreed_coverage=ag['n'] / total)
        lat.append(f"{lab} & {pct(m['llm_acc'])} & {pct(m['caught'])} & {pct(m['false_flag'])} & {pct(m['acc_accepted'])} & "
                   f"{pct(ag['caught'])} & {pct(ag['false_flag'])} \\\\")
        md.append(f"| {lab.replace('$^\\dagger$', ' (†)')} | {pct(m['llm_acc'])} | {pct(m['caught'])} ± {pct(sd['caught'])} | "
                  f"{pct(m['false_flag'])} ± {pct(sd['false_flag'])} | {pct(m['acc_accepted'])} | {pct(ag['caught'])} | {pct(ag['false_flag'])} |")
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    md.append(f"\nAgreed clauses cover {100 * next(iter(out.values()))['agreed_coverage']:.1f}% of questions.")
    (OUT / 'llm_verification.json').write_text(json.dumps(out, indent=1, default=float))
    return write('tab8_llm_verification', '\n'.join(lat), '\n'.join(md))


COVTYPE_ORDER = ['coverage', 'capitalization', 'leverage', 'minimum level', 'liquidity', 'asset-based']
_COVTYPE_RULES = [  # first match wins, per limit; a clause counts in every type among its limits
    ('asset-based', lambda lab, den, roles: bool(re.search(r'loan.to.value|collateral|asset pool|asset coverage|tangible assets to outstanding|borrowing base|liabilities to assets', lab))),
    ('liquidity', lambda lab, den, roles: bool(re.search(r'current ratio|quick ratio|liquid', lab))),
    ('minimum level', lambda lab, den, roles: not den),
    ('coverage', lambda lab, den, roles: bool(re.search(r'coverage|interest|fixed charge|debt service|income to debt|cash flow/total debt', lab)) and 'leverage' not in lab),
    ('leverage', lambda lab, den, roles: any(roles.get(n) == 'earnings' for n in den) or bool(re.search(r'ebitda', lab)))]


def covenant_types(clause_id):
    """Covenant types of a clause from its reference specification (label keywords and item roles); default: capitalization."""
    sp = specs()[clause_id]['spec']
    roles = {i['name']: i['role'] for i in sp['items']}
    out = set()
    for L in sp['limits']:
        lab, den = L['label'].lower(), [n for n, _ in L['denominator']]
        out.add(next((t for t, rule in _COVTYPE_RULES if rule(lab, den, roles)), 'capitalization'))
    return out


def table_covtypes():
    """Per-clause score by covenant type: FormaFin on three families (seed mean), SFT and the four readers."""
    S = system_stats()
    clauses = sorted({c['clause_id'] for c in cases().values()})
    ct = {k: covenant_types(k) for k in clauses}
    cols = [('rl_v2', 'Qwen'), ('ff_llama', 'Llama'), ('ff_mistral', 'Mistral'), ('sft', 'SFT'),
            ('qwen27b', '27B'), ('qwen32b', '32B'), ('llama70b', '70B'), ('qwen7b', '7B')]
    lat = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{3.5pt}',
           r'\caption{Per-clause score by covenant type (a clause with covenants of several types counts in each). Types '
           r'come from the reference specifications. FormaFin: mean over three training runs. Bold: best in each row.}',
           r'\label{tab:covtypes}', r'\begin{adjustbox}{max width=\linewidth}', r'\begin{tabular}{@{}lr' + 'c' * len(cols) + r'@{}}', r'\toprule',
           r' & & \multicolumn{3}{c}{\textbf{FormaFin}} & & \multicolumn{4}{c}{\textbf{Reads directly}} \\',
           r'\cmidrule(lr){3-5}\cmidrule(l){7-10}',
           r'\textbf{Covenant type} & \textbf{Clauses} & ' + ' & '.join(r'\textbf{' + l + '}' for _, l in cols) + r' \\', r'\midrule']
    md = ['| Type | Clauses | ' + ' | '.join(l for _, l in cols) + ' |', '|---' * (len(cols) + 2) + '|']
    out = {}
    for t in COVTYPE_ORDER:
        ks = [k for k in clauses if t in ct[k]]
        vals = [np.mean([S[c]['pc'][k] for k in ks]) for c, _ in cols]
        best = max(vals)
        out[t] = dict(n=len(ks), **{c: float(v) for (c, _), v in zip(cols, vals)})
        lat.append(f"{t.capitalize()} & {len(ks)} & " + ' & '.join((r'\textbf{' + pct(v) + '}') if v == best else pct(v) for v in vals) + r' \\')
        md.append(f"| {t} | {len(ks)} | " + ' | '.join(pct(v) for v in vals) + ' |')
    lat += [r'\bottomrule', r'\end{tabular}', r'\end{adjustbox}', r'\end{table}']
    (OUT / 'covtypes.json').write_text(json.dumps(out, indent=1))
    return write('tab9_covtypes', '\n'.join(lat), '\n'.join(md))


def build_all():
    out = dict(tables=[table_benchmark(), table_main(), table_paired(), table_verification(), table_selfcheck()[:2], table_sensitivity(), table_families(), table_llm_verification(), table_covtypes()],
               stats=text_stats(),
               figures=[fig_overview_tex(), fig_method_tex(), fig_main(), fig_size(), fig_training(), fig_verification(), fig_cases(), fig_llm_verification()])
    return out


if __name__ == '__main__':
    r = build_all()
    for latex, md in r['tables']:
        print(md, '\n')
    print([str(p) for p in r['figures']])

