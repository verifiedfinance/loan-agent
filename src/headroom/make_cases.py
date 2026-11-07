"""Headroom cases (PLAN.md): constructed compliance certificates for every kept spec, reference answers, validation.

Per clause, at each test date (one per schedule step) the varied item (earnings, else debt) is placed so that its
headroom equals each target (cushion -5, -1, +1, +5, +20 % of earnings; for a debt-only clause, debt at 105, 101, 99,
95, 80 % of its maximum), with the other items drawn at random until every other limit holds with >= 10% slack.
Plus one case per condition flag switched on, and one case with one needed item not reported.
All figures are constructed. Validation: the spec rendered as SMT and evaluated by the formula-arm code
(headroom.FormulaEvaluator) must reproduce every reference answer.

    python src/headroom/make_cases.py     # -> cases.jsonl, validation.json
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import datetime as dt
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

HERE = P.HEADROOM_DATA
import headroom as hr  # noqa: E402

TARGETS = (-5, -1, 1, 5, 20)
SEED = 2026
STATUS = ('Status: the credit agreement is in force; loans are outstanding. The test date is the last day of a fiscal '
          'quarter. Period amounts (earnings, interest, cash flow, charges) are for the four fiscal quarters ended on the '
          'test date; balance amounts (debt, assets, liabilities, net worth) are as of the test date. All amounts are as '
          'defined in the agreement. No waiver or amendment applies. These figures are constructed for this test.')


def quarter_ends(lo, hi):
    """Quarter-end dates (YYYYMMDD ints) in [lo, hi] (None = open), stepping from a sensible start."""
    start = lo or (hi - 30000 if hi else 20000101)
    y = start // 10000
    out = []
    for yy in range(y, y + 6):
        for md in (331, 630, 930, 1231):
            d = yy * 10000 + md
            if d >= start and (hi is None or d <= hi):
                out.append(d)
    return out


def test_dates(spec, year):
    steps = {(s.get('from'), s.get('to')) for l in spec['limits'] for s in l['schedule']}
    if len(steps) == 1 and steps == {(None, None)}:
        return [((year or 2000) + 1) * 10000 + 1231]
    dates = []
    for lo, hi in sorted(steps, key=lambda s: (s[0] or '', s[1] or '')):
        q = quarter_ends(hr.date_int(lo), hr.date_int(hi))
        if q:
            dates.append(q[min(1, len(q) - 1)])
    return sorted(set(dates))


def fmt_date(d):
    return dt.date(d // 10000, d // 100 % 100, d % 100).strftime('%B %-d, %Y')


def certificate(rec, values, date, flags):
    spec = rec['spec']
    lines = [f"Borrower: {rec['company'].title() if rec.get('company') else 'The Borrower'}", f'Test date: {fmt_date(date)}', STATUS,
             'Compliance certificate figures:']
    for it in spec['items']:
        v = values[it['name']]
        lines.append(f"  {it['description']}: " + ('not reported' if v is None else f'${float(v):,.0f}'))
    for f in spec['flags']:
        lines.append(f"  {f['description']}: " + {True: 'yes', False: 'no'}[flags[f['name']]])
    return '\n'.join(lines)


def slack_ok(spec, values, date, flags, skip):
    """Every applicable limit not using `skip` holds with >= 10% relative slack; all sides positive."""
    for lim in spec['limits']:
        for terms in (lim['numerator'], lim['denominator']):
            if terms and hr.side(terms, values) <= 0:
                return False
        if not hr.applies(lim, date, flags):
            continue
        if skip in {i for i, _ in lim['numerator'] + lim['denominator']}:
            continue
        num = hr.side(lim['numerator'], values)
        v = num if not lim['denominator'] else num / hr.side(lim['denominator'], values)
        t = hr.threshold_on(lim, date)
        if lim['op'] in ('<=', '<') and not v <= t * Fraction(9, 10):
            return False
        if lim['op'] in ('>=', '>') and not v >= t * Fraction(11, 10):
            return False
    return True


def draw(spec, date, flags, vary, rng):
    for _ in range(5000):
        vals = {i['name']: Fraction(round(10 ** rng.uniform(6.5, 9))) for i in spec['items']}
        if slack_ok(spec, vals, date, flags, vary):
            return vals
    return None


def place(spec, vals, date, flags, vary, role, target):
    """Set the varied item so that its headroom equals the target; None if impossible."""
    def feasible_with(item, x):
        v = dict(vals)
        v[item] = Fraction(x)
        return hr.spec_status(spec, v, date, flags) == 'comply'
    kind = 'cushion' if role == 'earnings' else 'capacity'
    b = hr._search(lambda x: feasible_with(vary, x), vals[vary], kind, hr.domain(spec, vals, date, flags, vary, kind))
    if isinstance(b, str):
        return None
    v = dict(vals)
    # cushion c%: x0 = x_min / (1 - c/100); debt at (100 - c)% of its maximum: x0 = x_max * (1 - c/100)
    v[vary] = Fraction(round(float(b / (1 - Fraction(target, 100)) if kind == 'cushion' else b * (1 - Fraction(target, 100)))))
    return v if v[vary] > 0 else None


def main():
    rng = random.Random(SEED)
    specs = [json.loads(l) for l in open(HERE / 'specs.jsonl')]
    cases, skipped = [], {}
    for rec in specs:
        spec = rec['spec']
        e, d = hr.roles(spec)
        vary, role = (e, 'earnings') if e else (d, 'debt')
        n = 0

        def add(kind, note, values, date, flags):
            nonlocal n
            ans = hr.reference_answers(spec, values, date, flags)
            cases.append(dict(case_id=f"{rec['id']}__{n:02d}", clause_id=rec['id'], kind=kind, note=note, test_date=date,
                              values={k: (None if v is None else str(v)) for k, v in values.items()}, flags=flags,
                              earnings_item=e, debt_item=d, answers=ans, certificate=certificate(rec, values, date, flags)))
            n += 1
        base_flags = {f['name']: False for f in spec['flags']}
        for date in test_dates(spec, rec.get('agreement_year')):
            vals = draw(spec, date, base_flags, vary, rng)
            if vals is None:
                skipped[rec['id']] = f'no comfortable draw at {date}'
                continue
            for t in TARGETS:
                v = place(spec, vals, date, base_flags, vary, role, t)
                if v is not None:
                    add('headroom', f'{role} headroom target {t:+d}% on {date}', v, date, base_flags)
        if n == 0:
            continue
        date = cases[-1]['test_date']
        for f in spec['flags']:
            fl = dict(base_flags, **{f['name']: True})
            vals = draw(spec, date, fl, vary, rng)
            v = place(spec, vals, date, fl, vary, role, 5) if vals else None
            if v is not None:
                add('flag_on', f"{f['name']} = yes, target +5%", v, date, fl)
        vals = draw(spec, date, base_flags, vary, rng)
        v = place(spec, vals, date, base_flags, vary, role, 20) if vals else None
        if v is not None:
            used = [i for l in spec['limits'] if hr.applies(l, date, base_flags) for i, _ in l['numerator'] + l['denominator'] if i != vary]
            miss = used[0] if used else vary
            v[miss] = None
            add('missing', f'{miss} not reported', v, date, base_flags)

    # validation through the formula-arm code
    by = {r['id']: r for r in specs}
    bad = []
    for c in cases:
        spec = by[c['clause_id']]['spec']
        ev = hr.FormulaEvaluator(hr.spec_to_smt(spec), spec)
        vals = {k: (None if v is None else Fraction(v)) for k, v in c['values'].items()}
        got = ev.answers(spec, vals, c['test_date'], c['flags'])
        ok = got['status'] == c['answers']['status'] and all(
            (isinstance(c['answers'][q], str) or c['answers'][q] is None) and got[q] == c['answers'][q]
            or (isinstance(c['answers'][q], float) and isinstance(got[q], float) and abs(got[q] - c['answers'][q]) <= 1e-6 * max(1, abs(c['answers'][q])))
            for q in ('cushion', 'capacity'))
        if not ok:
            bad.append(dict(case_id=c['case_id'], ref=c['answers'], smt=got))
    (HERE / 'cases.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in cases))
    from collections import Counter
    summary = dict(clauses=len({c['clause_id'] for c in cases}), cases=len(cases), skipped=skipped,
                   kinds=dict(Counter(c['kind'] for c in cases)), status=dict(Counter(c['answers']['status'] for c in cases)),
                   cushion_asked=sum(c['answers']['cushion'] is not None for c in cases),
                   capacity_asked=sum(c['answers']['capacity'] is not None for c in cases),
                   validation_correct=len(cases) - len(bad), validation_mismatches=bad[:20])
    (HERE / 'validation.json').write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != 'validation_mismatches'}, indent=1, default=str))
    if bad:
        print('MISMATCHES:', json.dumps(bad[:5], default=str))


if __name__ == '__main__':
    main()
