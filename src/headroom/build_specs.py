"""Reference reading of each test_unseen clause (PLAN.md): reader A (Claude) proposes items and limits; reader B
(Qwen3.8-27B, another model family) reads the limits independently, given only A's item names and descriptions.
A spec is kept when A and B agree on every limit and every threshold/date appears in the clause text.

    python src/headroom/build_specs.py read_a      # Claude, resume-safe -> reader_a.jsonl
    python src/headroom/build_specs.py read_b      # Qwen3.8-27B on the GPU  -> reader_b.jsonl
    python src/headroom/build_specs.py merge       # -> specs.jsonl, specs_excluded.jsonl, specs_summary.json
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = P.HEADROOM_DATA  # readings and specs are written here
CLAUSES = P.TEST_UNSEEN

FORMAT = """Spec format (JSON):
{"supported": true|false, "reason": "<why not supported, else empty>",
 "items": [{"name": "snake_case", "description": "<the defined term as the clause uses it>", "role": "earnings|debt|other"}],
 "flags": [{"name": "snake_case", "description": "<a condition that switches a limit on or off>"}],
 "limits": [{"label": "<the covenant's name>",
             "numerator": [["<item>", 1 or -1], ...],
             "denominator": [["<item>", 1 or -1], ...],          # [] for a level limit (e.g. minimum net worth in dollars)
             "op": "<=|<|>=|>",                                   # the condition for COMPLIANCE: value op threshold
             "unit": "ratio|pct|usd",                             # ratio: X to 1; pct: a percentage, threshold as a fraction (60% -> 0.60); usd: dollars
             "schedule": [{"from": "YYYY-MM-DD"|null, "to": "YYYY-MM-DD"|null, "threshold": <number>}],   # test dates, inclusive; one step with nulls if constant
             "when": null | {"flag": "<flag>", "value": true|false}}]}
Rules:
- One item per distinct defined amount the covenant uses (e.g. consolidated_ebitda, consolidated_total_debt,
  consolidated_interest_expense, capital_expenditures, current_assets). A ratio of A to B has A in the numerator and B
  in the denominator; "A minus C" is [["a",1],["c",-1]]. Use role "earnings" for the single earnings measure (EBITDA,
  EBIT, cash flow, net income) if there is one, "debt" for the single OUTSTANDING debt balance (total / funded / senior
  debt, indebtedness, liabilities) if there is one - never for debt service, interest or payments - otherwise "other".
- If the clause names a ratio without saying what it is a ratio of (e.g. "the Fixed Charge Coverage Ratio shall not be
  less than 1.20"), use one item for the ratio itself as a level limit.
- "not more than 3.50 to 1.00" -> op "<=", threshold 3.5; "less than 3.0 to 1.0" (must stay below) -> op "<";
  "at least 1.25" / "not less than" -> ">="; "greater than" (must exceed) -> ">".
- A schedule of thresholds by period: one step per period with its test-date range; "thereafter" has to null.
- supported = false (with a reason) if the clause has no numeric financial covenant, if a threshold depends on
  another quantity (e.g. "$60,000,000 plus 75% of net proceeds of equity issued"), if a threshold is redacted, or if
  test dates in a schedule cannot be pinned to calendar dates.
Return ONLY the JSON."""

PROMPT_A = """Read this loan-covenant clause and write its structured spec.

Clause:
{clause}

{fmt}"""

PROMPT_B = """Read this loan-covenant clause and write the "limits" of its structured spec. Use ONLY these items and
flags (names fixed):
items: {items}
flags: {flags}

Clause:
{clause}

{fmt}
Return ONLY {{"supported": ..., "reason": ..., "limits": [...]}}."""


def parse_json(text):
    m = re.search(r'\{.*\}', text or '', re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def clauses():
    return [json.loads(l) for l in open(CLAUSES)]


def read_a():
    from verifiqa.generation.llm_client import ChatMessage, ClaudeCliClient
    out = HERE / 'reader_a.jsonl'
    done = {json.loads(l)['id'] for l in open(out)} if out.exists() else set()
    todo = [c for c in clauses() if c['id'] not in done]
    client = ClaudeCliClient(model='claude-sonnet-5-5', effort='medium', timeout=300)

    def one(c):
        for _ in range(2):
            try:
                text = client.chat([ChatMessage('user', PROMPT_A.format(clause=c['text'].strip(), fmt=FORMAT))], stage='headroom_spec_a')
                return dict(id=c['id'], output=text, spec=parse_json(text))
            except Exception as e:  # noqa: BLE001
                err = f'{type(e).__name__}: {str(e)[:200]}'
        return dict(id=c['id'], error=err)
    with ThreadPoolExecutor(8) as ex, open(out, 'a') as h:
        for i, r in enumerate(ex.map(one, todo), 1):
            if 'error' in r:
                print('FAILED', r['id'], r['error'], flush=True)
                continue
            h.write(json.dumps(r) + '\n')
            h.flush()
            if i % 25 == 0:
                print(f'{i}/{len(todo)}', flush=True)
    print('READ_A_DONE', flush=True)


def read_b():
    import hf_generation as wf
    a = {json.loads(l)['id']: json.loads(l) for l in open(HERE / 'reader_a.jsonl')}
    items = []
    for c in clauses():
        s = (a.get(c['id']) or {}).get('spec')
        if not s or not s.get('supported') or not s.get('items'):
            continue
        it = json.dumps([{k: i[k] for k in ('name', 'description')} for i in s['items']])
        fl = json.dumps([{k: f[k] for k in ('name', 'description')} for f in s.get('flags', [])])
        items.append(dict(case_id=c['id'], clause_id=c['id'], messages=[{'role': 'user', 'content': PROMPT_B.format(
            items=it, flags=fl, clause=c['text'].strip(), fmt=FORMAT)}]))
    model_id = 'Qwen/Qwen3.8-27B'
    tok, model = wf.load(model_id)
    wf.run(tok, model, model_id, items, HERE / 'reader_b.jsonl', 8, 1536, 6144)
    print('READ_B_DONE', flush=True)


# ------------------------------------------------------------------------------------------- merge
def canon(limit):
    terms = lambda t: tuple(sorted((str(i), int(s)) for i, s in (t or [])))
    sched = tuple(sorted((st.get('from') or '', st.get('to') or '', round(float(st['threshold']), 6)) for st in limit['schedule']))
    w = limit.get('when')
    return (terms(limit['numerator']), terms(limit.get('denominator')), limit['op'], limit['unit'], sched,
            (w['flag'], bool(w['value'])) if w else None)


def number_in_text(x, unit, text):
    t = text.replace(',', '')
    cands = set()
    x = float(x)
    if unit == 'usd':
        cands |= {f'{x:.0f}', f'{x / 1e6:g} million', f'{x / 1e6:g},000,000'.replace(',', ''), f'{x / 1e3:g},000'.replace(',', '')}
    elif unit == 'pct':
        cands |= {f'{x * 100:g}%', f'{x * 100:g} %', f'{x * 100:g} percent', f'{x:g}'}
    else:
        cands |= {f'{x:g}', f'{x:.2f}', f'{x:.1f}', f'{x:.3f}'}
        cands |= {c[1:] for c in list(cands) if c.startswith('0.')}  # ".23"
    return any(re.search(r'(?<![\d.])' + re.escape(c) + r'(?![\d])', t, re.I) for c in cands if c)


def check_spec(spec, text):
    """Deterministic checks; returns a reason string or ''."""
    names = {i['name'] for i in spec['items']}
    flags = {f['name'] for f in spec.get('flags', [])}
    if not spec['limits']:
        return 'no limits'
    for lim in spec['limits']:
        used = [i for i, _ in lim['numerator'] + (lim.get('denominator') or [])]
        if not lim['numerator'] or any(i not in names for i in used):
            return f'limit uses an undeclared item: {used}'
        if lim['op'] not in ('<=', '<', '>=', '>') or lim['unit'] not in ('ratio', 'pct', 'usd'):
            return 'bad op or unit'
        if lim.get('when') and lim['when'].get('flag') not in flags:
            return 'undeclared flag'
        for st in lim['schedule']:
            if not number_in_text(st['threshold'], lim['unit'], text):
                return f"threshold {st['threshold']} ({lim['unit']}) not found in the text"
            for k in ('from', 'to'):
                if st.get(k) and st[k][:4] not in text and st[k][2:4] not in re.findall(r'\d\d', text):
                    return f'schedule date {st[k]} not in the text'
    used_all = {i for lim in spec['limits'] for i, _ in lim['numerator'] + (lim.get('denominator') or [])}
    if any(i['role'] in ('earnings', 'debt') and i['name'] not in used_all for i in spec['items']):
        return 'earnings/debt item not used by any limit'
    if not any(i['role'] in ('earnings', 'debt') for i in spec['items']):
        return 'no earnings or debt item (status-only clause)'
    if sum(i['role'] == 'earnings' for i in spec['items']) > 1 or sum(i['role'] == 'debt' for i in spec['items']) > 1:
        return 'more than one earnings or debt item'
    return ''


def merge():
    a = {json.loads(l)['id']: json.loads(l) for l in open(HERE / 'reader_a.jsonl')}
    b = {json.loads(l)['case_id']: json.loads(l) for l in open(HERE / 'reader_b.jsonl')} if (HERE / 'reader_b.jsonl').exists() else {}
    kept, excluded = [], []
    for c in clauses():
        sa = (a.get(c['id']) or {}).get('spec')
        why = ''
        if not sa:
            why = 'reader A: no parseable spec'
        elif not sa.get('supported'):
            why = 'reader A: unsupported - ' + str(sa.get('reason', ''))[:150]
        else:
            sb = parse_json((b.get(c['id']) or {}).get('output', ''))
            try:
                if not sb:
                    why = 'reader B: no parseable output'
                elif not sb.get('supported', True):
                    why = 'reader B: unsupported - ' + str(sb.get('reason', ''))[:150]
                elif Counter(map(canon, sa['limits'])) != Counter(map(canon, sb.get('limits', []))):
                    why = 'readers disagree'
                else:
                    why = check_spec(sa, c['text'])
            except (KeyError, TypeError, ValueError) as e:
                why = f'malformed spec: {type(e).__name__}'
        if why:
            excluded.append(dict(id=c['id'], reason=why))
        else:
            kept.append(dict(id=c['id'], contract_id=c['contract_id'], borrower=c['borrower'], company=c['company'],
                             agreement_year=c['agreement_year'], text=c['text'], spec=sa))
    (HERE / 'specs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in kept))
    (HERE / 'specs_excluded.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in excluded))
    reasons = Counter(r['reason'].split(' - ')[0].split(':')[0] if r['reason'].startswith('reader') else r['reason'].split(' (')[0][:60] for r in excluded)
    summary = dict(clauses=len(kept) + len(excluded), kept=len(kept), kept_borrowers=len({r['borrower'] for r in kept}),
                   excluded=len(excluded), exclusion_reasons=dict(reasons.most_common()),
                   with_earnings=sum(any(i['role'] == 'earnings' for i in r['spec']['items']) for r in kept),
                   with_debt=sum(any(i['role'] == 'debt' for i in r['spec']['items']) for r in kept),
                   with_schedule=sum(any(len(l['schedule']) > 1 for l in r['spec']['limits']) for r in kept),
                   multi_limit=sum(len(r['spec']['limits']) > 1 for r in kept))
    (HERE / 'specs_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    {'read_a': read_a, 'read_b': read_b, 'merge': merge}[sys.argv[1]]()
