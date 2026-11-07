"""Training prompts for headroom RL v2 (PLAN.md): reader A (Claude) names the line items of each of the 1,000 unseen
training clauses (data/rl/pool_fresh.jsonl, train borrowers only, disjoint from test_unseen); the prompt
is the formula-arm prompt with that parameter list. Only clauses without date schedules or condition flags are kept
(the execution reward cannot score those fairly with an open test date / flag).

    python src/headroom/rl_v2_pool.py read      # Claude, resume-safe -> data/rl/rl_v2/reader_a.jsonl
    python src/headroom/rl_v2_pool.py build     # -> data/rl/rl_v2/pool.jsonl, pool_summary.json
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import json
import sys
from collections import Counter
from pathlib import Path

OUT = P.RL_DATA / 'rl_v2'  # reader_a.jsonl, pool.jsonl, pool_summary.json
import build_specs as bs  # noqa: E402
import headroom as hr  # noqa: E402
from run_formula import PROMPT, SYSTEM, params_text  # noqa: E402

POOL = P.POOL_FRESH


def pool():
    return [json.loads(l) for l in open(POOL)]


def read():
    OUT.mkdir(exist_ok=True)
    bs.clauses = pool
    bs.HERE = OUT
    bs.read_a()


def build():
    a = {json.loads(l)['id']: json.loads(l) for l in open(OUT / 'reader_a.jsonl')}
    test_borrowers = {json.loads(l)['borrower'] for l in open(bs.CLAUSES)}
    keep, why = [], Counter()
    for c in pool():
        s = (a.get(c['id']) or {}).get('spec')
        if not s or not s.get('supported'):
            why['unsupported or unparseable'] += 1
            continue
        try:
            reason = bs.check_spec(s, c['text'])
        except (KeyError, TypeError, ValueError):
            reason = 'malformed spec'
        if reason and not reason.startswith(('no earnings or debt item', 'earnings/debt item', 'more than one earnings')):
            why[reason.split(' (')[0][:50]] += 1
            continue
        if s.get('flags') or any(len(l['schedule']) != 1 or l['schedule'][0].get('to') for l in s['limits']):
            why['schedule or flag'] += 1
            continue
        spec = dict(items=s['items'], flags=[], limits=s['limits'])
        keep.append(dict(id=c['id'], contract_id=c['contract_id'], text=c['text'],
                         prompt=[{'role': 'system', 'content': SYSTEM},
                                 {'role': 'user', 'content': PROMPT.format(clause=c['text'].strip(), params=params_text(spec))}],
                         params=[list(p) for p in hr.params(spec)]))
    (OUT / 'pool.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in keep))
    summary = dict(pool=len(pool()), kept=len(keep), excluded=dict(why.most_common()),
                   test_borrowers_overlap='none by construction (pool_fresh is train borrowers; test_unseen excludes them)')
    (OUT / 'pool_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    {'read': read, 'build': build}[sys.argv[1]]()
