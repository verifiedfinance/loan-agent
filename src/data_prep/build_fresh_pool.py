"""pool_fresh.jsonl for exec_rl plan change 2: 1,000 financial_ratio clauses from TRAIN contracts that the SFT-789 start
never saw (not in pool_ratio.jsonl, not in sft_large/sft_pairs.jsonl), 200..2543 chars, seeded sample (2026).
Prompt format identical to pool_ratio.jsonl.
    python src/data_prep/build_fresh_pool.py
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import hashlib
import json
import random
import sys
from pathlib import Path

from parameter_semantics import student_prompt  # noqa: E402

ratio = [json.loads(l) for l in open(P.POOL_RATIO)]
template = {k: v for k, v in ratio[0].items() if k not in ('id', 'contract_id', 'category', 'text', 'prompt')}
used = {r['id'] for r in ratio} | {json.loads(l)['id'] for l in open(P.SFT_PAIRS)}
train = {json.loads(l)['contract_id'] for l in open(P.TRAIN)}
heldout = set()
for name in ('validation', 'test'):
    heldout |= {json.loads(l)['contract_id'] for l in open(P.SPLITS / f'{name}.jsonl')}
assert not train & heldout
cand = []
for line in open(P.PROCESSED / 'rl_rules.jsonl'):
    r = json.loads(line)
    if r.get('category') == 'financial_ratio' and r.get('contract_id') in train and r['id'] not in used \
            and 200 <= len(r['text']) <= 2543:
        cand.append({k: r[k] for k in ('id', 'contract_id', 'category', 'text')})
cand.sort(key=lambda r: r['id'])
rows = [{**r, 'prompt': student_prompt(r), **template} for r in random.Random(2026).sample(cand, 1000)]
assert not {r['id'] for r in rows} & used and not {r['contract_id'] for r in rows} & heldout
(P.POOL_FRESH).write_text(''.join(json.dumps(r) + '\n' for r in rows))
print(json.dumps(dict(candidates=len(cand), rows=len(rows),
                      sha256=hashlib.sha256((P.POOL_FRESH).read_bytes()).hexdigest())))
