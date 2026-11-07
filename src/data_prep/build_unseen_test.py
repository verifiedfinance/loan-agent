"""Borrower-disjoint, deduplicated test set (data audit fix 1, 2026-11-06).

The original splits separate contracts but not borrowers: 64% of test clauses come from a borrower that also has
agreements in train, and 114 of test_600 have a same-borrower near-copy in train. Every model was trained only on
datasets/train_unlabeled.jsonl (the SFT pairs and RL pools are subsets of it), so a test set whose borrowers never
appear in train is held out for all of them.

test_unseen = clauses from datasets/test.jsonl + datasets/validation.jsonl
  - minus data/splits/validation_250.jsonl (used as the RL held-out/dev set)
  - borrower (Compustat gvkey; EDGAR CIK when gvkey is missing) never appears in train_unlabeled
  - text not identical to any train clause after lower-casing, whitespace and number masking
  - no train clause (any borrower) with token-set Jaccard >= 0.7
Output: data/splits/test_unseen.jsonl (+ test_unseen_manifest.json with counts and sha256)

    python3 build_unseen_test.py
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import collections
import hashlib
import json
import re
from pathlib import Path

OUT = P.SPLITS
JACCARD = 0.7


def borrower_keys():
    control = json.load(open(P.RAW / 'main_control_file.json'))
    key, year, company = {}, {}, {}
    for v in control.values():
        cik = v['index_page'].split('/')[2] if v.get('index_page') else None
        g = v.get('gvkey')
        key[v['filehandle']] = f'g{g}' if g not in (None, '') else (f'c{cik}' if cik else None)
        year[v['filehandle']] = int(v['year']) if str(v.get('year', '')).isdigit() else None
        company[v['filehandle']] = v.get('company')
    return key, year, company


def main():
    rows = lambda p: [json.loads(l) for l in open(p)]
    train, val, test = rows(P.TRAIN), rows(P.VALIDATION), rows(P.TEST)
    key, year, company = borrower_keys()
    train_borrowers = {key.get(r['contract_id']) for r in train} - {None}
    dev = {r['id'] for r in rows(P.VALIDATION_250)}
    norm = lambda t: re.sub(r'\d[\d,.]*', '#', re.sub(r'\s+', ' ', t.lower())).strip()
    tok = lambda t: set(re.findall(r'[a-z]+', t.lower()))
    train_norm = {norm(r['text']) for r in train}
    train_tok = [tok(r['text']) for r in train]
    df = collections.Counter(w for s in train_tok for w in s)
    rare = lambda s: sorted(s, key=lambda w: (df[w], w))[:8]
    index = collections.defaultdict(list)
    for i, s in enumerate(train_tok):
        for w in rare(s):
            index[w].append(i)

    steps = collections.OrderedDict()
    pool = [r for r in val + test if r['id'] not in dev]
    steps['test + validation, minus validation_250'] = len(pool)
    pool = [r for r in pool if key.get(r['contract_id']) and key[r['contract_id']] not in train_borrowers]
    steps['borrower never in train'] = len(pool)
    pool = [r for r in pool if norm(r['text']) not in train_norm]
    steps['not a number-masked copy of a train clause'] = len(pool)
    kept = []
    for r in pool:
        s = tok(r['text'])
        cands = {i for w in rare(s) for i in index.get(w, [])}
        best = max((len(s & train_tok[i]) / len(s | train_tok[i]) for i in cands), default=0.0)
        if best < JACCARD:
            kept.append(dict(r, borrower=key[r['contract_id']], company=company.get(r['contract_id']),
                             agreement_year=year.get(r['contract_id']), max_train_jaccard=round(best, 3)))
    steps[f'no train clause with Jaccard >= {JACCARD}'] = len(kept)
    kept.sort(key=lambda r: r['id'])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'test_unseen.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in kept))
    t600 = {r['id'] for r in rows(P.TEST_600)}
    manifest = dict(steps=steps, clauses=len(kept), borrowers=len({r['borrower'] for r in kept}),
                    contracts=len({r['contract_id'] for r in kept}), in_test_600=sum(r['id'] in t600 for r in kept),
                    years=dict(sorted(collections.Counter((r['agreement_year'] or 0) // 5 * 5 for r in kept).items())),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT / 'test_unseen_manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
