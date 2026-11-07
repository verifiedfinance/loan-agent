"""Step 3 of the downstream pipeline: reference rules for the sampled covenants (no human labels).

Two independent sources, kept only where they agree:
  specs.SPECS  a structured reading of each clause (written by the assistant, an LLM)
  extract()    a deterministic pattern extractor over the raw clause text: every (direction,
               threshold) limit it can find, with the words that set the direction
A clause is `agreed` when every limit in its spec is found by the extractor with the same direction
(strict vs non-strict included) and threshold. Conditions (`when` flags) rest on the reading alone.
Clauses whose covenant text (preamble removed) duplicates an earlier agreed clause are dropped.

The reference decides comply/breach for a case: covenant_holds(spec, values, flags).
Assumed in every case sheet: the agreement is in force (commitments or obligations outstanding),
no consent or waiver has been given, and the date is a test date.

Outputs (results/downstream/):
  reference_rules.csv      one row per limit of every sampled clause that has a spec
  reference_agreement.csv  one row per sampled clause: status (included / excluded: reason /
                           extractor disagreement / duplicate) and details

    python -I downstream/reference_rules.py
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import csv
import json
import re
import sys
from pathlib import Path

ROOT = P.ROOT
from specs import SPECS, EXCLUDED  # noqa: E402

OPS = {'<=': lambda a, b: a <= b, '<': lambda a, b: a < b, '>=': lambda a, b: a >= b, '>': lambda a, b: a > b}


def covenant_holds(spec, values, flags):
    """Reference verdict: True = comply. Limits whose `when` flag does not match are not tested."""
    for lim in spec:
        when = lim.get('when')
        if when and bool(flags.get(when[0])) != when[1]:
            continue
        if not OPS[lim['op']](values[lim['metric']], lim['threshold']):
            return False
    return True


# ---------------------------------------------------------------- independent extractor
# Direction phrases mapped to the comply condition; the phrase closest before a threshold decides.
# Explicit phrases ("not greater than", "at least", "at no time exceed") mean the same in any sentence.
# A bare comparison ("less than or equal to", "exceed") is what is PROHIBITED after "permit"/"suffer"
# ("shall not permit X to exceed T": comply iff X <= T) and what is REQUIRED after "maintain", "shall
# have", "cause", "shall be" ("maintain X less than or equal to T": comply iff X <= T). Bare phrases
# therefore carry both readings: (prohibited, required).
W = r'(?:\w+ ){0,3}'
PHRASES = [
    (rf'not {W}less than|no less than|at least|minimum', '>='),
    (rf'not {W}greater than|no greater than|not {W}more than|no more than|not {W}in excess of|not (?:to )?exceed|maximum', '<='),
    (rf'(?:no time|never) {W}exceed|(?:no time|never) {W}greater than', '<='), (rf'(?:no time|never) {W}less than', '>='),
    (r'equal to or (?:in excess of|greater than)', ('<', '>=')), (r'equal to or less than', ('>', '<=')),
    (r'greater than or equal to', ('<', '>=')), (r'less than or equal to', ('>', '<=')),
    (r'exceed|greater than|higher than', ('<=', '>')),
    (r'less than', ('>=', '<')),
]
PROHIBIT = r'\bpermit\b|\bsuffer\b|\ballow\b'
REQUIRE = r'\bmaintain\b|\b(?:shall|will|must) have\b|\bcause\b|\b(?:shall|will|must)(?: \w+){0,3} be\b'


def resolve(op, governing):
    """Pick the reading of a bare comparison from the nearest governing verb before it."""
    if isinstance(op, str):
        return op
    verbs = [(m.end(), 'prohibit') for m in re.finditer(PROHIBIT, governing)] + \
            [(m.end(), 'require') for m in re.finditer(REQUIRE, governing)]
    mode = max(verbs)[1] if verbs else 'prohibit'
    return op[0] if mode == 'prohibit' else op[1]


NUM = (r'(?P<pct>\d*\.?\d+)\s*%'                                 # 45%, .60%
       r'|(?P<ratio>\d*\.?\d+)\s*(?:to|:)\s*1(?:\.0+)?(?!\d)'      # 3.50 to 1.00, 3.50:1.00, .65 to 1.0
       r'|\$\s?(?P<usd>\d{1,3}(?:,\d{3})+)')                       # $500,000,000



def extract(text):
    found = []
    for m in re.finditer(NUM, text):
        if m.group('pct'):
            value = float(m.group('pct')) / 100
        elif m.group('ratio'):
            value = float(m.group('ratio'))
        else:
            value = float(m.group('usd').replace(',', ''))
        window = text[max(0, m.start() - 160):m.start()].lower()
        best = None
        for pattern, op in PHRASES:
            for p in re.finditer(pattern, window):
                if best is None or p.end() > best[0] or (p.end() == best[0] and p.start() < best[2]):
                    best = (p.end(), op, p.start(), p.group(0))
        if best and len(window) - best[0] <= 120:
            governing = text[max(0, m.start() - 400):m.start()].lower()
            governing = governing[:len(governing) - (len(window) - best[2])]  # verbs before the phrase only
            found.append(dict(op=resolve(best[1], governing), threshold=value, phrase=best[3], at=m.start()))
    return found


def core_text(text):
    """Covenant text with the article preamble removed, normalized, for duplicate detection."""
    m = re.search(r':\s{1,3}(?=[A-Z0-9#*(\[])', text)
    body = text[m.end():] if m and m.start() < 1500 else text
    return re.sub(r'\W+', ' ', body).strip().lower()


def main():
    texts = {}
    for line in open(P.TEST_600):
        r = json.loads(line)
        texts[r['id']] = r['text']
    pool = [r for r in csv.DictReader(open(ROOT / 'results/downstream/clause_pool.csv')) if r['sampled'] == 'yes']
    sampled = [r['id'] for r in pool]
    missing = set(sampled) - set(SPECS) - set(EXCLUDED)
    assert not missing, f'sampled clauses without a spec or exclusion: {sorted(missing)}'
    assert not set(SPECS) & set(EXCLUDED)

    rules, agreement, seen = [], [], {}
    for cid in sampled:
        base = dict(id=cid, has_xbrl=next(r['has_xbrl'] for r in pool if r['id'] == cid))
        if cid in EXCLUDED:
            agreement.append({**base, 'status': 'excluded', 'limits': 0, 'conditional': '', 'detail': EXCLUDED[cid], 'extractor_extra_limits': ''})
            continue
        spec, text = SPECS[cid], texts[cid]
        got = extract(text)
        pairs = {(g['op'], round(g['threshold'], 6)) for g in got}
        problems = []
        for k, lim in enumerate(spec):
            hit = (lim['op'], round(lim['threshold'], 6)) in pairs
            if not hit:
                near = [g for g in got if abs(g['threshold'] - lim['threshold']) < 1e-9]
                problems.append(f"limit {k} {lim['op']} {lim['threshold']:g}: "
                                + (f"extractor reads {near[0]['op']} ({near[0]['phrase']!r})" if near else 'threshold not found'))
            rules.append(dict(id=cid, limit=k, metric=lim['metric'], label=lim['label'], comply_if=f"{lim['op']} {lim['threshold']:g}",
                              op=lim['op'], threshold=lim['threshold'], unit=lim['unit'],
                              condition=f"{lim['when'][0]} = {lim['when'][1]}" if lim.get('when') else '',
                              extractor_agrees='yes' if hit else 'no'))
        spec_pairs = {(l['op'], round(l['threshold'], 6)) for l in spec}
        extra = '; '.join(f"{g['op']} {g['threshold']:g} ({g['phrase']!r})" for g in got
                          if (g['op'], round(g['threshold'], 6)) not in spec_pairs)
        key = core_text(text)
        if problems:
            status, detail = 'extractor disagreement', '; '.join(problems)
        elif key in seen:
            status, detail = 'duplicate', f'same covenant text as {seen[key]}'
        else:
            status, detail = 'included', ''
            seen[key] = cid
        agreement.append({**base, 'status': status, 'limits': len(spec),
                          'conditional': 'yes' if any(l.get('when') for l in spec) else 'no',
                          'detail': detail, 'extractor_extra_limits': extra})

    out = ROOT / 'results/downstream'
    for name, rows in (('reference_rules.csv', rules), ('reference_agreement.csv', agreement)):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    from collections import Counter
    inc = [a for a in agreement if a['status'] == 'included']
    print(f"sampled {len(sampled)}: {dict(Counter(a['status'] for a in agreement))}")
    print(f"included {len(inc)} clauses, {sum(a['limits'] for a in inc)} limits; conditional {sum(a['conditional'] == 'yes' for a in inc)}; "
          f"with XBRL {sum(a['has_xbrl'] == 'yes' for a in inc)}")
    for a in agreement:
        if a['status'] in ('extractor disagreement', 'duplicate'):
            print(f"  {a['id']:11s} {a['status']}: {a['detail']}")


def included_specs():
    """Specs of the clauses that passed agreement (for the case builder and the pipeline)."""
    rows = csv.DictReader(open(ROOT / 'results/downstream/reference_agreement.csv'))
    return {r['id']: SPECS[r['id']] for r in rows if r['status'] == 'included'}


if __name__ == '__main__':
    main()
