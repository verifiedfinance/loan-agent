"""Formula arm (PLAN.md): the model writes covenant_holds over exactly the certificate's parameters (one formula per
clause); facts come straight from the certificate; Z3 answers the three questions (headroom.FormulaEvaluator).

    python src/headroom/run_formula.py models/sft789/final sft789
    python src/headroom/run_formula.py models/rl_v1/final rl789_s42
    python src/headroom/run_formula.py Qwen/Qwen2.5-Coder-7B-Instruct qwen7b_prompted
Output: results/headroom/formulas/<name>.jsonl, results/headroom/answers/formula_<name>.jsonl
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import headroom as hr  # noqa: E402

SYSTEM = ('Translate the real financial covenant into SMT-LIB2. Return exactly a set-logic command and one '
          'define-fun named covenant_holds that returns Bool and is true exactly when the borrower complies.')
PROMPT = """Covenant clause:

{clause}

Write (define-fun covenant_holds (<parameters>) Bool <body>) with EXACTLY these parameters, in this order and with
these types (do not add, rename or drop any):
{params}
Amounts are in US dollars as reported in the borrower's compliance certificate; test_date is the test date as an
integer YYYYMMDD (e.g. 20121231), so a threshold that changes over time is written with test_date comparisons.
covenant_holds is true exactly when every financial covenant in the clause that applies on test_date is satisfied.
Return only the SMT-LIB2 (set-logic QF_NRA or QF_LIA/QF_LRA as needed, then the define-fun)."""


def params_text(spec):
    lines = [f"({i['name']} Real) ; {i['description']}" for i in spec['items']]
    lines.append('(test_date Int) ; test date as YYYYMMDD')
    lines += [f"({f['name']} Bool) ; {f['description']}" for f in spec['flags']]
    return '\n'.join(lines)


def extract_smt(text):
    text = re.sub(r'```(?:smt2?|smtlib2?|lisp|scheme)?', '', text or '')
    m = re.search(r'\(\s*define-fun\s+covenant_holds', text)
    return text[m.start():].strip() if m else text.strip()


def main():
    model, name = sys.argv[1:3]
    lenient = '--lenient' in sys.argv  # post-hoc sensitivity (PLAN.md): reuse the formulas, accept a typed parameter subset
    specs = {json.loads(l)['id']: json.loads(l) for l in open(P.HEADROOM_DATA / 'specs.jsonl')}
    cases = [json.loads(l) for l in open(P.HEADROOM_DATA / 'cases.jsonl')]
    fpath = P.HEADROOM_RESULTS / 'formulas' / f'{name}.jsonl'
    fpath.parent.mkdir(exist_ok=True)
    if lenient and not fpath.exists():
        sys.exit('lenient scoring reuses existing formulas; run the strict pass first')
    if not lenient and (not fpath.exists() or sum(1 for _ in open(fpath)) < len(specs)):
        import hf_generation as wf
        items = [dict(case_id=k, clause_id=k, messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content':
                 PROMPT.format(clause=r['text'].strip(), params=params_text(r['spec']))}]) for k, r in specs.items()]
        tok, mdl = wf.load(model)
        wf.run(tok, mdl, model, items, fpath, 8, 1024, 6144)
        del mdl
    formulas = {json.loads(l)['case_id']: json.loads(l) for l in open(fpath)}
    out = P.HEADROOM_RESULTS / ('answers_lenient' if lenient else 'answers') / f'formula_{name}.jsonl'
    out.parent.mkdir(exist_ok=True)
    evs, errs = {}, {}
    for k, r in specs.items():
        f = formulas.get(k)
        try:
            if not f or f.get('error'):
                raise ValueError(f.get('error') if f else 'no formula')
            evs[k] = hr.FormulaEvaluator(extract_smt(f['output']), r['spec'], lenient=lenient)
        except Exception as e:  # noqa: BLE001
            errs[k] = f'{type(e).__name__}: {str(e)[:150]}'
    with open(out, 'w') as h:
        for c in cases:
            k = c['clause_id']
            if k in errs:
                ans, err = dict(status='error', cushion='error', capacity='error'), errs[k]
            else:
                vals = {n: (None if v is None else Fraction(v)) for n, v in c['values'].items()}
                try:
                    ans, err = evs[k].answers(specs[k]['spec'], vals, c['test_date'], c['flags']), ''
                except Exception as e:  # noqa: BLE001
                    ans, err = dict(status='error', cushion='error', capacity='error'), f'{type(e).__name__}: {str(e)[:150]}'
            h.write(json.dumps(dict(case_id=c['case_id'], answer=ans, error=err)) + '\n')
    print(f'FORMULA_DONE {name}: {len(specs) - len(errs)}/{len(specs)} formulas usable', flush=True)


if __name__ == '__main__':
    main()
