"""Sensitivity (PLAN.md): regenerate direct-arm answers that hit the 1,024-token limit with a 4,096-token limit.

    python src/headroom/rerun_truncated.py Qwen/Qwen2.5-Coder-32B-Instruct qwen32b
Output: results/headroom/answers_long/direct_<name>.jsonl (cut-off cases regenerated, all others copied)
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import json
import sys
from pathlib import Path

import run_direct as rd  # noqa: E402


def main():
    model, name = sys.argv[1:3]
    cases, specs = rd.load_cases()
    by = {c['case_id']: c for c in cases}
    rows = [json.loads(l) for l in open(P.HEADROOM_RESULTS / 'answers' / f'direct_{name}.jsonl')]
    cut = [r['case_id'] for r in rows if r.get('error') == 'hit_token_limit']
    print(f'{name}: {len(cut)} cut-off answers to regenerate', flush=True)
    out_dir = P.HEADROOM_RESULTS / 'answers_long'
    out_dir.mkdir(exist_ok=True)
    raw = out_dir / f'direct_{name}.raw.jsonl'
    if cut:
        import hf_generation as wf
        items = [dict(case_id=k, clause_id=by[k]['clause_id'], messages=[{'role': 'system', 'content': rd.SYSTEM},
                 {'role': 'user', 'content': rd.prompt(by[k], specs[by[k]['clause_id']]['text'])}]) for k in cut]
        tok, mdl = wf.load(model)
        wf.run(tok, mdl, model, items, raw, max(1, int(__import__('os').environ.get('READER_BATCH', 16)) // 2), 4096, 8192)  # 8 by default; halves READER_BATCH
    new = {json.loads(l)['case_id']: json.loads(l) for l in open(raw)} if raw.exists() else {}
    with open(out_dir / f'direct_{name}.jsonl', 'w') as h:
        for r in rows:
            if r['case_id'] in new:
                n = new[r['case_id']]
                r = dict(case_id=r['case_id'], output=n['output'], answer=rd.parse(n['output']), error=n.get('error', ''), regenerated=True)
            h.write(json.dumps(r) + '\n')
    still = sum(new[k].get('error') == 'hit_token_limit' for k in new)
    print(f'RERUN_DONE {name}: regenerated {len(new)}, still cut off at 4,096: {still}', flush=True)


if __name__ == '__main__':
    main()
