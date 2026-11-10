"""Direct arm (PLAN.md): the model reads clause + certificate and answers the three questions as JSON.

    python src/headroom/run_direct.py claude claude-sonnet-5-5 claude_sonnet55
    python src/headroom/run_direct.py hf Qwen/Qwen3.8-27B qwen27b
Output: results/headroom/answers/direct_<name>.jsonl (resume-safe)
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SYSTEM = 'You are a careful credit analyst who checks borrowers against the financial covenants of their loan agreements.'
PROMPT = """Covenant clause from the borrower's credit agreement:

{clause}

{certificate}

Answer three questions about the test date, using only the clause and the figures above.
1. Status: "comply" if every financial covenant in the clause that applies on the test date is satisfied, "breach" if
   any is violated, "undetermined" if the answer depends on a figure that is not reported.
2. Earnings cushion: by what percentage could "{earnings}" fall, with every other figure unchanged, before any
   covenant is breached? Negative if already in breach. "none" if no covenant depends on it.
   "undetermined" if it cannot be computed.
3. Debt capacity: by how many US dollars could "{debt}" increase, with every other figure unchanged, before any
   covenant is breached? Negative if already in breach. "none" if no covenant limits it. "undetermined" if it cannot
   be computed (a needed figure is missing, or no amount of debt restores compliance).
Ratios with a zero or negative denominator count as breached. Work it out briefly, then end with one line of JSON:
{{"status": "...", "earnings_cushion_pct": <number or "none" or "undetermined">, "debt_capacity_usd": <number or "none" or "undetermined">}}"""
NA = 'not asked (answer "none")'


def prompt(case, clause):
    desc = {i['name']: i['description'] for i in case['_items']}
    return PROMPT.format(clause=clause, certificate=case['certificate'],
                         earnings=desc.get(case['earnings_item'], NA), debt=desc.get(case['debt_item'], NA))


def parse(text):
    for m in reversed(re.findall(r'\{[^{}]*"status"[^{}]*\}', text or '', re.S)):
        try:
            d = json.loads(m)
        except json.JSONDecodeError:
            continue
        def val(x):
            if isinstance(x, str):
                x = x.strip().lower()
                if x in ('none', 'undetermined'):
                    return x
                try:
                    return float(x.replace(',', '').replace('$', '').replace('%', ''))
                except ValueError:
                    return None
            return float(x) if isinstance(x, (int, float)) else None
        return dict(status=str(d.get('status', '')).strip().lower(), cushion=val(d.get('earnings_cushion_pct')),
                    capacity=val(d.get('debt_capacity_usd')))
    return None


def load_cases():
    specs = {json.loads(l)['id']: json.loads(l) for l in open(P.HEADROOM_DATA / 'specs.jsonl')}
    cases = [json.loads(l) for l in open(P.HEADROOM_DATA / 'cases.jsonl')]
    for c in cases:
        c['_items'] = specs[c['clause_id']]['spec']['items']
    return cases, specs


def main():
    backend, model, name = sys.argv[1:4]
    cases, specs = load_cases()
    out = P.HEADROOM_RESULTS / 'answers' / f'direct_{name}.jsonl'
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(l)['case_id'] for l in open(out)} if out.exists() else set()
    todo = [c for c in cases if c['case_id'] not in done]
    if backend == 'claude':
        from verifiqa.generation.llm_client import ChatMessage, ClaudeCliClient
        client = ClaudeCliClient(model=model, effort='medium', timeout=300)

        def one(c):
            for _ in range(2):
                try:
                    t = client.chat([ChatMessage('system', SYSTEM), ChatMessage('user', prompt(c, specs[c['clause_id']]['text']))],
                                    stage='headroom_direct')
                    return dict(case_id=c['case_id'], output=t, answer=parse(t))
                except Exception as e:  # noqa: BLE001
                    err = str(e)[:200]
            return dict(case_id=c['case_id'], error=err)
        with ThreadPoolExecutor(8) as ex, open(out, 'a') as h:
            for i, r in enumerate(ex.map(one, todo), 1):
                if 'error' in r:
                    print('FAILED', r['case_id'], r['error'], flush=True)
                    continue
                h.write(json.dumps(r) + '\n')
                h.flush()
                if i % 50 == 0:
                    print(f'{i}/{len(todo)}', flush=True)
    else:
        import hf_generation as wf
        items = [dict(case_id=c['case_id'], clause_id=c['clause_id'], messages=[
            {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt(c, specs[c['clause_id']]['text'])}]) for c in todo]
        tok, mdl = wf.load(model)
        tmp = out.with_suffix('.raw.jsonl')
        wf.run(tok, mdl, model, items, tmp, int(__import__('os').environ.get('READER_BATCH', 16)), 1024, 8192)  # READER_BATCH: smaller batches for the 70B reader if memory runs short
        with open(out, 'a') as h:
            for l in open(tmp):
                r = json.loads(l)
                if r['case_id'] not in done:
                    h.write(json.dumps(dict(case_id=r['case_id'], output=r['output'], answer=parse(r['output']), error=r.get('error', ''))) + '\n')
    print('DIRECT_DONE', name, flush=True)


if __name__ == '__main__':
    main()
