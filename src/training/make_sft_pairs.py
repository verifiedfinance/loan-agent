#!/usr/bin/env python3
"""Labeled SFT data for the fair SFT baseline: teacher pairs on the SAME 1,000 clauses GRPO trained on.

Comparison (as in FormaRL): RL without labels (GRPO, run_main) vs SFT with labels on the same clauses
and the same base model. Labels are teacher formalizations kept only when they pass every filter of the
original SFT data (experiments/sft_parameter_semantics): Z3-valid in the supported fragment, complete
parameter comments, a symbol table quoting the source exactly, and acceptance by a separate judge call.

The frozen generator (generate_smt_pairs.generate_one, unchanged) is reused. Only the model transport
differs: Sol is no longer available through Codex, so both teacher and judge calls go to Claude through
the logged-in `claude -p` CLI (ClaudeCliClient). Teacher and judge are separate, isolated calls.

    python src/training/make_sft_pairs.py

Outputs here: generation_audit.jsonl (every attempt), sft_pairs.jsonl (accepted pairs that fit 4,096
tokens, in the student's chat format), generation_summary.json. Resume-safe.
"""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

OUT = P.SFT_DATA

import verifiqa.generation.llm_client as llm  # noqa: E402

TEACHER = JUDGE = 'claude-sonnet-5-5'
EFFORT, WORKERS, MAX_SEQUENCE_TOKENS = 'medium', 6, 4096


def _claude(model, reasoning_effort='medium', timeout=240):
    return llm.ClaudeCliClient(model=model, effort=reasoning_effort, timeout=300)


llm.CodexCliClient = _claude  # route the frozen generator's calls to the Claude CLI (inherited by forked workers)

from generate_smt_pairs import generate_one, read  # noqa: E402
from parameter_semantics import VERSION, student_prompt  # noqa: E402


def main():
    from transformers import AutoTokenizer
    # the 1,000 RL clauses of the first GRPO run, in its order (data/rl/pool_ratio.jsonl; identical ids, order,
    # text and prompts to that run's run_config.json record_ids over its grpo_train_pm.jsonl)
    pool_rows = read(P.POOL_RATIO)
    ids = [r['id'] for r in pool_rows]
    pool = {r['id']: r for r in pool_rows}
    records = [pool[i] for i in ids]
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')
    path = OUT / 'generation_audit.jsonl'
    done = {r['id']: r for r in read(path)} if path.exists() else {}

    def publish():
        order = {i: k for k, i in enumerate(ids)}
        accepted = sorted([r for r in done.values() if r.get('training_eligible')], key=lambda r: order[r['id']])
        pairs = [{'id': a['id'], 'contract_id': pool[a['id']]['contract_id'], 'data_origin': 'teacher_generated_real_covenant',
                  'parameter_semantics': VERSION,
                  'messages': student_prompt(pool[a['id']]) + [{'role': 'assistant', 'content': a['candidate']['smt2']}],
                  'verification': {'teacher_model': TEACHER, 'judge_model': JUDGE, 'z3': a['z3'], 'semantic_judge': a['judge'],
                                   'parameter_comments': a['parameter_comment_coverage']}} for a in accepted]
        (OUT / 'sft_pairs.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in pairs))
        from collections import Counter
        summary = {'source': 'data/rl/pool_ratio.jsonl (the GRPO training clauses)',
                   'clauses': len(ids), 'attempted': len(done), 'accepted_and_fits_context': len(pairs),
                   'rejections': dict(Counter(r.get('rejection', 'error' if 'error' in r else 'accepted') for r in done.values())),
                   'teacher': TEACHER, 'judge': JUDGE, 'transport': 'claude_cli', 'reasoning_effort': EFFORT,
                   'max_sequence_tokens': MAX_SEQUENCE_TOKENS}
        (OUT / 'generation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary

    publish()
    todo = [r for r in records if r['id'] not in done or 'error' in done[r['id']]]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(generate_one, r, 'train', TEACHER, JUDGE, EFFORT, True): r for r in todo}
        for fut in as_completed(futures):
            audit = fut.result()
            audit['training_eligible'] = False
            if audit['accepted']:
                messages = student_prompt(pool[audit['id']]) + [{'role': 'assistant', 'content': audit['candidate']['smt2']}]
                audit['sequence_tokens'] = len(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False))
                audit['training_eligible'] = audit['sequence_tokens'] <= MAX_SEQUENCE_TOKENS
            done[audit['id']] = audit
            with path.open('a') as handle:
                handle.write(json.dumps(audit) + '\n')
            s = publish()
            print(f"{s['attempted']}/{len(ids)} attempted; {s['accepted_and_fits_context']} accepted: {audit['id']} "
                  f"{audit.get('rejection', audit.get('error', 'accepted'))}", flush=True)
    print(json.dumps(publish(), indent=2))
    print('SFT_LARGE_GENERATION_DONE')


if __name__ == '__main__':
    main()
