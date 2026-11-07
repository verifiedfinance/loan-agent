#!/usr/bin/env python3
"""Generate/resume actual teacher pairs; publish only Z3 + judge accepted targets."""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import re
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
from autoformalization import PROTOCOL, TEACHER, JUDGE, source_payload, student_prompt, validate_smt, judge_accepts
from build_rl_datasets import write_jsonl
from evaluate_smt_predictions import save_references


def request(client, system, content, stage):
    from verifiqa.generation.llm_client import ChatMessage
    return client.chat([ChatMessage('system', system), ChatMessage('user', json.dumps(content, ensure_ascii=False))], stage=stage)


def generate_one(record, split, teacher_model, judge_model, effort, parameter_comments=False):
    from verifiqa.generation.llm_client import CodexCliClient
    teacher_prompt, judge_prompt = TEACHER, JUDGE
    if parameter_comments:
        from parameter_semantics import with_parameter_semantics, comment_coverage
        teacher_prompt = with_parameter_semantics(TEACHER) + '\nThe symbols table and SMT comments must express the same meanings.'
        judge_prompt = with_parameter_semantics(JUDGE)
    digest = hashlib.sha256(json.dumps(source_payload(record), sort_keys=True).encode()).hexdigest()
    audit = {'id': record['id'], 'split': split, 'source_sha256': digest,
             'teacher_model': teacher_model, 'judge_model': judge_model,
             'judge_independence': 'separate_isolated_call_same_model' if teacher_model == judge_model else 'separate_model',
             'accepted': False, 'reasoning_effort': effort, 'formalization_protocol': PROTOCOL,
             'source_context': record.get('source_context', []),
             'declared_assumptions': record.get('declared_assumptions', []),
             'teacher_prompt_sha256': hashlib.sha256(teacher_prompt.encode()).hexdigest(),
             'judge_prompt_sha256': hashlib.sha256(judge_prompt.encode()).hexdigest()}
    if parameter_comments:
        audit['parameter_comments'] = True
    try:
        teacher = CodexCliClient(model=teacher_model, reasoning_effort=effort, timeout=240)
        audit['teacher_raw'] = request(teacher, teacher_prompt, source_payload(record), 'covenant_teacher')
        candidate = json.loads(audit['teacher_raw'])
        audit['candidate'] = candidate
        if candidate.get('status') != 'formalized':
            audit['rejection'] = 'teacher_needs_context'
            return audit
        validation = validate_smt(candidate.get('smt2'))
        audit['z3'] = validation
        if not validation['valid']:
            audit['rejection'] = 'z3_or_supported_fragment'
            return audit
        if parameter_comments:
            audit['parameter_comment_coverage'] = comment_coverage(candidate['smt2'], validation['parameters'])
            if not audit['parameter_comment_coverage']['complete']:
                audit['rejection'] = 'missing_or_duplicate_parameter_comments'
                return audit
        symbols = candidate.get('symbols')
        if (not isinstance(symbols, list) or not symbols
                or any(not isinstance(s, dict) or not s.get('meaning')
                       or not isinstance(s.get('source_quote'), str) or not s['source_quote']
                       or s['source_quote'] not in '\n'.join([record['text']] + [x['text'] if isinstance(x, dict) else x for x in record.get('source_context', [])]) for s in symbols)
                or sorted(s.get('name', '') for s in symbols) != validation['symbols']):
            audit['rejection'] = 'ungrounded_symbol_table'
            return audit
        judge = CodexCliClient(model=judge_model, reasoning_effort=effort, timeout=240)
        # Do not show the teacher rationale, status, or identity to the judge.
        audit['judge_raw'] = request(judge, judge_prompt, {
            **source_payload(record), 'candidate_smt2': candidate['smt2']}, 'covenant_independent_judge')
        audit['judge'] = json.loads(audit['judge_raw'])
        audit['accepted'] = judge_accepts(audit['judge'])
        if not audit['accepted']:
            audit['rejection'] = 'semantic_review'
    except Exception as exc:
        audit['error'] = f'{type(exc).__name__}: {exc}'
    return audit


def publish(datasets, audits, source_dir=None):
    """Write SFT targets and separate references; never rewrite source split files."""
    source_dir = source_dir or datasets
    sources = {split: {r['id']: r for r in read(source_dir / f'{filename}.jsonl')}
               for split, filename in [('train', 'train_unlabeled'), ('validation', 'validation'), ('test', 'test')]}
    sft, references = [], {'validation': [], 'test': []}
    for audit in audits.values():
        row = dict(sources[audit['split']][audit['id']])
        row.update(source_context=audit.get('source_context', []),
                   declared_assumptions=audit.get('declared_assumptions', []))
        digest = hashlib.sha256(json.dumps(source_payload(row), sort_keys=True).encode()).hexdigest()
        if digest != audit['source_sha256']:
            raise ValueError(f'Source changed since generation: {row["id"]}')
        verification = {'z3': audit.get('z3'), 'semantic_judge': audit.get('judge'),
                        'teacher_model': audit['teacher_model'], 'judge_model': audit['judge_model'],
                        'judge_independence': audit['judge_independence']}
        if audit['split'] == 'train':
            if audit['accepted']:
                prompt_builder = student_prompt
                if audit.get('parameter_comments'):
                    from parameter_semantics import student_prompt as prompt_builder
                sft.append({'id': row['id'], 'contract_id': row['contract_id'],
                            'data_origin': 'teacher_generated_real_covenant', 'formalization_protocol': PROTOCOL,
                            'messages': prompt_builder(row) + [{'role': 'assistant', 'content': audit['candidate']['smt2']}],
                            'verification': verification})
        else:
            references[audit['split']].append({
                'id': row['id'], 'contract_id': row['contract_id'], 'formalization_protocol': PROTOCOL,
                'source_text_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
                'source_context': row['source_context'], 'declared_assumptions': row['declared_assumptions'],
                'reference_smt2': audit['candidate']['smt2'] if audit['accepted'] else None,
                'reference_status': 'model_reviewed_not_gold' if audit['accepted'] else 'verification_failed_or_needs_context',
                'reference_verification': verification, 'semantic_test_cases': []})
    write_jsonl(datasets / 'sft_synthetic.jsonl', sorted(sft, key=lambda row: row['id']))
    for split, rows in references.items():
        save_references(datasets / 'references' / PROTOCOL / f'{split}.jsonl', rows)
    summary = {'formalization_protocol': PROTOCOL, 'attempted': len(audits),
               'accepted_sft_pairs': len(sft),
               'accepted_by_split': {split: sum(a['split'] == split and a.get('accepted', False) for a in audits.values())
                                     for split in ('train', 'validation', 'test')},
               'errors': sum('error' in a for a in audits.values()),
               'reference_directory': str(datasets / 'references' / PROTOCOL)}
    (datasets / 'generation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def read(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-model', default='gpt-6-sol')
    parser.add_argument('--judge-model', default='gpt-6-astra')
    parser.add_argument('--reasoning-effort', default='medium', choices=['low', 'medium', 'high', 'xhigh'])
    parser.add_argument('--train-limit', type=int, default=10)
    parser.add_argument('--accepted-train-target', type=int, help='Stop scheduling training records once this many accepted pairs exist')
    parser.add_argument('--eval-limit', type=int, default=3)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--selection', choices=['shortest', 'concise', 'hash'], default='hash')
    parser.add_argument('--parameter-comments', action='store_true', help='Apply the shared source-grounded parameter-comment rules')
    parser.add_argument('--output-dir', type=Path, help='Write a separate generation experiment; source splits stay in datasets/')
    args = parser.parse_args()
    if min(args.train_limit, args.eval_limit) < 0 or args.workers < 1:
        parser.error('Limits must be nonnegative and workers positive')
    if args.accepted_train_target is not None and args.accepted_train_target < 1:
        parser.error('--accepted-train-target must be positive')
    datasets = P.SPLITS
    output = args.output_dir or datasets
    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'generation_audit.jsonl'
    audits = {f'{a["split"]}:{a["id"]}': a for a in read(cache)} if cache.exists() else {}
    for audit in audits.values():
        if audit.get('formalization_protocol') != PROTOCOL or audit['teacher_model'] != args.teacher_model or audit['judge_model'] != args.judge_model:
            raise ValueError('Existing audit uses different models; use a separate experiment directory')
        if bool(audit.get('parameter_comments')) != args.parameter_comments:
            raise ValueError('Parameter-comment convention changed; use --output-dir for a separate experiment')
    jobs = []
    for split, filename, limit in [('train', 'train_unlabeled', args.train_limit),
                                    ('validation', 'validation', args.eval_limit), ('test', 'test', args.eval_limit)]:
        rows = read(datasets / f'{filename}.jsonl')
        if args.selection == 'concise':
            rows = [r for r in rows if 160 <= len(r['text']) <= 1400
                    and re.search(r'\b(shall|will|must|maintain)\b', r['text'], re.I)
                    and re.search(r'\b(exceed|exceeding|less than|greater than|at least|not more than)\b', r['text'], re.I)]
        key = (lambda r: (len(r['text']), r['id'])) if args.selection in ('shortest', 'concise') else (lambda r: hashlib.sha256(r['id'].encode()).hexdigest())
        for row in sorted(rows, key=key)[:limit]:
            if f'{split}:{row["id"]}' not in audits:
                jobs.append((row, split))
    (output / 'generation_config.json').write_text(json.dumps(vars(args), indent=2, default=str) + '\n')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = iter(jobs)
        pending = set()
        exhausted = False
        accepted = sum(a['split'] == 'train' and a['accepted'] for a in audits.values())
        while pending or not exhausted:
            while len(pending) < args.workers and not exhausted:
                job = next(jobs, None)
                if job is None:
                    exhausted = True
                    break
                row, split = job
                if split == 'train' and args.accepted_train_target is not None and accepted >= args.accepted_train_target:
                    continue
                options = {'parameter_comments': True} if args.parameter_comments else {}
                pending.add(pool.submit(generate_one, row, split, args.teacher_model, args.judge_model, args.reasoning_effort, **options))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                audit = future.result()
                audits[f'{audit["split"]}:{audit["id"]}'] = audit
                accepted += int(audit['split'] == 'train' and audit['accepted'])
                with cache.open('a') as handle:
                    handle.write(json.dumps(audit, ensure_ascii=False) + '\n')
                publish(output, audits, datasets)
                print(f'{audit["split"]} {audit["id"]}: accepted={audit["accepted"]} {audit.get("rejection", audit.get("error", ""))}; accepted train pairs={accepted}', flush=True)
    print(json.dumps(publish(output, audits, datasets), indent=2))


if __name__ == '__main__':
    main()
