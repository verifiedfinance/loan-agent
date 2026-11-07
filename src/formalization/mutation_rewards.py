"""The existing GRPO reward with the requested, bounded mutation-witness penalty."""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
from concurrent.futures import ThreadPoolExecutor
import json

import z3
from autoformalization import source_payload, validate_smt
from audit_mutation_witnesses import prepare
from audit_witness_feasibility import ORACLE, ask, check_state

WITNESS_MODELS = ('gpt-6-astra', 'gpt-6-astra', 'gpt-6-sol')
REWARD_CONFIG = {
    'formula': '0 if invalid; base if no usable witnesses; otherwise max(0.05, base - 0.1 * E)',
    'E': 'Error-exposing witnesses divided by reliably labeled witnesses; exact duplicate states count once.',
    'reliability': 'Grounded inputs; all three calls agree on feasible=true and the same Boolean holds label.',
    'witness_models': list(WITNESS_MODELS),
    'mutations': 'Unchanged audit_mutation_witnesses.mutate: two existing controlled edits.',
    'unavailable_witness_feedback': 'Exclude unavailable states; fall back to the base reward if none remain.',
}


def reliable_error(witness):
    verdicts = witness['verdicts']
    if (len(verdicts) != 3 or any(v.get('feasible') is not True or type(v.get('holds')) is not bool for v in verdicts)
            or len({v['holds'] for v in verdicts}) != 1):
        return None
    return verdicts[0]['holds'] != witness['original_holds']


def penalty(base, valid, evidence):
    if not valid:
        return {'reward': 0.0, 'E': None, 'usable_witnesses': 0, 'error_witnesses': 0}
    errors = [error for w in evidence.get('witnesses', []) if (error := reliable_error(w)) is not None]
    fraction = sum(errors) / len(errors) if errors else None
    return {'reward': min(base, max(0.05, base - 0.1 * fraction)) if errors else base,
            'E': fraction, 'usable_witnesses': len(errors), 'error_witnesses': sum(errors)}


def mutation_evidence(record, smt):
    candidate = {'id': 'online', 'sample': 0, 'smt2': smt, 'validation': validate_smt(smt)}
    try:
        prepared = prepare(record, candidate)
    except (ValueError, z3.Z3Exception, RecursionError) as exc:
        return {'status': 'unsupported_mutation', 'reason': str(exc), 'witnesses': [], 'api_errors': 0}
    evidence = {'status': prepared['schema']['status'], 'schema': prepared['schema'],
                'mutations': prepared['mutations'], 'witnesses': [],
                'api_errors': int(prepared['schema']['status'] == 'error')}
    if prepared['schema']['status'] != 'grounded':
        return evidence
    seen = set()
    for mutation in prepared['mutations']:
        if mutation['status'] != 'distinguishing_state':
            continue
        key = json.dumps(mutation['values'], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        # Neither the candidate, the mutant, nor their outcomes are shown to the state judge.
        payload = {**source_payload(record), 'symbols': prepared['schema']['symbols'], 'observation': mutation['values']}
        def judge(model):
            try:
                answer = ask(model, ORACLE, payload, 'blind_mutation_witness_classification',
                             lambda verdict: check_state(verdict, record))
                return dict(answer, model=model)
            except Exception as exc:
                return {'model': model, 'feasible': None, 'holds': None, 'error': str(exc)}
        # Only independent API calls run concurrently; all Z3 work stays on the main thread.
        with ThreadPoolExecutor(max_workers=3) as pool:
            verdicts = list(pool.map(judge, WITNESS_MODELS))
        evidence['api_errors'] += sum('error' in v for v in verdicts)
        evidence['witnesses'].append({'values': mutation['values'], 'original_holds': mutation['candidate_a_holds'],
                                      'verdicts': verdicts})
    return evidence


def make_mutation_grpo_reward(base_reward, audit_callback=None, *, witness_provider=mutation_evidence):
    """Keep base-judge failure behavior; optional witness failures never become penalties."""
    cache = {}
    def mutation_reward(completions, text, declared_assumptions=None, source_context=None, **kwargs):
        assumptions = declared_assumptions if declared_assumptions is not None else [[] for _ in text]
        contexts = source_context if source_context is not None else [[] for _ in text]
        ids = kwargs.get('id', [None] * len(text))
        if any(len(column) != len(completions) for column in (text, assumptions, contexts, ids)):
            raise ValueError('Source columns must align with completions')
        # Unavailable base semantic feedback still aborts the batch, as in standard GRPO.
        bases = base_reward(completions=completions, text=text, declared_assumptions=assumptions,
                            source_context=contexts, **kwargs)
        scores = []
        for record_id, completion, clause, declared, context, base in zip(ids, completions, text, assumptions, contexts, bases):
            smt = completion
            if isinstance(smt, list):
                smt = smt[0].get('content', '') if len(smt) == 1 and isinstance(smt[0], dict) else ''
            validity = validate_smt(smt)
            evidence = {'status': 'invalid_smt', 'witnesses': [], 'api_errors': 0}
            if validity['valid']:
                record = {'text': clause, 'declared_assumptions': declared, 'source_context': context}
                key = json.dumps({**source_payload(record), 'candidate_smt2': smt}, sort_keys=True)
                if key not in cache:
                    evidence = witness_provider(record, smt)
                    # Do not freeze transient failures in the in-memory cache.
                    if not evidence['api_errors']:
                        if len(cache) >= 1024:
                            cache.pop(next(iter(cache)))
                        cache[key] = json.dumps(evidence)
                else:
                    evidence = json.loads(cache[key])
            result = penalty(base, validity['valid'], evidence)
            scores.append(result['reward'])
            if audit_callback:
                audit_callback({'step': getattr(kwargs.get('trainer_state'), 'global_step', None), 'id': record_id,
                                'base_reward': base, **result, 'evidence': evidence})
        return scores
    return mutation_reward
