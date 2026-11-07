#!/usr/bin/env python3
"""Compare SMT predictions to available references; never call unreviewed rows gold."""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import argparse
from collections import Counter
import json
import hashlib
import shutil
from pathlib import Path
import z3
from autoformalization import validate_smt
from legacy_autoformalization import validate_smt as validate_legacy_smt
from pointwise import parse_function, definedness


CORE_FIELDS = ('id', 'contract_id', 'category', 'text')


def save_references(path, records):
    """Merge ID-keyed references without overwriting an existing reviewed formula."""
    path = Path(path)
    existing = {}
    if path.exists():
        with path.open() as handle:
            existing = {r['id']: r for r in (json.loads(line) for line in handle)}
    for row in records:
        old = existing.get(row['id'])
        if old and old.get('source_text_sha256') != row.get('source_text_sha256'):
            raise ValueError(f'Source changed for reference {row["id"]}')
        if old and old.get('reference_smt2'):
            continue
        existing[row['id']] = row
    if not existing:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(existing[key], ensure_ascii=False) + '\n' for key in sorted(existing)))
    temporary.replace(path)


def separate_references(split_path, reference_dir, clean_split=False):
    """One-time migration of annotated splits; original annotated files are backed up."""
    split_path, reference_dir = Path(split_path), Path(reference_dir)
    rows = [json.loads(line) for line in split_path.open() if line.strip()]
    groups = {}
    for row in rows:
        if not row.get('reference_smt2'):
            continue
        protocol = row.get('formalization_protocol', 'legacy_assertions')
        groups.setdefault(protocol, []).append({
            **{key: value for key, value in row.items() if key not in ('text', 'category')},
            'source_text_sha256': hashlib.sha256(row['text'].encode()).hexdigest()})
    for protocol, refs in groups.items():
        save_references(reference_dir / protocol / split_path.name, refs)
    if clean_split:
        backup = reference_dir / 'imported_annotated_splits' / split_path.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(split_path, backup)
        split_path.write_text(''.join(json.dumps({key: row[key] for key in CORE_FIELDS}, ensure_ascii=False) + '\n' for row in rows))
    return sum(map(len, groups.values()))


def similarity_breakdown(results, scores):
    buckets = {}
    for row in results:
        score = scores.get(row['id'])
        label = ('unavailable' if score is None else '<0.80' if score < .8 else
                 '0.80–0.90' if score < .9 else '0.90–0.95' if score < .95 else '>=0.95')
        buckets.setdefault(label, Counter())[row['status']] += 1
    summary = {}
    for label, counts in buckets.items():
        denominator = counts['equivalent_to_reference'] + counts['differs_from_reference'] + counts['invalid_candidate']
        summary[label] = {'status_counts': dict(counts), 'scored_denominator': denominator,
                          'reference_agreement_rate': counts['equivalent_to_reference'] / denominator if denominator else None}
    return summary


def compare(candidate, reference, symbol_map=None):
    functional = '(define-fun' in candidate
    if functional != ('(define-fun' in reference):
        return {'status': 'protocol_mismatch'}
    validator = validate_smt if functional else validate_legacy_smt
    left, right = validator(candidate), validator(reference)
    if not left['valid']:
        return {'status': 'invalid_candidate', 'validation': left}
    if not right['valid']:
        return {'status': 'invalid_reference', 'validation': right}
    mapping = symbol_map or {name: name for name in left['symbols']}
    if set(mapping) != set(left['symbols']) or set(mapping.values()) != set(right['symbols']) or len(mapping) != len(set(mapping.values())):
        return {'status': 'needs_symbol_alignment'}
    from z3.z3util import get_vars
    if functional:
        a, av, _, _ = parse_function(candidate)
        b, bv, _, _ = parse_function(reference)
    else:
        a, b = z3.And(*z3.parse_smt2_string(candidate)), z3.And(*z3.parse_smt2_string(reference))
        av = {str(v): v for v in get_vars(a)}; bv = {str(v): v for v in get_vars(b)}
    replacements = []
    for source, destination in mapping.items():
        if source not in av or destination not in bv or av[source].sort() != bv[destination].sort():
            return {'status': 'needs_symbol_alignment'}
        replacements.append((av[source], bv[destination]))
    a = z3.substitute(a, *replacements)
    solver = z3.Solver(); solver.set(timeout=3000, rlimit=300000)
    disagreement = (z3.Or(z3.Xor(definedness(a), definedness(b)),
                             z3.And(definedness(a), definedness(b), z3.Xor(a, b)))
                    if functional else z3.Xor(a, b))
    solver.add(disagreement); result = solver.check()
    if result == z3.unsat:
        return {'status': 'equivalent_to_reference'}
    if result == z3.sat:
        return {'status': 'differs_from_reference', 'counterexample': str(solver.model())}
    return {'status': 'solver_unknown', 'reason': solver.reason_unknown()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--predictions', type=Path, help='JSONL: id, smt2, optional symbol_map')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--similarity', type=Path, help='Optional JSONL with id and similarity-to-training score')
    parser.add_argument('--import-split', type=Path, help='Import existing annotated references without changing the split')
    parser.add_argument('--reference-dir', type=Path, default=P.SPLITS / 'references')
    parser.add_argument('--clean-split', action='store_true', help='Back up an annotated split and keep only source fields')
    args = parser.parse_args()
    if args.import_split:
        print(json.dumps({'references_preserved': separate_references(args.import_split, args.reference_dir, args.clean_split)}))
        return
    if not all((args.references, args.predictions, args.output)):
        parser.error('--references, --predictions and --output are required for evaluation')
    references = {r['id']: r for r in (json.loads(line) for line in args.references.open())}
    predictions = [json.loads(line) for line in args.predictions.open()]
    if len({p['id'] for p in predictions}) != len(predictions):
        raise ValueError('Duplicate prediction IDs')
    results = []
    for prediction in predictions:
        row = references.get(prediction['id'])
        if not row or not row.get('reference_smt2') or row.get('reference_status') not in ('model_reviewed_not_gold', 'cross_model_reviewed_not_gold', 'independently_verified'):
            result = {'status': 'no_verified_reference'}
        else:
            result = compare(prediction['smt2'], row['reference_smt2'], prediction.get('symbol_map'))
            result['reference_status'] = row['reference_status']
        results.append({'id': prediction['id'], **result})
    counts = Counter(r['status'] for r in results)
    judged = counts['equivalent_to_reference'] + counts['differs_from_reference'] + counts['invalid_candidate']
    report = {'predictions': len(predictions), 'reference_rows': len(references), 'status_counts': dict(counts),
              'scored_denominator': judged,
              'reference_agreement_rate': counts['equivalent_to_reference'] / judged if judged else None,
              'warning': 'Agreement with model-reviewed references is not gold autoformalization accuracy. Symbol alignment must be semantically justified; mismatches and missing references are reported separately.',
              'results': results}
    if args.similarity:
        scores = {r['id']: r['similarity'] for r in (json.loads(line) for line in args.similarity.open())}
        report['by_training_similarity'] = similarity_breakdown(results, scores)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
