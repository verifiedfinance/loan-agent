#!/usr/bin/env python3
"""Build grouped real-data splits for NL-to-SMT autoformalization."""
import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
from autoformalization import PROTOCOL, CONVENTION, student_prompt


def write_jsonl(path, records):
    temporary = path.with_suffix('.jsonl.tmp')
    with temporary.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    temporary.replace(path)


def grouped_split(records, seed):
    """Keep contracts and exact normalized duplicate rule texts together."""
    parents = {record['contract_id']: record['contract_id'] for record in records}
    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key
    seen = {}
    for record in records:
        fingerprint = hashlib.sha256(' '.join(record['text'].casefold().split()).encode()).hexdigest()
        contract = record['contract_id']
        if fingerprint in seen:
            a, b = sorted((find(contract), find(seen[fingerprint])))
            parents[b] = a
        seen[fingerprint] = contract
    groups = defaultdict(list)
    for record in records:
        groups[find(record['contract_id'])].append(record)
    result = {'train_unlabeled': [], 'validation': [], 'test': []}
    targets = dict(zip(result, (len(records) * .8, len(records) * .1, len(records) * .1)))
    ordered = sorted(groups.items(), key=lambda item: hashlib.sha256(f'{seed}:{item[0]}'.encode()).hexdigest())
    for _, group in ordered:
        destination = max(result, key=lambda name: targets[name] - len(result[name]))
        result[destination].extend(group)
    for rows in result.values():
        rows.sort(key=lambda record: record['id'])
    return result, len(groups), max(map(len, groups.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=P.PROCESSED / 'financial_ratio_records.jsonl')
    parser.add_argument('--output-dir', type=Path, default=P.SPLITS)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Input must contain records with unique IDs')
    splits, group_count, largest = grouped_split(records, args.seed)
    output = args.output_dir
    planned = {name: ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
               for name, rows in splits.items()}
    # Split files are immutable. Check all destinations before writing anything.
    for name, contents in planned.items():
        path = output / f'{name}.jsonl'
        if path.exists() and path.read_text() != contents:
            raise ValueError(f'{path} differs from this split. Use a new output directory; existing splits are not overwritten.')
    output.mkdir(parents=True, exist_ok=True)
    for name, contents in planned.items():
        path = output / f'{name}.jsonl'
        if not path.exists():
            path.write_text(contents)
    write_jsonl(output / 'grpo_train.jsonl', [
        dict(row, prompt=student_prompt(row), task='financial_autoformalization',
             declared_assumptions=[], formalization_protocol=PROTOCOL,
             instantiation_convention=CONVENTION)
        for row in splits['train_unlabeled']])
    manifest = {
        'seed': args.seed, 'source_file': str(args.input), 'source_sha256': source_sha256,
        'source_records': len(records), 'target_fractions': [0.8, 0.1, 0.1],
        'split_records': {name: len(rows) for name, rows in splits.items()},
        'split_contracts': {name: len({r['contract_id'] for r in rows}) for name, rows in splits.items()},
        'contract_duplicate_components': group_count, 'largest_component_records': largest,
        'formalization_protocol': PROTOCOL, 'instantiation_convention': CONVENTION,
        'split_policy': 'Immutable; contracts and exact normalized duplicates stay together.',
        'near_duplicates': 'Reported separately; numeric variants are retained.',
    }
    (output / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
