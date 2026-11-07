#!/usr/bin/env python3
"""Download Zenodo 20014336 and export source-grounded covenant/default records.

Standard library only (curl is needed for --download). Run --help for options.
"""
from __future__ import annotations

import sys as _sys  # repository bootstrap: src/ and its sub-folders on the import path
from pathlib import Path as _Path
_SRC = _Path(__file__).resolve().parents[1]
_sys.path[:0] = [p for p in [str(_SRC)] + [str(d) for d in sorted(_SRC.iterdir()) if d.is_dir() and not d.name.startswith(('_', '.'))] if p not in _sys.path]
import paths as P  # noqa: E402
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

BASE = Path(__file__).resolve().parent
SOURCE = 'https://zenodo.org/records/20014336'
CHECKSUMS = {
    'publicly_filed_contracts.zip': '767909a5d451bebbb5d836bdcc72c199',
    'main_control_file.json': 'b5307dd43fae807eaf3d5ee7007e0552',
    'breakpoint_file.json': '18262a13d567b89b5295f7a61d6ba0ff',
}
TAG = re.compile(r'^\s*<[^>]+>\s*$')
CLAUSE = re.compile(r'^\s*(?:(?:SECTION|Section)\s+)?(?:\d+(?:\.\d+)*[.)]?|\([A-Za-z0-9]+\)|[A-Za-z][.)]|[ivxlcdm]+[.)])\s+')
HEADING = re.compile(r'\b(?:covenants?|events? of default)\b', re.I)
RATIO = re.compile(r'\b(?:ratio|leverage|interest coverage|fixed charge coverage|debt service coverage)\b', re.I)
CATEGORIES = [
    ('financial_threshold', r'\b(?:net worth|minimum liquidity|minimum (?:consolidated )?EBITDA|capital expenditures)\b'),
    ('reporting', r'\b(?:financial statements|compliance certificate|reporting requirements|books and records|inspection)\b'),
    ('restricted_payments', r'\b(?:restricted payments?|dividends?|distributions?)\b'),
    ('liens', r'\b(?:liens?|negative pledge)\b'),
    ('indebtedness', r'\b(?:indebtedness|borrowings?|guarant(?:ee|y|ies))\b'),
    ('asset_disposition', r'\b(?:disposition|disposal|sale of assets|sales of assets|merger|consolidation)\b'),
    ('investments', r'\b(?:investments?|loans and advances)\b'),
    ('insurance', r'\binsurance\b'),
    ('taxes', r'\b(?:taxes|taxation)\b'),
    ('compliance', r'\b(?:compliance with laws|environmental|ERISA)\b'),
]
CATEGORIES = [(name, re.compile(pattern, re.I)) for name, pattern in CATEGORIES]


def md5(path):
    h = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_inputs(data_dir, download=False):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in CHECKSUMS.items():
        path = data_dir / name
        if path.exists() and md5(path) == expected:
            continue
        if not download:
            raise ValueError(f'{path}: missing or checksum mismatch; run with --download')
        # Preserve any invalid existing file; only resume our own .part download.
        partial = path.with_name(name + '.part')
        subprocess.run(['curl', '-fL', '--retry', '5', '-C', '-', '-o', str(partial),
                        f'{SOURCE}/files/{name}?download=1'], check=True)
        if md5(partial) != expected:
            raise ValueError(f'Checksum mismatch: {partial}')
        partial.replace(path)


def reflow(lines, first_line):
    """Join wrapped nonblank lines; preserve blank/tag rows for parser offsets.

    Control-file coordinates are 1-based, inclusive. Breakpoint coordinates are
    0-based within reflowed sections. Keep physical source ranges for auditing.
    """
    # The supplied parser trims leading blank lines before reflowing.
    while lines and not lines[0].strip():
        lines = lines[1:]
        first_line += 1
    rows, pending = [], []
    start = first_line
    for number, raw in enumerate(lines, first_line):
        value = raw.strip()
        if not value or TAG.fullmatch(value):
            if pending:
                rows.append((' '.join(pending), start, number - 1))
                pending = []
            rows.append((value, number, number))
        else:
            if not pending:
                start = number
            pending.append(value)
    if pending:
        rows.append((' '.join(pending), start, first_line + len(lines) - 1))
    return rows


def clean(rows):
    values = []
    for text, _, _ in rows:
        if TAG.fullmatch(text) or re.fullmatch(r'[-\s]*\d+[-\s]*', text):
            continue
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'(?<!\S)[_-]{3,}(?!\S)', ' ', text)
        if text.strip():
            values.append(text)
    return re.sub(r'\s+', ' ', ' '.join(values)).strip()


def fallback_segment(rows, reason):
    """Recover explicit numbered headings when supplied offsets do not align.

    Require a consistent article prefix and strictly increasing rule numbers.
    Otherwise leave the complete section intact for downstream review.
    """
    pattern = re.compile(r'^\s*(?:(?:SECTION|Section)\s+)?(\d+)\.(\d+)[.)]?\s+')
    matches = [(i, pattern.match(row[0])) for i, row in enumerate(rows)]
    matches = [(i, match) for i, match in matches if match]
    if len(matches) >= 2:
        prefixes = {match.group(1) for _, match in matches}
        numbers = [int(match.group(2)) for _, match in matches]
        if len(prefixes) == 1 and numbers == sorted(set(numbers)):
            starts = [i for i, _ in matches]
            context = clean(rows[:starts[0]])
            return [(start, end, context, 'numbered_heading_fallback')
                    for start, end in zip(starts, starts[1:] + [len(rows)])]
    return [(0, len(rows), '', reason)]


def segment(rows, metadata):
    """Keep top-level rules with nested conditions/exceptions intact.

    If offsets cannot be verified, retain the complete section rather than
    silently cutting text at unreliable positions.
    """
    points = metadata.get('breakpoints', []) if isinstance(metadata, dict) else []
    if not points:
        return fallback_segment(rows, 'section_fallback_no_breakpoints')
    positions = [p.get('line') for p in points]
    if (any(not isinstance(p, int) or p < 0 or p >= len(rows) for p in positions)
            or positions != sorted(set(positions))):
        return fallback_segment(rows, 'section_fallback_invalid_offsets')
    # A root heading at row zero is context, not a separate obligation.
    root = points[0]
    if (HEADING.search(rows[root['line']][0][:200])
            and all(p['depth'] > root['depth'] for p in points[1:])):
        points = points[1:]
    if not points:
        return fallback_segment(rows, 'section_fallback_heading_only')
    # Validate all supplied boundaries, including nested clauses. The publisher's
    # reflow procedure is not supplied; unverified sections stay whole.
    if any(not CLAUSE.match(rows[p['line']][0]) for p in points):
        return fallback_segment(rows, 'section_fallback_unaligned_breakpoints')
    depth = min(p['depth'] for p in points)
    starts = [p['line'] for p in points if p['depth'] == depth]
    context = clean(rows[:starts[0]])
    return [(start, end, context, 'top_level_breakpoints')
            for start, end in zip(starts, starts[1:] + [len(rows)])]


def categorize(body, section_type, section_heading):
    if section_type == 'default_section':
        return 'event_of_default'
    # Heading-first classification prevents a reference buried in exceptions from
    # routinely overriding the rule's subject. These are heuristic labels.
    heading = body[:250]
    for candidate in (heading, body):
        if RATIO.search(candidate):
            return 'financial_ratio'
        for category, pattern in CATEGORIES:
            if pattern.search(candidate):
                return category
    if re.search(r'negative covenants', section_heading, re.I):
        return 'negative_covenant'
    if re.search(r'affirmative covenants', section_heading, re.I):
        return 'affirmative_covenant'
    return 'covenant'


def extract(data_dir, output_dir, primary_only=False, limit=None):
    control = json.loads((data_dir / 'main_control_file.json').read_text())
    breakpoints = json.loads((data_dir / 'breakpoint_file.json').read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    counts, methods, issues = Counter(), Counter(), []
    contracts, record_id = set(), 0
    names = ['rl_rules.jsonl', 'financial_ratio_records.jsonl', 'provenance.jsonl']
    handles = [open(output_dir / (name + '.tmp'), 'w', encoding='utf-8') for name in names]
    rules, financial, provenance = handles
    try:
        with zipfile.ZipFile(data_dir / 'publicly_filed_contracts.zip') as archive:
            members = set(archive.namelist())
            selected = sorted(control)
            if primary_only:
                selected = [key for key in selected if control[key].get('primary_contract') == 1]
            if limit is not None:
                selected = selected[:limit]
            for index, key in enumerate(selected):
                entry = control[key]
                if key not in members:
                    issues.append({'contract': key, 'issue': 'missing_archive_member'})
                    continue
                raw = archive.read(key)
                try:
                    text = raw.decode('utf-8')
                    encoding = 'utf-8'
                except UnicodeDecodeError:
                    text = raw.decode('cp1252', errors='replace')
                    encoding = 'cp1252_with_replacement'
                lines = text.splitlines()
                for section_type in ('covenant_section', 'default_section'):
                    for section_id, bounds in sorted((entry.get(section_type) or {}).items()):
                        top, bottom = bounds['top_line'], bounds['bottom_line']
                        if not (1 <= top <= bottom <= len(lines)):
                            issues.append({'contract': key, 'section': section_type,
                                           'section_id': section_id, 'issue': 'invalid_source_range',
                                           'top': top, 'bottom': bottom, 'line_count': len(lines)})
                            continue
                        rows = reflow(lines[top - 1:bottom], top)
                        metadata = ((breakpoints.get(key) or {}).get(section_type) or {}).get(section_id, {})
                        segments = segment(rows, metadata)
                        context_end = rows[segments[0][0] - 1][2] if segments[0][2] else None
                        for start, end, context, method in segments:
                            body = clean(rows[start:end])
                            if not body:
                                continue
                            record_id += 1
                            category = (section_type if method.startswith('section_fallback_')
                                        else categorize(body, section_type, rows[0][0]))
                            record = {'id': f'fin_{record_id:05d}',
                                      'contract_id': entry.get('filehandle', Path(key).stem),
                                      'category': category,
                                      'text': f'{context}\n\n{body}' if context else body}
                            serialized = json.dumps(record, ensure_ascii=False) + '\n'
                            rules.write(serialized)
                            if category == 'financial_ratio':
                                financial.write(serialized)
                            source = {'id': record['id'], 'archive_path': key,
                                      'section_type': section_type, 'section_id': section_id,
                                      'source_start_line': rows[start][1],
                                      'source_end_line': rows[end - 1][2],
                                      'context_start_line': top if context else None,
                                      'context_end_line': context_end if context else None,
                                      'extraction_method': method, 'category_method': 'keyword_heuristic',
                                      'encoding': encoding,
                                      'primary_contract': entry.get('primary_contract')}
                            provenance.write(json.dumps(source) + '\n')
                            counts[category] += 1
                            methods[method] += 1
                            contracts.add(key)
                if (index + 1) % 1000 == 0:
                    print(f'Processed {index + 1}/{len(selected)} contracts; {record_id} records', flush=True)
    finally:
        for handle in handles:
            handle.close()
    for name in names:
        (output_dir / (name + '.tmp')).replace(output_dir / name)
    summary = {'source': SOURCE, 'input_md5': CHECKSUMS, 'control_contracts': len(control),
               'selected_contracts': len(selected), 'contracts_with_records': len(contracts),
               'records': record_id, 'categories': dict(counts), 'extraction_methods': dict(methods),
               'issues': issues, 'primary_only': primary_only,
               'limitations': ['Categories are keyword heuristics, not human-verified labels.',
                              'Unaligned breakpoints use heuristic numbered headings where possible; otherwise sections remain whole with section categories.',
                              'Shared section context and nested exceptions are retained; external definitions remain in contracts.',
                              'These are rule-text records, not RL prompts, rewards, or gold outcomes.',
                              'Split downstream train/evaluation data by contract_id to reduce leakage.']}
    (output_dir / 'extraction_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'issues'}, indent=2))
    print(f'Source-range/missing-file issues: {len(issues)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=P.RAW)
    parser.add_argument('--output-dir', type=Path, default=P.PROCESSED)
    parser.add_argument('--download', action='store_true', help='Download missing files and verify publisher MD5s')
    parser.add_argument('--primary-only', action='store_true', help='Only primary_contract=1 (not the full paper sample filter)')
    parser.add_argument('--limit', type=int, help='Process first N contracts for a smoke test')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    verify_inputs(args.data_dir, args.download)
    extract(args.data_dir, args.output_dir, args.primary_only, args.limit)


if __name__ == '__main__':
    main()
