import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from evaluate_smt_predictions import save_references, similarity_breakdown

BASE = Path(__file__).resolve().parent.parent


class DatasetResponsibilitiesTests(unittest.TestCase):
    def test_splitter_keeps_splits_immutable_and_does_not_create_sft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'records.jsonl'; output = root / 'datasets'
            records = [{'id': str(i), 'contract_id': str(i // 2), 'category': 'financial_ratio', 'text': f'Rule {i}'} for i in range(100)]
            source.write_text(''.join(json.dumps(row) + '\n' for row in records))
            command = [sys.executable, str(BASE / 'src/data_prep/build_rl_datasets.py'), '--input', str(source), '--output-dir', str(output)]
            subprocess.run(command, check=True, capture_output=True)
            self.assertFalse((output / 'sft_synthetic.jsonl').exists())
            paths = [output / f'{name}.jsonl' for name in ('train_unlabeled', 'validation', 'test')]
            original = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(original, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])
            different = subprocess.run(command + ['--seed', '43'], capture_output=True)
            self.assertNotEqual(different.returncode, 0)
            self.assertEqual(original, [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths])

    def test_reviewed_reference_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.jsonl'
            gold = {'id': 'a', 'source_text_sha256': 'source', 'reference_smt2': 'reviewed formula', 'reference_status': 'independently_verified'}
            save_references(path, [gold])
            save_references(path, [dict(gold, reference_smt2=None, reference_status='pending_verification')])
            save_references(path, [dict(gold, reference_smt2='different generated formula')])
            self.assertEqual(json.loads(path.read_text()), gold)

    def test_similarity_reports_denominators(self):
        rows = [{'id': 'a', 'status': 'equivalent_to_reference'}, {'id': 'b', 'status': 'no_verified_reference'}]
        report = similarity_breakdown(rows, {'a': .99, 'b': .99})
        self.assertEqual(report['>=0.95']['scored_denominator'], 1)
        self.assertEqual(report['>=0.95']['status_counts']['no_verified_reference'], 1)


if __name__ == '__main__':
    unittest.main()
