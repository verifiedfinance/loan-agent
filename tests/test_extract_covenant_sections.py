import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from extract_covenant_sections import clean, reflow, segment, categorize, extract


class ExtractionTests(unittest.TestCase):
    def test_reflow_offsets_and_page_breaks(self):
        rows = reflow(['', 'SECTION 8. Negative covenants.', 'The borrower will not:', '',
                       '8.1. Permit the current ratio', 'to fall below 1.5.', '', '12', '<PAGE>', '',
                       '8.2. Incur debt, except:', '', '(a) existing debt;', '', '(b) permitted debt.'], 100)
        self.assertEqual(rows[0][1:3], (101, 102))
        self.assertEqual(rows[2][0], '8.1. Permit the current ratio to fall below 1.5.')
        metadata = {'breakpoints': [{'line': 2, 'depth': 2}, {'line': 7, 'depth': 2},
                                    {'line': 9, 'depth': 3}, {'line': 11, 'depth': 3}]}
        result = segment(rows, metadata)
        self.assertEqual(len(result), 2)
        self.assertIn('will not:', result[0][2])
        self.assertIn('(b) permitted debt.', clean(rows[result[1][0]:result[1][1]]))
        self.assertNotIn('<PAGE>', clean(rows))
        self.assertEqual(categorize(clean(rows[2:3]), 'covenant_section', ''), 'financial_ratio')

    def test_misaligned_offsets_preserve_entire_section(self):
        rows = reflow(['Heading', '', '1. Full rule'], 1)
        self.assertEqual(segment(rows, {'breakpoints': [{'line': 1, 'depth': 1}]})[0][:2], (0, 3))
        self.assertEqual(segment(rows, {'breakpoints': [{'line': 20, 'depth': 1}]})[0][:2], (0, 3))

    def test_numbered_heading_recovery(self):
        rows = reflow(['The borrower will not:', '', '8.1. Incur debt except:', '',
                       '(a) permitted debt.', '', '8.2. Grant liens.'], 1)
        result = segment(rows, {'breakpoints': [{'line': 500, 'depth': 1}]})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][3], 'numbered_heading_fallback')
        self.assertEqual(result[0][2], 'The borrower will not:')
        self.assertIn('permitted debt', clean(rows[result[0][0]:result[0][1]]))

    def test_export_shared_context_range_stays_constant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            key = 'agreement.txt'
            text = 'The borrower will not:\n\n8.1. Incur debt.\n\n8.2. Grant liens.\n'
            with zipfile.ZipFile(root / 'publicly_filed_contracts.zip', 'w') as archive:
                archive.writestr(key, text)
            (root / 'main_control_file.json').write_text(json.dumps({key: {
                'filehandle': 'agreement', 'covenant_section': {
                    '1': {'top_line': 1, 'bottom_line': 5}}}}))
            (root / 'breakpoint_file.json').write_text(json.dumps({key: None}))
            with contextlib.redirect_stdout(io.StringIO()):
                extract(root, root / 'output')
            provenance = [json.loads(line) for line in
                          (root / 'output/provenance.jsonl').read_text().splitlines()]
            self.assertEqual(len(provenance), 2)
            self.assertEqual([p['context_end_line'] for p in provenance], [2, 2])
            records = [json.loads(line) for line in
                       (root / 'output/rl_rules.jsonl').read_text().splitlines()]
            self.assertEqual(records[1]['text'], 'The borrower will not:\n\n8.2. Grant liens.')

    def test_default_parent_is_context(self):
        rows = reflow(['9.1 Events of Default. The following constitute default:', '',
                       '(a) Failure to pay;', '', '(b) Insolvency.'], 1)
        result = segment(rows, {'breakpoints': [{'line': 0, 'depth': 1},
                          {'line': 2, 'depth': 2}, {'line': 4, 'depth': 2}]})
        self.assertEqual(len(result), 2)
        self.assertIn('constitute default', result[1][2])
        self.assertEqual(categorize('Breach of leverage ratio', 'default_section', ''), 'event_of_default')


if __name__ == '__main__':
    unittest.main()
