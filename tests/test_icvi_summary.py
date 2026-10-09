import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.export_icvi_summary import ROOT, export_summary, validate_records


class IcviSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads((ROOT / 'reports/contest-v3/published_partition_metrics.json').read_text(encoding='utf-8'))

    def test_published_records_and_undefined_density_are_valid(self):
        validate_records(self.records)
        self.assertTrue(any(row['S_Dbw'] is None for row in self.records))

    def test_invalid_sizes_metrics_and_duplicate_candidates_are_rejected(self):
        for field, value in [('n', -8), ('k', 0), ('min_cluster_size', 100000),
                             ('n', True), ('SW', 8), ('CH', -100), ('S_Dbw', -1),
                             ('AVI', 2), ('AVU', -1), ('NewmanQ', 9),
                             ('SW', float('nan')), ('CH', float('inf'))]:
            with self.subTest(field=field, value=value):
                rows = copy.deepcopy(self.records)
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    validate_records(rows)
        with self.assertRaises(ValueError):
            validate_records([self.records[0], self.records[0]])

    def test_invalid_comparison_does_not_write_partial_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            directory = Path(tmp)
            rows = copy.deepcopy(self.records)
            next(row for row in rows if row['k'] == 4)['SW'] = None
            source = directory / 'input.json'
            source.write_text(json.dumps(rows), encoding='utf-8')
            with self.assertRaises(ValueError):
                export_summary(source, directory / 'report', directory / 'fragment.html')
            self.assertFalse((directory / 'report').exists())
            self.assertFalse((directory / 'fragment.html').exists())


if __name__ == '__main__':
    unittest.main()
