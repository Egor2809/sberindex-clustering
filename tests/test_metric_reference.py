"""Guards for the independent metric-audit entrypoint, without network access."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.verify_metric_reference import compare_partition, load_reference


class MetricReferenceAudit(unittest.TestCase):
    def test_changed_reference_is_rejected_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed_reference.py"
            path.write_text("raise AssertionError('This source must never execute')", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_reference(path)

    def test_analytical_path_and_deliberately_wrong_reference(self):
        graph = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], float)
        def correct(a, labels):
            return {"AVI": 2 / 3, "AVU": 1, "modularity": 1 / 6}
        good = compare_partition(graph, [4, 4, 8, 8], correct)
        self.assertEqual(good["failures"], [])
        self.assertAlmostEqual(good["comparisons"]["TurboMQ_identity"]["actual"], 4 / 3)
        def wrong(a, labels):
            return {"AVI": 2 / 3, "AVU": 0.5, "modularity": 1 / 6}
        bad = compare_partition(graph, [4, 4, 8, 8], wrong)
        self.assertEqual(bad["failures"], ["AVU"])

    def test_zero_denominator_stays_explicitly_undefined(self):
        graph = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], float)
        def zero_convention(a, labels):
            return {"AVI": 1, "AVU": 0, "modularity": 0.5}
        row = compare_partition(graph, [0, 0, 1, 1], zero_convention)
        self.assertEqual(row["failures"], [])
        self.assertEqual(row["comparisons"]["AVU"]["status"], "undefined_by_strict_contract")
        self.assertEqual(row["AVU_undefined_ordered_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
