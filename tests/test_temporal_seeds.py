import json
import unittest
from pathlib import Path

import numpy as np

from sbercluster import temporal_seeds as ts
from sbercluster.io import sha256

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports' / 'temporal-seeds'


class TemporalSeedsTests(unittest.TestCase):
    def test_pairwise_ari_is_label_invariant(self):
        a = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])]
        b = [np.array([5, 5, 2, 2]), np.array([3, 4, 4, 4])]
        out = ts.pairwise_ari({1: a, 2: b, 3: a})
        self.assertEqual(out['pairs'], 3)
        self.assertAlmostEqual(out['monthly']['min'], 1.0)

    def test_cell_labels_best_match_and_modal(self):
        cells = {7: {(0, 0), (0, 1), (1, 0)}, 9: {(1, 1)}}
        labels = ts.cell_labels(cells, 2, 2, {7: 0})
        self.assertEqual(labels.tolist(), [[0, 0], [0, -1]])
        self.assertEqual(ts.modal(labels).tolist(), [0, 0])
        self.assertEqual(ts.best_match({(0, 0), (1, 0)}, cells), (7, 2 / 3))

    def test_published_summary(self):
        summary = json.loads((REPORT / 'summary.json').read_text('utf-8'))
        self.assertEqual(len(summary['seeds']), 20)
        self.assertTrue({1729, 2718, 3141} <= set(summary['seeds']))
        self.assertEqual(summary['omega_relative'], 0.01)
        self.assertEqual(len(summary['groups']), 7)
        for seed, digest in summary['labels_sha256'].items():
            self.assertEqual(sha256(REPORT / 'labels' / f'seed{seed}.json.gz'), digest)
        profiles = json.loads((ROOT / 'reports' / 'temporal-groups' / 'profiles.json').read_text('utf-8'))
        self.assertEqual(sorted(g['size_cells'] for g in summary['groups']), sorted(g['cells'] for g in profiles['groups']))
        self.assertEqual(summary['groups_found_in_all_seeds'], sum(g['seeds_found'] == 20 for g in summary['groups']))
        a = summary['pairwise_ari']['monthly']
        self.assertEqual(summary['pairwise_ari']['pairs'], 190)
        self.assertTrue(0 < a['min'] <= a['median'] <= 1)


if __name__ == '__main__':
    unittest.main()
