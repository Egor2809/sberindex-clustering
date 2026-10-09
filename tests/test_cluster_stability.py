"""Semantic checks for diagnostics over saved partitions."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.report_cluster_stability import (
    cluster_matches, entity_neighborhoods, load_saved, pair_frequency_histograms,
)
from sbercluster.selection_frontier import REVISION


class ClusterStabilityTests(unittest.TestCase):
    def test_saved_ari_rejects_nonfinite_values_even_with_matching_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pins = {}

            def save(relative, value):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(value, np.ndarray):
                    np.save(path, value, allow_pickle=False)
                else:
                    payload = value if isinstance(value, str) else json.dumps(value)
                    path.write_text(payload, encoding='utf-8')
                pins[relative] = hashlib.sha256(path.read_bytes()).hexdigest()

            labels = np.repeat(np.arange(4), 2)
            features = np.zeros((8, 5))
            features[:, 0] = labels
            ids = [f'test_{i}' for i in range(8)]
            scaler = {'calibration_end': '2023-12-01'}
            model = {'feature_scaler': scaler, 'cluster_mapping': [0, 1, 2, 3],
                     'rule': {'revision': REVISION, 'transform': np.eye(5).tolist(),
                              'centers': features[::2].tolist(), 'biases': [0., 0., 0., 0.]}}
            records = [{'draw': i, 'seed': 2200 + i, 'month_indices': list(range(12)),
                        'cluster_sizes': [2, 2, 2, 2], 'ARI': 1.} for i in range(30)]
            save('constrained/labels.json', {'ids': ids, 'labels': {'historical_kmeans4': labels.tolist()}})
            save('constrained/provenance.json', {'ids': ids, 'scaler': scaler})
            save('constrained/shortlist.json', {'ids': ['historical_kmeans4']})
            save('constrained/source/scripts/benchmark_frontier.py', '# synthetic test fixture\n')
            save('constrained/features.npy', features)
            save('models/historical_kmeans4.json', model)
            for i in range(30):
                save(f'validation/historical_kmeans4__month_draw{i}.npy', labels)

            def update_ari(value):
                records[0]['ARI'] = value
                save('validation/stability.json', {'historical_kmeans4': records})
                (root / 'manifest.json').write_text(json.dumps({'files_sha256': pins}), encoding='utf-8')

            update_ari(1.)
            self.assertEqual(load_saved(root)[0], ids)
            for value in (float('nan'), float('inf'), float('-inf')):
                with self.subTest(saved_ari=value):
                    update_ari(value)
                    with self.assertRaisesRegex(ValueError, 'ARI must be finite'):
                        load_saved(root)
            update_ari(0.)
            with self.assertRaisesRegex(ValueError, 'ARI differs from metadata'):
                load_saved(root)

    def test_arbitrary_label_permutation_preserves_every_group(self):
        reference = [0, 0, 1, 1, 2, 2]
        reordered = [17, 17, 8, 8, 42, 42]
        matches = cluster_matches(reference, reordered)
        self.assertEqual([m['best_jaccard'] for m in matches], [1., 1., 1.])
        jaccard, retention = entity_neighborhoods(reference, reordered)
        np.testing.assert_array_equal(jaccard, np.ones(6))
        np.testing.assert_array_equal(retention, np.ones(6))

    def test_split_does_not_merge_fragments_to_claim_perfect_stability(self):
        matches = cluster_matches([0, 0, 0, 0, 1, 1], [7, 7, 9, 9, 4, 4])
        self.assertEqual(matches[0]['best_jaccard'], .5)
        self.assertEqual(matches[0]['matched_labels'], [7, 9])
        self.assertEqual(matches[1]['best_jaccard'], 1.)

    def test_merge_can_match_two_reference_groups_to_one_new_group(self):
        matches = cluster_matches([0, 0, 1, 1, 2, 2], [9, 9, 9, 9, 8, 8])
        self.assertEqual(matches[0]['matched_labels'], [9])
        self.assertEqual(matches[1]['matched_labels'], [9])
        self.assertEqual([m['best_jaccard'] for m in matches], [.5, .5, 1.])

    def test_entity_moving_group_loses_original_peers(self):
        jaccard, retention = entity_neighborhoods([0, 0, 0, 1, 1], [0, 0, 1, 1, 1])
        self.assertEqual(jaccard[2], 0.)
        self.assertEqual(retention[2], 0.)
        self.assertEqual(jaccard[0], .5)
        self.assertEqual(retention[3], 1.)
        self.assertLess(jaccard[3], 1.)

    def test_pair_histogram_counts_distinct_unordered_pairs_without_self(self):
        rows = pair_frequency_histograms([0, 0, 1], [[8, 8, 2], [8, 2, 2]])
        self.assertEqual(sum(r['pairs'] for r in rows), 3)
        counts = {(r['reference_left'], r['reference_right'], r['joint_repeats']): r['pairs'] for r in rows}
        self.assertEqual(counts[(0, 0, 1)], 1)
        self.assertEqual(counts[(0, 1, 0)], 1)
        self.assertEqual(counts[(0, 1, 1)], 1)
        self.assertEqual(sum(r['pairs'] for r in rows if r['reference_left'] == r['reference_right'] == 1), 0)

    def test_joint_frequency_is_invariant_to_each_repeat_label_numbers(self):
        baseline = [0, 0, 1, 1]
        first = pair_frequency_histograms(baseline, [[0, 0, 1, 1], [0, 1, 1, 1]])
        second = pair_frequency_histograms(baseline, [[7, 7, 4, 4], [8, 6, 6, 6]])
        self.assertEqual(first, second)

    def test_missing_or_differently_sized_entity_universe_is_rejected(self):
        with self.assertRaises(ValueError):
            cluster_matches([0, 1], [1])
        with self.assertRaises(ValueError):
            entity_neighborhoods([0, 1], [-1, 0])
        with self.assertRaises(ValueError):
            cluster_matches([0, 1], [0., np.nan])

    def test_singleton_is_not_counted_as_its_own_stable_pair(self):
        jaccard, retention = entity_neighborhoods([0, 1], [8, 7])
        np.testing.assert_array_equal(jaccard, [1., 1.])
        self.assertTrue(np.isnan(retention).all())


if __name__ == '__main__':
    unittest.main()
