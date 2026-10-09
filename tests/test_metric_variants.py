import unittest

import numpy as np
from scipy.sparse import csr_matrix

from sbercluster.metrics import (
    all_metrics,
    calinski_harabasz_indices,
    network_indices,
    network_quality_candidates,
    s_dbw,
    s_dbw_paper,
    partition_labels,
    aligned_stability,
)


class MetricVariantTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0.0], [1.0], [9.0], [10.0]])
        self.labels = np.array([0, 0, 1, 1])
        self.path = csr_matrix(
            [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
            dtype=float,
        )

    def test_ch_and_ch_per_n_match_hand_calculation(self):
        result = calinski_harabasz_indices(self.x, self.labels)
        self.assertAlmostEqual(result["CH"], 162.0)
        self.assertAlmostEqual(result["CH_per_n"], 40.5)
        self.assertEqual(result["CH_status"], "ok")
        self.assertFalse(result["CH_is_infinite"])

    def test_feature_and_label_contract_rejects_malformed_partitions(self):
        for features in ([0, 1, 9, 10], np.empty((4, 0)), np.zeros((4, 1, 1))):
            with self.subTest(features=np.shape(features)), self.assertRaises(ValueError):
                partition_labels(features, self.labels)
        for labels in ([0, 0, np.nan, np.nan], [0, 0, np.inf, np.inf],
                       [0, 0, None, None], ['a', 'a', np.nan, np.nan],
                       [[0], [0], [1], [1]], 1):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                partition_labels(self.x, labels)
        _, encoded, k = partition_labels(self.x, ['a', 'a', 'b', 'b'])
        np.testing.assert_array_equal(encoded, self.labels)
        self.assertEqual(k, 2)

    def test_stability_rejects_zip_truncation_and_missing_ids(self):
        for ids, labels in [(['a', 'b'], [0, 1, 7]), (['a', 'b', 'c'], [0, 1]),
                            (['a', None], [0, 1]), (['a', 'a'], [0, 1])]:
            with self.subTest(ids=ids, labels=labels), self.assertRaises(ValueError):
                aligned_stability(ids, labels, ['a', 'b'], [0, 1])
        self.assertEqual(aligned_stability(['b', 'a'], [7, 3], ['a', 'b'], [0, 1]),
                         {'common_n': 2, 'ARI': 1.0})

    def test_ch_zero_ssw_reports_positive_infinity_as_null(self):
        x = np.array([[0.0], [0.0], [10.0], [10.0]])
        result = calinski_harabasz_indices(x, self.labels)
        self.assertIsNone(result["CH"])
        self.assertIsNone(result["CH_per_n"])
        self.assertTrue(result["CH_is_infinite"])
        self.assertEqual(result["CH_status"], "positive_infinity_zero_within_dispersion")

    def test_ch_zero_over_zero_is_distinguished(self):
        result = calinski_harabasz_indices(np.zeros((4, 1)), self.labels)
        self.assertIsNone(result["CH"])
        self.assertIsNone(result["CH_per_n"])
        self.assertFalse(result["CH_is_infinite"])
        self.assertEqual(result["CH_status"], "undefined_zero_within_and_between_dispersion")

    def test_s_dbw_literal_eq14_is_diagnostic_and_can_reverse_definedness(self):
        point_masses = np.array([[0.0], [0.0], [10.0], [10.0]])
        self.assertEqual(s_dbw(point_masses, self.labels), 0.0)
        with self.assertRaisesRegex(ValueError, "literal Eq. 14"):
            s_dbw_paper(point_masses, self.labels)

        with self.assertRaisesRegex(ValueError, "zero centroid density"):
            s_dbw(self.x, self.labels)
        self.assertAlmostEqual(s_dbw_paper(self.x, self.labels), 2.0 + 0.25 / 20.5)

    def test_binary_path_network_indices_and_named_candidates(self):
        indices = network_indices(self.path, self.labels)
        candidates = network_quality_candidates(self.path, self.labels)
        self.assertAlmostEqual(indices["AVI"], 2.0 / 3.0)
        self.assertAlmostEqual(indices["AVU"], 1.0)
        self.assertAlmostEqual(candidates["TurboMQ"], 4.0 / 3.0)
        self.assertAlmostEqual(candidates["NewmanQ"], 1.0 / 6.0)

    def test_weighted_graph_variant_uses_original_weights(self):
        weighted_path = csr_matrix(
            [[0, 2, 0, 0], [2, 0, 1, 0], [0, 1, 0, 2], [0, 0, 2, 0]],
            dtype=float,
        )
        indices = network_indices(weighted_path, self.labels, binary=False)
        candidates = network_quality_candidates(weighted_path, self.labels, binary=False)
        self.assertAlmostEqual(indices["AVI"], 0.8)
        self.assertAlmostEqual(indices["AVU"], 1.0)
        self.assertAlmostEqual(candidates["TurboMQ"], 1.6)
        self.assertAlmostEqual(candidates["NewmanQ"], 0.3)
        self.assertIn("weighted", candidates["network_quality_convention"])

    def test_zero_edge_graph_has_turbomq_zero_and_newmanq_null(self):
        candidates = network_quality_candidates(csr_matrix((4, 4)), self.labels)
        self.assertEqual(candidates["TurboMQ"], 0.0)
        self.assertIsNone(candidates["NewmanQ"])
        self.assertIn("zero_total_edge_weight", candidates["NewmanQ_status"])

    def test_all_metrics_reports_mq_as_newman_modularity(self):
        result = all_metrics(self.x, self.labels, self.path)
        self.assertEqual(result["CH"], 162.0)
        self.assertEqual(result["CH_per_n"], 40.5)
        self.assertIn("S_Dbw_paper", result)
        self.assertIn("Eq14", result["S_Dbw_paper_convention"])
        self.assertAlmostEqual(result["TurboMQ"], 4.0 / 3.0)
        self.assertAlmostEqual(result["NewmanQ"], 1.0 / 6.0)
        self.assertAlmostEqual(result["MQ"], 1.0 / 6.0)
        self.assertIn("Newman modularity", result["MQ_status"])


if __name__ == "__main__":
    unittest.main()
