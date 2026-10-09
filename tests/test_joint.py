import unittest
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
from sbercluster.joint import joint_objective, fit_graph_regularized_kmeans
from scripts.verify_joint_refinement import exact_stationarity, verify
from sbercluster.round2 import same_region_graph


class JointTests(unittest.TestCase):
    def setUp(self):
        self.x = np.arange(4, dtype=float)[:, None]
        self.z = np.array([0, 0, 1, 1])
        self.a = csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])

    def test_hand_calculated_objective_uses_each_edge_once(self):
        score = joint_objective(self.x, self.a, self.z, .6)
        self.assertAlmostEqual(score['SSE'], 1)
        self.assertAlmostEqual(score['TSS'], 5)
        self.assertAlmostEqual(score['graph_cut_fraction'], 1/3)
        self.assertAlmostEqual(score['objective'], .4)

    def test_alpha_zero_reuses_baseline_exactly(self):
        z, info = fit_graph_regularized_kmeans(self.x, self.a, self.z, alpha=0)
        np.testing.assert_array_equal(z, self.z)
        self.assertEqual(info['sweeps'], 0)
        self.assertAlmostEqual(info['objective'], .2)
        self.assertFalse(info['converged'])
        self.assertEqual(info['convergence_status'], 'baseline_reused_without_optimization')
        supplied = np.array([10, 10, 20, 20])
        labels, _ = fit_graph_regularized_kmeans(self.x, self.a, supplied, alpha=0)
        np.testing.assert_array_equal(labels, supplied)

    def test_centroid_reoptimization_move_is_not_missed(self):
        # Frozen-centroid assignment regards this split as fixed, but moving
        # point 3 changes SSE 9 -> 26/3 without changing the path's cut mass.
        x = np.array([0., 3., 4., 7.])[:, None]
        labels, info = fit_graph_regularized_kmeans(x, self.a, self.z, alpha=.1)
        self.assertLess(info['objective'], joint_objective(x, self.a, self.z, .1)['objective'])
        self.assertAlmostEqual(info['SSE'], 26/3)
        self.assertTrue(info['converged'])
        self.assertEqual(info['algorithm_revision'], 'exact_centroid_moves_v2')
        self.assertEqual(len(np.unique(labels)), 2)

    def test_converged_solution_has_no_improving_reoptimized_vertex_move(self):
        rng = np.random.default_rng(71)
        # An independent objective oracle enumerates every valid transfer;
        # this checks counts/centroids, graph mass, and sequential updates.
        for seed in range(12):
            x = rng.normal(size=(9, 3))
            weights = rng.uniform(size=(9, 9))
            weights = np.triu(weights, 1)
            a = csr_matrix(weights + weights.T)
            initial = np.arange(9) % 3
            labels, info = fit_graph_regularized_kmeans(x, a, initial, alpha=.8, seed=seed)
            self.assertTrue(info['converged'])
            self.assertTrue(np.all(np.diff([r['objective'] for r in info['trace']]) <= 1e-12))
            counts = np.bincount(labels, minlength=3)
            self.assertTrue(np.all(counts > 0))
            for i in range(9):
                if counts[labels[i]] == 1:
                    continue
                for destination in range(3):
                    candidate = labels.copy()
                    candidate[i] = destination
                    self.assertGreaterEqual(joint_objective(x, a, candidate, .8)['objective'],
                                            info['objective'] - 1e-12)

    def test_capped_fit_does_not_claim_coordinate_convergence(self):
        x = np.array([0., 3., 4., 7.])[:, None]
        _, info = fit_graph_regularized_kmeans(x, self.a, self.z, alpha=.1, max_sweeps=1)
        self.assertFalse(info['converged'])
        self.assertEqual(info['convergence_status'], 'sweep_cap_reached')
        self.assertEqual(info['warnings'], ['sweep_cap_reached'])

    def test_overflowing_objective_inputs_rejected(self):
        with self.assertRaisesRegex(ValueError, 'finite TSS'):
            joint_objective(self.x * 1e200, self.a, self.z, .1)
        with self.assertRaisesRegex(ValueError, 'finite TSS'):
            joint_objective(self.x, self.a * 1e308, self.z, .1)

    def test_public_stationarity_oracle_detects_frozen_centroid_false_convergence(self):
        x = np.array([0., 3., 4., 7.])[:, None]
        before = exact_stationarity(x, self.a, self.z, alpha=.1)
        self.assertEqual(before['improving_vertices'], 2)
        self.assertAlmostEqual(before['best_move_delta'], -1/75)
        labels, _ = fit_graph_regularized_kmeans(x, self.a, self.z, alpha=.1)
        self.assertTrue(exact_stationarity(x, self.a, labels, alpha=.1)['coordinate_stationary'])

    def test_public_verifier_refuses_existing_destination_before_any_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / 'keep.txt'
            sentinel.write_text('preserved', encoding='utf-8')
            with patch('scripts.verify_joint_refinement.fit_graph_regularized_kmeans') as fit:
                with self.assertRaises(FileExistsError):
                    verify(Path(__file__).resolve().parents[1], directory)
                fit.assert_not_called()
            self.assertEqual(sentinel.read_text('utf-8'), 'preserved')
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_refinement_artifact_manifest_labels_and_stationarity(self):
        root = Path(__file__).resolve().parents[1]
        output = root / 'reports/technical-core-2026-10-03/joint-refinement'
        if not output.exists():
            self.skipTest('Refinement evidence not generated in this checkout')
        manifest = json.loads((output / 'manifest.json').read_text('utf-8'))
        for name, expected in manifest['files_sha256'].items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected, name)
        provenance = json.loads((output / 'provenance.json').read_text('utf-8'))
        for name, expected in provenance['source_objects'].items():
            source = output / 'source_objects' / (expected + '.py')
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), expected, name)
        features = np.load(output / 'features.npy', allow_pickle=False)
        labels = json.loads((output / 'labels.json').read_text('utf-8'))
        regions = json.loads((output / 'regions.json').read_text('utf-8'))
        self.assertEqual(labels['ids'], regions['ids'])
        road = load_npz(root / 'reports/graphs/road_2024_2016.npz')
        regional = same_region_graph(regions['region_codes'])
        rows = json.loads((output / 'comparisons.json').read_text('utf-8'))
        self.assertEqual(len(rows), 8)
        for row in rows:
            graph = regional if row['candidate'].startswith('joint_same_region') else road
            z = np.asarray(labels['labels'][row['id']])
            self.assertEqual(len(z), len(labels['ids']))
            self.assertTrue(exact_stationarity(features, graph, z)['coordinate_stationary'])
            self.assertAlmostEqual(joint_objective(features, graph, z, .25)['objective'], row['fit']['objective'])

    def test_real_graph_changes_solution_monotonically(self):
        a = csr_matrix([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]])
        z, info = fit_graph_regularized_kmeans(self.x, a, self.z, alpha=3)
        self.assertFalse(np.array_equal(z, self.z))
        self.assertLess(info['graph_cut_fraction'], 1)
        self.assertTrue(np.all(np.diff([r['objective'] for r in info['trace']]) <= 1e-12))
        self.assertEqual(len(np.unique(z)), 2)
        self.assertTrue(info['converged'])
        np.testing.assert_array_equal(self.z, [0,0,1,1])

    def test_uniform_scaling_translation_and_seed_reproduction(self):
        z, info = fit_graph_regularized_kmeans(self.x, self.a, self.z, alpha=.7)
        zz, scaled = fit_graph_regularized_kmeans(7*self.x+100, 3*self.a, self.z, alpha=.7)
        np.testing.assert_array_equal(z, zz)
        self.assertAlmostEqual(info['objective'], scaled['objective'])

    def test_invalid_data_rejected(self):
        for a in [csr_matrix((4,4)), csr_matrix(np.eye(4)),
                  csr_matrix([[0,1,0,0],[0,0,1,0],[0,1,0,1],[0,0,1,0]])]:
            with self.assertRaises(ValueError):
                fit_graph_regularized_kmeans(self.x, a, self.z)
        for alpha in [-1, float('nan')]:
            with self.assertRaises(ValueError):
                fit_graph_regularized_kmeans(self.x, self.a, self.z, alpha=alpha)
        with self.assertRaises(ValueError):
            fit_graph_regularized_kmeans(np.ones((4,1)), self.a, self.z)


if __name__ == '__main__':
    unittest.main()
