"""Small analytical tests; the optional Torch dependency is not needed by CI."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from sbercluster.dmon import (SparseDMoN, _torch_sparse, dmon_loss,
                                  fit_dmon, normalized_adjacency, PlateauMonitor)


@unittest.skipUnless(HAS_TORCH, "optional requirements-dmon.txt not installed")
class DMoNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.a = sparse.csr_matrix([
            [0, 2, .2, 0, 0], [2, 0, 0, .1, 0],
            [.2, 0, 0, 3, 0], [0, .1, 3, 0, 0], [0, 0, 0, 0, 0]])
        self.x = np.array([[1, 0], [1, .1], [0, 1], [.1, 1], [.5, .5]])

    @staticmethod
    def dense_loss(assignments, a):
        d = a.sum(dim=1)
        volume = d.sum()
        modularity_matrix = a - torch.outer(d, d) / volume
        spectral = -torch.trace(assignments.T @ modularity_matrix @ assignments) / volume
        collapse = assignments.sum(dim=0).norm() * np.sqrt(assignments.shape[1]) / len(assignments) - 1
        return spectral + collapse

    def test_plateau_requires_consecutive_loss_and_label_stability(self):
        m = PlateauMonitor({'check_every': 2, 'min_epochs': 4, 'patience': 2,
                            'min_delta': .01, 'max_label_change_fraction': 0})
        self.assertFalse(m.observe(0, -.1, [0, 1]))
        self.assertFalse(m.observe(2, -.2, [0, 1]))
        self.assertFalse(m.observe(4, -.205, [0, 1]))
        self.assertFalse(m.observe(6, -.206, [1, 1]))
        self.assertFalse(m.observe(8, -.207, [1, 1]))
        self.assertTrue(m.observe(10, -.208, [1, 1]))
        self.assertEqual(m.history[-1]['consecutive_plateau_windows'], 2)

    def test_plateau_fit_stops_and_preserves_best_checkpoint(self):
        rule = {'check_every': 2, 'min_epochs': 4, 'patience': 2,
                'min_delta': 1, 'max_label_change_fraction': 1}
        _, info = fit_dmon(self.x, self.a, 2, 17, 20, hidden_dim=4, plateau=rule)
        self.assertEqual(info['stop_reason'], 'training_plateau')
        self.assertEqual(info['epochs'], 6)
        self.assertEqual(len(info['trace']), 7)
        self.assertEqual(info['best_objective'], min(r['objective'] for r in info['trace']))
        self.assertTrue(info['plateau_reached'])
        self.assertEqual(info['max_epochs'], 20)

    def test_sparse_value_and_assignment_gradient_equal_dense_equation(self):
        a = _torch_sparse(self.a, torch.float64)
        logits = torch.tensor([[1., .1], [.7, .2], [.2, 1.], [.1, .8], [.3, .4]],
                              dtype=torch.float64, requires_grad=True)
        assignments = logits.softmax(dim=1)
        actual = dmon_loss(assignments, a)[0]
        expected = self.dense_loss(assignments, a.to_dense())
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
        actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-13, rtol=1e-13)
        self.assertGreater(float(actual_grad.norm()), 0)

    def test_loss_and_assignment_gradient_are_invariant_to_extreme_weight_units(self):
        logits = torch.tensor([[1., .1], [.7, .2], [.2, 1.], [.1, .8], [.3, .4]],
                              dtype=torch.float32, requires_grad=True)
        assignments = logits.softmax(dim=1)
        baseline = dmon_loss(assignments, _torch_sparse(self.a))[0]
        baseline_gradient = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
        for scale in (1e-25, 1e25):
            actual = dmon_loss(assignments, _torch_sparse(self.a * scale))[0]
            gradient = torch.autograd.grad(actual, logits, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(actual))
            torch.testing.assert_close(actual, baseline, atol=2e-7, rtol=2e-7)
            torch.testing.assert_close(gradient, baseline_gradient, atol=2e-7, rtol=2e-7)

    def test_small_asymmetric_graph_is_not_hidden_by_absolute_tolerance(self):
        graph = sparse.csr_matrix([[0, 1e-7], [0, 0]])
        with self.assertRaisesRegex(ValueError, 'symmetric'):
            fit_dmon(np.array([[0.], [1.]]), graph, 2, 17, 1)

    def test_float32_graph_volume_overflow_rejected_before_training(self):
        graph = self.a * 1e38
        with self.assertRaisesRegex(ValueError, 'float32 range'):
            fit_dmon(self.x, graph, 2, 17, 1)

    def test_graph_weight_gradient_equal_dense_equation(self):
        indices = torch.tensor([[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]])
        weights = torch.tensor([2., 3., .2], dtype=torch.float64, requires_grad=True)
        values = weights.repeat_interleave(2)
        graph = torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()
        assignments = torch.tensor([[.8,.2], [.7,.3], [.1,.9], [.2,.8]], dtype=torch.float64)
        actual = dmon_loss(assignments, graph)[0]
        dense = torch.zeros((4,4), dtype=torch.float64).index_put(tuple(indices), values)
        expected = self.dense_loss(assignments, dense)
        grad_a = torch.autograd.grad(actual, weights, retain_graph=True)[0]
        grad_b = torch.autograd.grad(expected, weights)[0]
        torch.testing.assert_close(grad_a, grad_b, atol=1e-13, rtol=1e-13)
        self.assertGreater(float(grad_a.norm()), 0)

    def test_attribute_and_parameter_gradients_equal_dense_network(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17)
            model = SparseDMoN(2, 4, 2).double()
        x = torch.tensor(self.x, dtype=torch.float64, requires_grad=True)
        norm = _torch_sparse(normalized_adjacency(self.a), torch.float64)
        graph = _torch_sparse(self.a, torch.float64)
        actual = dmon_loss(model(x, norm), graph)[0]
        projected = x @ model.kernel
        hidden = torch.nn.functional.selu(norm.to_dense() @ projected
                                          + projected * model.skip_weight + model.bias)
        assignments = (hidden @ model.assignment.weight.T + model.assignment.bias).softmax(1)
        expected = self.dense_loss(assignments, graph.to_dense())
        variables = (x, *model.parameters())
        grad_a = torch.autograd.grad(actual, variables, retain_graph=True)
        grad_b = torch.autograd.grad(expected, variables)
        for left, right in zip(grad_a, grad_b):
            torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
        self.assertGreater(float(grad_a[0].norm()), 0)

    def test_hard_partition_matches_manual_modularity_and_collapse(self):
        a = _torch_sparse(sparse.block_diag([[[0., 1.], [1., 0.]]] * 2), torch.float64)
        balanced = torch.tensor([[1.,0.], [1.,0.], [0.,1.], [0.,1.]], dtype=torch.float64)
        total, spectral, collapse = dmon_loss(balanced, a)
        self.assertAlmostEqual(float(spectral), -.5)
        self.assertAlmostEqual(float(collapse), 0)
        self.assertAlmostEqual(float(total), -.5)
        collapsed = torch.tensor([[1.,0.]] * 4, dtype=torch.float64)
        total, spectral, collapse = dmon_loss(collapsed, a)
        self.assertAlmostEqual(float(spectral), 0)
        self.assertAlmostEqual(float(collapse), np.sqrt(2)-1, places=6)

    def test_normalization_no_loops_and_isolates_finite(self):
        normalized = normalized_adjacency(self.a).toarray()
        self.assertTrue(np.isfinite(normalized).all())
        np.testing.assert_array_equal(normalized.diagonal(), np.zeros(5))
        np.testing.assert_array_equal(normalized[4], np.zeros(5))
        np.testing.assert_allclose(normalized, normalized.T)

    def test_seed_checkpoint_trace_and_original_inputs(self):
        original = self.a.copy()
        x_before = self.x.copy()
        with tempfile.TemporaryDirectory() as directory:
            labels, info, soft = fit_dmon(self.x, self.a, 2, 1729, 5,
                hidden_dim=4, return_assignments=True, output_dir=directory)
            labels2, info2, soft2 = fit_dmon(self.x, self.a, 2, 1729, 5,
                hidden_dim=4, return_assignments=True)
            np.testing.assert_array_equal(labels, labels2)
            np.testing.assert_array_equal(soft, soft2)
            self.assertEqual(info, info2)
            self.assertEqual(len(info['trace']), 6)
            self.assertEqual(info['best_objective'], min(row['objective'] for row in info['trace']))
            self.assertEqual(info['stop_reason'], 'fixed_epoch_budget')
            self.assertNotIn('converged', info)
            self.assertEqual(info['isolated_nodes'], 1)
            np.testing.assert_allclose(soft.sum(axis=1), 1, atol=1e-6)
            self.assertEqual(json.loads((Path(directory)/'info.json').read_text()), info)
            with np.load(Path(directory)/'best_model.npz', allow_pickle=False) as saved:
                model = SparseDMoN(2, 4, 2)
                model.load_state_dict({key: torch.from_numpy(saved[key]) for key in saved.files})
            with torch.no_grad():
                restored = model(torch.tensor(self.x, dtype=torch.float32),
                                 _torch_sparse(normalized_adjacency(self.a.astype(np.float32)))).numpy()
            np.testing.assert_array_equal(restored, soft)
        np.testing.assert_array_equal(self.x, x_before)
        self.assertEqual((self.a != original).nnz, 0)

    def test_fit_changes_when_attributes_or_graph_change(self):
        _, _, baseline = fit_dmon(self.x, self.a, 2, 1729, 2, hidden_dim=4, return_assignments=True)
        changed_x = self.x.copy()
        changed_x[0] += [2, -1]
        _, _, features = fit_dmon(changed_x, self.a, 2, 1729, 2, hidden_dim=4, return_assignments=True)
        changed_a = self.a.copy().tolil()
        changed_a[0, 2] = changed_a[2, 0] = 4
        _, _, graph = fit_dmon(self.x, changed_a.tocsr(), 2, 1729, 2,
                              hidden_dim=4, return_assignments=True)
        self.assertFalse(np.allclose(baseline, features))
        self.assertFalse(np.allclose(baseline, graph))

    def test_invalid_inputs_rejected(self):
        for graph in [sparse.csr_matrix((5,5)), sparse.eye(5), self.a.toarray(),
                      -self.a, sparse.triu(self.a)]:
            with self.assertRaises(ValueError):
                fit_dmon(self.x, graph, 2, 17, 1)
        for args in [{'n_clusters':1}, {'n_clusters':6}, {'epochs':0},
                     {'seed':-1}, {'hidden_dim':0}, {'learning_rate':float('nan')}]:
            config = dict(n_clusters=2, seed=17, epochs=1)
            config.update(args)
            with self.assertRaises(ValueError):
                fit_dmon(self.x, self.a, **config)


if __name__ == '__main__':
    unittest.main()
