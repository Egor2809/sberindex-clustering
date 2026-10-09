import unittest
import numpy as np
from sbercluster.selection_contrasts import denominator_direction, contrast_projection, contrast_basis_transform, contrast_annual, axis_decomposition


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.scaler = {'mode': 'log_ratios_to_total', 'level_weight': 0., 'ratio_iqr': [.2, .5, .3, .8, 1.]}
        self.monthly = np.random.default_rng(12).normal(size=(12, 30, 5))

    def test_rank_four_projector_and_monthly_denominator_invariance(self):
        projection = contrast_projection(self.scaler)
        np.testing.assert_allclose(projection @ projection, projection, atol=2e-16)
        direction = denominator_direction(self.scaler)
        np.testing.assert_allclose(projection @ direction, 0,
                                   atol=16 * np.finfo(float).eps * np.linalg.norm(direction))
        self.assertEqual(np.linalg.matrix_rank(projection), 4)
        basis = contrast_basis_transform(self.scaler)
        np.testing.assert_allclose(basis @ basis.T, projection, atol=1e-15)
        np.testing.assert_allclose(direction @ basis, 0, atol=2e-15)
        self.assertTrue(np.all(contrast_annual(self.monthly, self.scaler)[:, -1] == 0))
        changes = np.random.default_rng(20).normal(size=(12, 30))
        shifted = self.monthly - changes[:, :, None] * denominator_direction(self.scaler)
        np.testing.assert_allclose(contrast_annual(shifted, self.scaler),
                                   contrast_annual(self.monthly, self.scaler), atol=3e-15)

    def test_projection_and_annual_median_do_not_commute(self):
        correct = contrast_annual(self.monthly, self.scaler)
        wrong = np.median(self.monthly, axis=0) @ contrast_basis_transform(self.scaler)
        self.assertGreater(np.linalg.norm(correct - wrong), .1)

    def test_axis_partition_variance_has_expected_limits(self):
        direction = denominator_direction(self.scaler)
        x = np.array([-2., -1., 1., 2.])[:, None] * direction
        result = axis_decomposition(x, np.array([0, 0, 1, 1]), self.scaler)
        self.assertAlmostEqual(result['axis_total_fraction'], 1.)
        self.assertAlmostEqual(result['axis_between_fraction'], 1.)
        self.assertAlmostEqual(result['axis_eta_squared'], .9)


if __name__ == '__main__':
    unittest.main()
