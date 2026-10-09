import unittest
import numpy as np
from sbercluster.metrics import s_dbw, s_dbw_diagnostics


class DensityDiagnostics(unittest.TestCase):
    def test_two_empty_balls_are_undefined(self):
        x = np.array([[-1.], [1.], [9.], [11.]])
        labels = np.array([0, 0, 1, 1])
        result = s_dbw_diagnostics(x, labels)
        self.assertAlmostEqual(result['radius'], np.sqrt(2) / 2)
        self.assertEqual(result['centroid_densities'], [0, 0])
        self.assertEqual(result['undefined_pairs'], 1)
        with self.assertRaises(ValueError):
            s_dbw(x, labels)

    def test_one_empty_ball_does_not_make_a_pair_undefined(self):
        x = np.array([[-1.], [1.], [10.], [10.]])
        labels = np.array([0, 0, 1, 1])
        result = s_dbw_diagnostics(x, labels)
        self.assertEqual(result['centroid_densities'], [0, 2])
        self.assertEqual(result['undefined_pairs'], 0)
        self.assertTrue(np.isfinite(s_dbw(x, labels)))


if __name__ == '__main__':
    unittest.main()
