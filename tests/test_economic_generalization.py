import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import numpy as np
from sbercluster.economic_generalization import (fit_region_representation, fit_predict_probe,
        polynomial_design, score_predictions, type_design)


class EconomicGeneralizationTests(unittest.TestCase):
    def test_fold_scaler_matches_other_region_raw_months(self):
        rng = np.random.default_rng(410)
        raw = rng.normal(size=(12, 48, 5))
        regions = np.repeat([1, 2, 3], 16)
        result = fit_region_representation(raw, regions, 2)
        ref = raw[:, regions != 2].reshape(-1, 5)
        np.testing.assert_array_equal(result["artifact"]["scaler"]["ratio_center"], np.median(ref, axis=0))
        np.testing.assert_array_equal(result["artifact"]["scaler"]["ratio_iqr"], np.quantile(ref, .75, axis=0) - np.quantile(ref, .25, axis=0))
        self.assertEqual(result["artifact"]["calibration_observations"], 32 * 12)
        self.assertNotIn(2, result["artifact"]["training_regions"])

    def test_missing_outcomes_cannot_remove_expense_fit_entities(self):
        # The representation API deliberately has no outcome argument.
        raw = np.random.default_rng(8).normal(size=(12, 24, 5))
        result = fit_region_representation(raw, np.repeat([1, 2, 3], 8), 1)
        self.assertEqual(result["artifact"]["training_entities"], 16)

    def test_reject_degenerate_iqr_nonfinite_unknown_fold_and_region_identity(self):
        raw = np.ones((12, 24, 5))
        for regions, held_out in [(np.repeat([1, 2, 3], 8), 1), (np.repeat([1, 2, 3], 8), 4),
                                   (np.repeat([1., 2., 3.], 8), 1)]:
            with self.assertRaises(ValueError):
                fit_region_representation(raw, regions, held_out)
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            fit_region_representation(raw, np.repeat([1, 2, 3], 8), 1)

    def test_quadratic_uses_training_scale_and_additive_dummies(self):
        train = np.array([[0., 2.], [1., 4.], [0., 6.], [1., 8.]])
        test = np.array([[1., 1000.]])
        a, b, fitted = polynomial_design(train, test, [1])
        self.assertEqual(fitted["mean"], [5.])
        np.testing.assert_array_equal(a[:, 0], train[:, 0])
        self.assertEqual(b[0, 0], 1.)
        self.assertEqual(a.shape[1], 3)

    def test_controlled_duplicate_columns_do_not_create_false_rank(self):
        x = np.arange(12, dtype=float)
        design = np.column_stack([x, 2 * x, np.ones(12)])
        prediction, metadata = fit_predict_probe(design, 3 * x + 5, design)
        np.testing.assert_allclose(prediction, 3 * x + 5, atol=1e-12)
        self.assertEqual(metadata["rank"], 2)

    def test_unknown_type_is_encoded_without_test_fitted_vocabulary(self):
        train, test, vocabulary = type_design(["a", "a", "b"], ["c", "b"])
        self.assertEqual(vocabulary, ["a", "b"])
        np.testing.assert_array_equal(test, [[0, 0], [0, 1]])

    def test_centered_error_loss_invariant_to_arbitrary_region_offsets(self):
        y = np.array([1., 2., 3., 4., 6., 8.])
        regions = np.repeat([1, 2], 3)
        predictions = {"first": np.arange(6.), "second": np.arange(6.) + np.repeat([100., -50.], 3)}
        rows = score_predictions(y, predictions, regions, bootstrap=99)
        self.assertEqual(rows[0]["equal_region_centered_error_sample_variance"], rows[1]["equal_region_centered_error_sample_variance"])
        self.assertNotEqual(rows[0]["equal_region_mse"], rows[1]["equal_region_mse"])
        self.assertEqual(rows[1]["comparisons"]["first"]["equal_region_centered_error_variance_improvement"], 0.)

    def test_paired_intervals_are_symmetric_and_singletons_excluded_only_from_centered_loss(self):
        y = np.arange(5.)
        predictions = {"a": np.zeros(5), "b": np.ones(5)}
        rows = score_predictions(y, predictions, np.array([1, 1, 2, 2, 3]), bootstrap=99)
        self.assertEqual(rows[0]["regions"], 3)
        self.assertEqual(rows[0]["regions_with_pairs"], 2)
        a = rows[0]["comparisons"]["b"]
        b = rows[1]["comparisons"]["a"]
        np.testing.assert_allclose(a["conditional_region_bootstrap_95"], -np.array(b["conditional_region_bootstrap_95"])[::-1])
        np.testing.assert_allclose(a["conditional_centered_region_bootstrap_95"], -np.array(b["conditional_centered_region_bootstrap_95"])[::-1])

    def test_nonfinite_probe_and_unpaired_prediction_rejected(self):
        with self.assertRaises(ValueError):
            fit_predict_probe([[1], [2], [3]], [1, np.nan, 3], [[4]])
        with self.assertRaises(ValueError):
            score_predictions(np.arange(4.), {"bad": np.ones(3)}, np.ones(4), bootstrap=99)

    def test_windows_priority_uses_platform_constant(self):
        from scripts.validate_economic_generalization import set_runtime_priority
        process = Mock()
        module = SimpleNamespace(Process=lambda: process, BELOW_NORMAL_PRIORITY_CLASS=16384)
        with patch("scripts.validate_economic_generalization.sys.platform", "win32"), patch.dict("sys.modules", {"psutil": module}):
            set_runtime_priority()
        process.nice.assert_called_once_with(16384)

    def test_posix_priority_has_no_windows_constant_dependency(self):
        from scripts.validate_economic_generalization import set_runtime_priority
        for current, expected in [(0, 10), (15, 15)]:
            process = Mock()
            process.nice.return_value = current
            module = SimpleNamespace(Process=lambda: process)
            with patch("scripts.validate_economic_generalization.sys.platform", "linux"), patch.dict("sys.modules", {"psutil": module}):
                set_runtime_priority()
            self.assertEqual(process.nice.call_args_list[-1].args, (expected,))


if __name__ == "__main__":
    unittest.main()
