"""Regression tests for the fitted feature space and its frozen application."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from sbercluster.features import make_slices, profile_matrix, transform_frozen, scalers_equal
from sbercluster.io import CATEGORIES, TOTAL


class FeatureContracts(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((Path(__file__).resolve().parents[1] / "configs/pilot.json").read_text("utf-8"))
        self.frame = pd.DataFrame([
            {"entity_id": entity, "period": period, TOTAL: 100 * base,
             **{category: (j + 1) * base ** 2 for j, category in enumerate(CATEGORIES)}}
            for period in ["2023-12-01", "2024-01-01"]
            for entity, base in [("a", 1), ("b", 2), ("c", 3)]])

    def test_frozen_transform_keeps_level_weight_and_dimensions(self):
        self.cfg["features"]["level_weight"] = .35
        slices, scaler = make_slices(self.frame, self.cfg)
        future = self.frame[self.frame.period == "2024-01-01"].copy()
        expected = slices[1][2]
        np.testing.assert_array_equal(transform_frozen(future, scaler), expected)
        self.assertEqual(expected.shape, (3, len(CATEGORIES) + 1))
        future[TOTAL] *= 2
        shifted = transform_frozen(future, scaler)
        np.testing.assert_allclose(shifted[:, -1] - expected[:, -1],
                                   np.sqrt(.35) * np.log(2) / scaler["level_iqr"])

    def test_hellinger_accepts_zero_and_constant_categories(self):
        self.cfg["features"]["mode"] = "hellinger_with_residual"
        self.cfg["data_contract"].update(common_denominator_verified=True,
                                          categories_disjoint_verified=True)
        self.frame[CATEGORIES] = 0.
        self.frame[CATEGORIES[0]] = 10.
        slices, scaler = make_slices(self.frame, self.cfg)
        future = self.frame[self.frame.period == "2024-01-01"]
        np.testing.assert_array_equal(slices[1][2], profile_matrix(future))
        np.testing.assert_array_equal(transform_frozen(future, scaler), profile_matrix(future))

    def test_legacy_zero_level_scaler_remains_compatible(self):
        slices, scaler = make_slices(self.frame, self.cfg)
        current = dict(scaler)
        del scaler["level_weight"]
        self.assertTrue(scalers_equal(scaler, current))
        self.assertFalse(scalers_equal(scaler, {**current, "level_weight": .1}))
        np.testing.assert_array_equal(transform_frozen(self.frame.iloc[:3], scaler), slices[0][2])

    def test_missing_key_does_not_silently_drop_an_observation(self):
        for column in ["entity_id", "period"]:
            with self.subTest(column=column):
                frame = self.frame.copy()
                frame.loc[0, column] = None
                with self.assertRaisesRegex(ValueError, "complete entity-month keys"):
                    make_slices(frame, self.cfg)

    def test_noncanonical_month_cannot_cross_lexical_cutoff(self):
        for period in ["2023-2-01", "2023-12-02", "2023-13-01"]:
            with self.subTest(period=period):
                frame = self.frame.copy()
                frame.loc[0, "period"] = period
                with self.assertRaises(ValueError):
                    make_slices(frame, self.cfg)

    def test_scaling_cannot_be_fitted_on_confirmation_data(self):
        self.cfg["features"]["calibration_end"] = "2024-01-01"
        with self.assertRaisesRegex(ValueError, "development period"):
            make_slices(self.frame, self.cfg)

    def test_invalid_frozen_scaler_fails_before_distances(self):
        _, scaler = make_slices(self.frame, self.cfg)
        for value in [0., np.nan, np.inf]:
            with self.subTest(scale=value):
                bad = copy.deepcopy(scaler)
                bad["ratio_iqr"][0] = value
                with self.assertRaisesRegex(ValueError, "Invalid frozen"):
                    transform_frozen(self.frame, bad)


if __name__ == "__main__":
    unittest.main()
