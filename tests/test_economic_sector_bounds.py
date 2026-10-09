import itertools
import unittest
import numpy as np
from sbercluster.economic_sector_bounds import squared_loss_difference_bounds


class SectorBoundsTests(unittest.TestCase):
    def test_exact_bounds_match_all_endpoint_fills_with_unequal_regions(self):
        y = np.array([.1, np.nan, .8, np.nan, np.nan])
        a = np.array([.1, .3, .8, .9, .4])
        b = np.array([.5, .7, .6, .1, .6])
        regions = np.array([1, 1, 2, 2, 2])
        result = squared_loss_difference_bounds(y, a, b, regions)
        values, municipal = [], []
        for values_missing in itertools.product([0., 1.], repeat=3):
            target = y.copy()
            target[~np.isfinite(y)] = values_missing
            loss = (target - b) ** 2 - (target - a) ** 2
            values.append(np.mean([loss[regions == r].mean() for r in [1, 2]]))
            municipal.append(loss.mean())
        self.assertAlmostEqual(result["equal_region_lower"], min(values))
        self.assertAlmostEqual(result["equal_region_upper"], max(values))
        self.assertAlmostEqual(result["municipality_lower"], min(municipal))
        self.assertAlmostEqual(result["municipality_upper"], max(municipal))

    def test_region_without_any_observed_target_retains_weight_and_denominator(self):
        result = squared_loss_difference_bounds([0., np.nan, np.nan], [0., 0., 0.], [1., 1., 1.], np.array([1, 2, 2]))
        self.assertEqual(result["eligible_regions"], 2)
        self.assertEqual(result["observed_regions"], 1)
        self.assertEqual(result["regions_with_no_observed_target"], [2])
        self.assertEqual(result["equal_region_lower"], 0.)
        self.assertAlmostEqual(result["municipality_lower"], -1 / 3)

    def test_known_targets_collapse_interval_to_realized_loss(self):
        y = np.array([0., .2, .8])
        a, b = y.copy(), np.ones(3) * .5
        result = squared_loss_difference_bounds(y, a, b, np.array([1, 1, 2]))
        self.assertEqual(result["equal_region_lower"], result["equal_region_upper"])
        self.assertEqual(result["conclusion"], "candidate_better_for_every_missing_fill")

    def test_all_missing_nonmonotonic_candidate_and_reference_reverse_bounds(self):
        a, b, r = [.1, .7], [.8, .2], np.array([1, 2])
        first = squared_loss_difference_bounds([np.nan, np.nan], a, b, r)
        second = squared_loss_difference_bounds([np.nan, np.nan], b, a, r)
        self.assertAlmostEqual(first["equal_region_lower"], -second["equal_region_upper"])
        self.assertAlmostEqual(first["equal_region_upper"], -second["equal_region_lower"])

    def test_invalid_observed_data_never_clipped_or_silently_dropped(self):
        for y in [[-0.01], [1.01], [np.inf]]:
            with self.assertRaises(ValueError):
                squared_loss_difference_bounds(y, [.5], [.5], np.array([1]))

    def test_missing_predictions_and_unbounded_predictions_rejected(self):
        for pred in [[np.nan], [1.1], [-.1]]:
            with self.assertRaises(ValueError):
                squared_loss_difference_bounds([np.nan], pred, [.5], np.array([1]))


if __name__ == "__main__":
    unittest.main()
