import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


types = load("type_validation")
january = load("january_check")


def synthetic(effect, seed=0, n=600):
    rng = np.random.default_rng(seed)
    region = rng.integers(0, 12, n)
    cluster = rng.integers(0, 4, n)
    population = np.exp(rng.normal(10, 1, n) + 0.3 * cluster)
    y = 0.2 * region + 0.5 * np.log(population) + effect * (cluster == 2) + rng.normal(0, 1, n)
    return pd.DataFrame({"region_code": region.astype(str), "cluster": cluster, "population": population,
                         "municipal_district_type": rng.integers(0, 3, n).astype(str), "y": y})


class IncrementTests(unittest.TestCase):
    outcome = {"column": "y", "transform": "identity", "label": "y"}

    def run_model(self, effect):
        return types.increment_over_population(synthetic(effect), self.outcome, "region_population",
                                               np.random.default_rng(1), 99)

    def test_type_effect_is_detected(self):
        result = self.run_model(1.0)
        self.assertGreater(result["delta_r2"], 0.02)
        self.assertLess(result["cluster_robust_wald"]["p_value"], 0.01)
        self.assertLessEqual(result["freedman_lane_permutation_p"], 0.02)
        self.assertAlmostEqual(result["type_coefficients"]["2"]["estimate"], 1.0, delta=0.3)

    def test_no_effect_keeps_increment_small(self):
        result = self.run_model(0.0)
        self.assertLess(result["delta_r2"], 0.01)
        self.assertGreater(result["freedman_lane_permutation_p"], 0.01)

    def test_r2_is_nested(self):
        result = self.run_model(0.5)
        self.assertLessEqual(result["r2_population_only"], result["r2_base"])
        self.assertLessEqual(result["r2_base"], result["r2_with_type"])


class JanuaryTests(unittest.TestCase):
    def test_trend_and_step_are_separated(self):
        t = np.arange(24)
        fit = january.trend_step(1 + 0.5 * t + 2.0 * (t >= 12))
        self.assertAlmostEqual(fit["monthly_trend"], 0.5)
        self.assertAlmostEqual(fit["step_at_january_2024"], 2.0)

    def test_seam_summary_reads_calendar(self):
        periods = [f"{y}-{m:02d}-01" for y in (2023, 2024) for m in range(1, 13)]
        series = pd.Series(np.arange(24, dtype=float), index=periods)
        summary = january.seam_summary(series)
        self.assertEqual(summary["december_2023_to_january_2024"], 1.0)
        self.assertEqual(summary["january_2024_minus_january_2023"], 12.0)
        self.assertEqual(summary["february_to_november_2024"], 9.0)


class PublishedResultsTests(unittest.TestCase):
    def test_published_validation_matches_configured_outputs(self):
        cfg = json.loads((ROOT / "configs/type_validation.json").read_text("utf-8"))
        path = ROOT / cfg["output"] / "results.json"
        if not path.exists():
            self.skipTest("results not built")
        result = json.loads(path.read_text("utf-8"))
        self.assertEqual(len(result["increment_over_population"]), len(cfg["outcomes"]) * len(cfg["controls"]))
        means = result["peer_bootstrap"]["mean_absolute_growth_error_pp"]
        for name, value in cfg["published_peer_means"].items():
            self.assertAlmostEqual(means[name], value, delta=5e-4)
        for contrast in result["peer_bootstrap"]["contrasts"]:
            low, high = contrast["ci_95"]
            self.assertLessEqual(low, contrast["mean_improvement"])
            self.assertLessEqual(contrast["mean_improvement"], high)


if __name__ == "__main__":
    unittest.main()
