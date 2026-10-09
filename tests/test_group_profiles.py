import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("group_profiles", ROOT / "scripts" / "group_profiles.py")
group_profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(group_profiles)


class GroupProfilesTests(unittest.TestCase):
    def test_eta_squared_bounds(self):
        self.assertAlmostEqual(group_profiles.eta_squared([1, 1, 5, 5], ["a", "a", "b", "b"]), 1.0)
        self.assertAlmostEqual(group_profiles.eta_squared([1, 5, 1, 5], ["a", "a", "b", "b"]), 0.0)

    def test_mirkin_relative_deviation(self):
        self.assertAlmostEqual(group_profiles.mirkin(0.15, 0.1), 0.5)

    def test_published_profiles(self):
        data = json.loads((ROOT / "reports" / "temporal-groups" / "profiles.json").read_text("utf-8"))
        self.assertEqual(len(data["groups"]), 7)
        self.assertEqual(data["north"]["north_municipalities"], 72)
        self.assertEqual(data["north"]["north_in_north_groups"], 72)
        wage = next(row for row in data["explained"] if row["outcome"] == "wage_2023")
        self.assertGreater(wage["eta2_groups"], wage["eta2_kmeans4"])


if __name__ == "__main__":
    unittest.main()
