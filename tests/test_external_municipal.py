import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load("prepare_external_municipal")
val = load("external_municipal_validation")


def bridge_row(entity_id, oktmo):
    return {
        "entity_id": entity_id, "territory_id": entity_id[4:], "cluster": "0", "region_code": "1",
        "region_name": "R", "municipal_district_type": "городской округ",
        "intracity_moscow_petersburg": "False", "oktmo": oktmo,
    }


class OktmoMatchingTests(unittest.TestCase):
    def test_dashed_code_is_reduced_to_eight_digits(self):
        self.assertEqual(prep.oktmo8("79-701-000-000"), "79701000")
        self.assertEqual(prep.oktmo8("45301000"), "45301000")
        with self.assertRaises(ValueError):
            prep.oktmo8("79-701-405-101")

    def test_exact_code_match_ambiguity_and_absence(self):
        bridge = [bridge_row("tid_1", "79-701-000-000"), bridge_row("tid_2", "45-301-000-000"), bridge_row("tid_3", "40-301-000-000")]
        rosstat = {
            "79701000": [(1000.0, 800.0, "Город")],
            "45301000": [(500.0, 500.0, "A"), (700.0, 700.0, "B")],
        }
        total = prep.LABOUR_TOTAL
        population = {("79701000", "Всего"): [(990.0, "Город", "CD")], ("79701000", "Старше трудоспособного возраста"): [(250.0, "Город", "CD")]}
        headcount = {
            ("79701000", total): [(200.0, "Город", "CD")],
            ("79701000", "Раздел С Обрабатывающие производства"): [(50.0, "Город", "CD")],
            ("40301000", total): [(10.0, "X", "CD"), (12.0, "X", "CD")],
        }
        wage = {("79701000", total): [(60000.0, "Город", "CD")], ("45301000", total): [(1.0, "Y", prep.ANOMALY)]}
        rows = {row["entity_id"]: row for row in prep.build_rows(bridge, rosstat, population, headcount, wage)}

        self.assertEqual(rows["tid_1"]["rosstat_status"], "matched")
        self.assertEqual(rows["tid_1"]["population"], 1000)
        self.assertAlmostEqual(rows["tid_1"]["urban_share_2024"], 0.8)
        self.assertAlmostEqual(rows["tid_1"]["above_working_age_share_2023"], 250 / 990)
        self.assertAlmostEqual(rows["tid_1"]["jobs_per_1000"], 1000 * 200 / 990)
        self.assertAlmostEqual(rows["tid_1"]["share_manufacturing"], 0.25)
        self.assertNotIn("share_trade", rows["tid_1"])

        self.assertEqual(rows["tid_2"]["rosstat_status"], "ambiguous")
        self.assertNotIn("population_2024_rosstat", rows["tid_2"])
        self.assertEqual(rows["tid_2"]["wage_status"], "flagged_anomalous")
        self.assertIsNone(rows["tid_2"]["wage_2023"])

        self.assertEqual(rows["tid_3"]["rosstat_status"], "no_code_match")
        self.assertEqual(rows["tid_3"]["population_source"], "none")
        self.assertEqual(rows["tid_3"]["headcount_status"], "ambiguous")

    def test_cyrillic_section_letters_are_normalised(self):
        self.assertEqual(prep.section_letter("Раздел А Сельское хозяйство"), "A")
        self.assertEqual(prep.section_letter("Раздел C Обрабатывающие производства"), "C")
        self.assertIsNone(prep.section_letter(prep.LABOUR_TOTAL))


class EtaSquaredTests(unittest.TestCase):
    def test_eta_squared_matches_hand_computation(self):
        y = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 9.0])
        groups = np.array([0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(val.eta_squared(y, groups), 54 / 58)
        self.assertAlmostEqual(val.within_eta_squared(y, groups, ["all"] * 6), 54 / 58)
        self.assertIsNone(val.eta_squared(np.ones(4), np.array([0, 0, 1, 1])))

    def test_region_control_removes_region_shifts(self):
        rng = np.random.default_rng(3)
        groups = np.tile([0, 1, 2], 20)
        regions = np.repeat(np.arange(6), 10)
        y = groups * 0.5 + rng.normal(size=60)
        shifted = y + regions * 10.0
        self.assertAlmostEqual(
            val.within_eta_squared(y, groups, regions), val.within_eta_squared(shifted, groups, regions)
        )
        self.assertLess(val.eta_squared(shifted, groups), 0.05)


if __name__ == "__main__":
    unittest.main()
