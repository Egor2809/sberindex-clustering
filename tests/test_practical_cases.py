"""Check preserved inputs, comparison denominators and the descriptive peer rule."""
import base64
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_practical_cases import ROOT, compile_cases, eligible, html_block, markdown_report, read_csv, read_payload


class PracticalCasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = read_payload(ROOT / "docs/index.html")
        cls.bridge = read_csv(ROOT / "reports/external-v4/municipality_bridge.csv")
        cls.cohort = read_csv(ROOT / "reports/external-national-2026-10-03/cohort-2023.csv")
        cls.paired = read_csv(ROOT / "reports/temporal-core-2026-10-03/paired-reference/assignments.csv")
        cls.abazinsky = json.loads((ROOT / "reports/round2-cases/abazinsky.json").read_text("utf-8"))
        cls.result = compile_cases(cls.data, copy.deepcopy(cls.bridge), cls.cohort, cls.paired, cls.abazinsky)

    def test_decode_compressed_and_plain_payloads(self):
        original = {"entities": [{"id": "tid_1", "name": "Тест"}]}
        raw = json.dumps(original, ensure_ascii=False).encode()
        packed = {"encoding": "gzip-base64", "data": base64.b64encode(gzip.compress(raw)).decode()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "atlas.html"
            for value in (packed, original):
                path.write_text("const packed=" + json.dumps(value) + ";", "utf-8")
                self.assertEqual(read_payload(path), original)
            path.write_text("const packed={};", "utf-8")
            with self.assertRaises(ValueError):
                read_payload(path)

    def test_rule_is_symmetric_and_includes_exact_boundaries(self):
        a = {"municipal_district_type": "район", "expense_2023": 100.0}
        b = {"municipal_district_type": "район", "expense_2023": 125.0}
        self.assertTrue(eligible(a, b))
        self.assertTrue(eligible(b, a))
        b["expense_2023"] = 125.01
        self.assertFalse(eligible(a, b))
        b["expense_2023"] = 100.0
        b["municipal_district_type"] = "город"
        self.assertFalse(eligible(a, b))

    def test_saved_abazinsky_distances_ids_and_rule(self):
        rows = self.result["analogue_case"]["neighbors"]
        self.assertEqual(len(rows), 15)
        self.assertEqual([row["entity_id"] for row in rows], [row["entity_id"] for row in self.abazinsky["neighbors"]])
        self.assertEqual([row["entity_id"] for row in rows if row["eligible_factor_1_25"]], ["tid_201"])
        self.assertEqual(sum(row["same_type"] for row in rows), 1)
        self.assertAlmostEqual(rows[2]["expense_ratio_to_selected"], 20335 / 20259.5)
        self.assertAlmostEqual(rows[2]["category_delta_pp"][3], 36.329179 - 33.537467)

    def test_group_coverage_conservation(self):
        profiles = self.result["profiles"]
        self.assertEqual([p["n"] for p in profiles], [644, 859, 386, 127])
        self.assertEqual(sum(p["n"] for p in profiles), 2016)
        self.assertEqual(sum(p["market_access_n"] for p in profiles), 2004)
        self.assertEqual(sum(p["wage_n"] for p in profiles), 1830)
        for p in profiles:
            self.assertEqual(sum(p["municipal_types"].values()), p["n"])
            self.assertLessEqual(p["wage_n"], p["n"])

    def test_distinct_temporal_channels_and_local_example(self):
        self.assertEqual(self.result["dynamics"]["counts"], {"raw_changed": 380, "population_relative_changed": 207, "paired_relative_changed": 218})
        example = self.result["dynamics"]["examples"][0]
        self.assertEqual(example["entity_id"], "tid_1018")
        self.assertEqual(example["group_2023"], example["paired_relative_group_2024"])
        self.assertNotEqual(example["group_2023"], example["raw_group_2024"])

    def test_rejects_duplicate_and_missing_join_rows(self):
        for invalid in (self.bridge + [self.bridge[0]], self.bridge[1:]):
            with self.assertRaises(ValueError):
                compile_cases(self.data, copy.deepcopy(invalid), self.cohort, self.paired, self.abazinsky)

    def test_rejects_expense_disagreement(self):
        invalid = copy.deepcopy(self.bridge)
        invalid[0]["expense_2023"] = "1"
        with self.assertRaises(ValueError):
            compile_cases(self.data, invalid, self.cohort, self.paired, self.abazinsky)

    def test_html_has_no_script_and_links_to_existing_sections(self):
        html = html_block(self.result)
        self.assertNotIn("<script", html)
        self.assertIn('id="practical-title"', html)
        template = (ROOT / "web/research_template.html").read_text("utf-8")
        for anchor in ("territory-title", "neighbors-title", "flows-title"):
            self.assertIn(f'href="#{anchor}"', html)
            self.assertIn(f'id="{anchor}"', template)

    def test_rejects_duplicated_or_wrong_calendar_at_same_length(self):
        for changed_period in ("2023-01-01", "2023-02-02", "2023-13-01"):
            invalid = copy.deepcopy(self.data)
            invalid["periods"][1] = changed_period
            with self.subTest(period=changed_period), self.assertRaises(ValueError):
                compile_cases(invalid, copy.deepcopy(self.bridge), self.cohort, self.paired, self.abazinsky)
        invalid = copy.deepcopy(self.data)
        invalid["periods"][0], invalid["periods"][1] = invalid["periods"][1], invalid["periods"][0]
        with self.assertRaises(ValueError):
            compile_cases(invalid, copy.deepcopy(self.bridge), self.cohort, self.paired, self.abazinsky)

    def test_rejects_annual_ratio_inconsistent_with_saved_months(self):
        invalid = copy.deepcopy(self.data)
        for entity in invalid["entities"]:
            entity["annual_ratios"][1] = 999.0
        with self.assertRaises(ValueError):
            compile_cases(invalid, copy.deepcopy(self.bridge), self.cohort, self.paired, self.abazinsky)

    def test_accepts_independent_six_decimal_rounding(self):
        invalid = copy.deepcopy(self.data)
        # The median of unrounded months is 10.00000049, serialized as 10.000000.
        # Rounded middle months serialize as 10.000000 and 10.000001.
        invalid["entities"][0]["ratios"][:12] = [[10.0] * 5 for _ in range(6)] + [[10.000001] * 5 for _ in range(6)]
        invalid["entities"][0]["annual_ratios"] = [10.0] * 5
        result = compile_cases(invalid, copy.deepcopy(self.bridge), self.cohort, self.paired, self.abazinsky)
        self.assertEqual(result["n"], 2016)

    def test_missing_selected_wage_renders_explicit_absence(self):
        cohort = [row for row in self.cohort if row["entity_id"] != "tid_616"]
        result = compile_cases(self.data, copy.deepcopy(self.bridge), cohort, self.paired, self.abazinsky)
        self.assertIsNone(result["low_marketplace_case"]["wage_2023_rubles"])
        document = markdown_report(result)
        html = html_block(result)
        self.assertIn("нет данных о зарплате организаций без малого бизнеса этого района за 2023 год", document)
        self.assertNotIn("None", document + html)
        self.assertNotIn("0,00 руб.; низкая интенсивность", document)


if __name__ == "__main__":
    unittest.main()
