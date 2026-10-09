import argparse
import base64
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts import build_research_atlas as atlas

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/panel.csv"
GRAPH = ROOT / "reports/graph_preparation.json"
PARTITIONS = ROOT / "reports/experiments/2026-09-23/partitions.csv"
METRICS = ROOT / "reports/experiments/2026-09-23/metrics.json"
TEMPLATE = ROOT / "web/research_template.html"
EDGE_RULES = ROOT / "reports/edge-rules"
VALIDATION = ROOT / "reports/experiments/2026-09-23-v2/validation"
HAS_PRIVATE_DATA = all(path.exists() for path in (PANEL, GRAPH, PARTITIONS, METRICS))


class ResearchAtlasUnitTests(unittest.TestCase):
    def test_script_json_escaping_and_nonfinite_rejection(self):
        encoded = atlas.encode_payload({"text": "</script>&\u2028"})
        self.assertNotIn("</script>", encoded)
        self.assertIn("\\u003c/script\\u003e\\u0026\\u2028", encoded)
        with self.assertRaises(ValueError):
            atlas.encode_payload({"bad": math.nan})

    def test_compressed_payload_is_deterministic_lossless_and_safe(self):
        value = {"name": "</script>\u2028", "values": [1, 2.25, None], "nested": {"id": "tid_1"}}
        encoded = atlas.package_payload(value)
        self.assertEqual(encoded, atlas.package_payload(value))
        self.assertNotIn("</script>", encoded)
        envelope = json.loads(encoded)
        self.assertEqual(json.loads(gzip.decompress(base64.b64decode(envelope["data"]))), value)
        with self.assertRaises(ValueError):
            atlas.package_payload({"invalid": math.inf})

    def test_stability_contract_accepts_scientific_ari_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stability.json"
            path.write_text('[{"candidate":"kmeans_k4","kind":"seed","parameter":2718,"ARI":-0.2,"k":4}]', encoding="utf-8")
            self.assertEqual(atlas.read_stability(path)[0]["ARI"], -0.2)
            path.write_text('[{"candidate":"kmeans_k4","kind":"seed","parameter":2718,"ARI":1.2,"k":4}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                atlas.read_stability(path)

    def test_lifecycle_tracks_follow_identity_not_reused_labels(self):
        def step(events):
            counts = {kind: sum(e["event"] == kind for e in events) for kind in ("continue", "grow", "shrink", "split", "merge", "birth", "death")}
            return {"counts": counts, "events": events}
        block = {"sizes": [{"0": 5, "1": 3, "2": 2}, {"0": 6, "2": 4}, {"0": 3, "2": 3, "5": 4}],
                 "steps": [step([{"event": "merge", "from": [0, 1], "to": [0], "size_before": 8, "size_after": 6},
                                 {"event": "grow", "from": [2], "to": [2], "size_before": 2, "size_after": 4}]),
                           step([{"event": "split", "from": [0], "to": [0, 5], "size_before": 6, "size_after": 7},
                                 {"event": "death", "from": [2], "to": [], "size_before": 4, "size_after": 0},
                                 {"event": "birth", "from": [], "to": [2], "size_before": 0, "size_after": 3}])]}
        members = [{"0": [0] * 5, "1": [1] * 3, "2": [2] * 2}, {"0": [0] * 6, "2": [2] * 4}, {"0": [0] * 3, "2": [3] * 3, "5": [0] * 4}]
        result = atlas.lifecycle_tracks(block, members)
        self.assertEqual([t["sizes"] for t in result["tracks"]], [[5, 6, 4], [3, 0, 0], [2, 4, 0], [0, 0, 3], [0, 0, 3]])
        self.assertEqual([t["profile"] for t in result["tracks"]], [0, 1, 2, 0, 3])
        self.assertEqual([(m["t"], m["event"]) for m in result["marks"]], [(1, "merge"), (2, "split"), (2, "death"), (2, "birth")])
        self.assertEqual(result["groups"], [3, 2, 3])
        block["steps"][1]["events"].pop()
        with self.assertRaisesRegex(ValueError, "cover every group"):
            atlas.lifecycle_tracks(block, members)

    def test_spectral_groups_align_colors_without_relabelling_membership(self):
        labels, groups = atlas._aligned_groups([3, 3, 0, 0, 1, 2, 2, 2], [0, 0, 1, 1, 2, 3, 3, 3])
        self.assertEqual(labels, [0, 0, 1, 1, 2, 3, 3, 3])
        self.assertEqual([(g["id"], g["n"], g["share"]) for g in groups], [(0, 2, 1.0), (1, 2, 1.0), (2, 1, 1.0), (3, 3, 1.0)])
        with self.assertRaisesRegex(ValueError, "four spectral groups"):
            atlas._aligned_groups([0, 1, 2], [0, 1, 2])

    def test_template_renders_edge_rules_icvi_and_lifecycle(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        for marker in ('id="edges-section"', 'id="edge-jaccard"', 'id="events-chart"', "data-icvi-k", "renderEdgeRules", "drawEvents",
                       "не определён: нулевая плотность у центра", "модулярность Ньюмана", "fixed_profiles_deseasonalized", 'id="icvi-rho"'):
            self.assertIn(marker, text)
        self.assertNotIn("официальное определение конкурсного MQ пока не подтверждено", text)

    def test_template_has_one_map_and_early_method_comparison(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(text.count('class="hero-map"'), 1)
        self.assertNotIn('id="territory-map"', text)
        self.assertNotIn("Сетевая типология</button>", text)
        self.assertEqual(text.count("<!-- METHOD_COMPARISON -->"), 1)
        self.assertLess(text.index("<!-- METHOD_COMPARISON -->"), text.index('id="act-core"'))
        self.assertLess(text.index('id="act-compare"'), text.index('id="act-dossier"'))
        for marker in ("Главная модель: группы по месяцам", "Под ней: каркас из трёх уровней", "Базовая линия: KMeans4"):
            self.assertIn(marker, text)
        self.assertNotIn("сам выбрал", text)
        self.assertNotIn("KMeans4 считаются по 2023 году, а 2024 год служит проверкой", text)
        self.assertNotIn("Каркас лучше KMeans", text)
        self.assertIn("первой версии правила отбора", text)
        self.assertIn("TQ.own_next_by_seed", text)

    def test_public_template_states_neighbor_recalculation_scope(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("Кластеры не переобучаются", text)
        self.assertIn("не означает структурный перелом", text)
        self.assertNotIn("по поручению", text.lower())
        self.assertNotIn("искусственн", text.lower())

    def test_external_municipal_summary_feeds_type_cards(self):
        names = ["Сбалансированный: средние МО", "Продуктовый: малые районы", "Городской сервисный",
                 "Удалённый: мало маркетплейсов"]
        contest = {"profiles": [{"id": i, "name": name} for i, name in enumerate(names)]}
        value = atlas.read_external_municipal(ROOT / "reports/external-municipal", contest)
        self.assertEqual([p["name"] for p in value["profiles"]], [name.removesuffix(" профиль") for name in names])
        self.assertEqual(value["profiles"][0]["population"]["median"], 35844.0)
        self.assertEqual(value["profiles"][2]["wage"]["median"], 98762.5)
        self.assertEqual(value["eta_squared"]["population"], {"none": 0.4531, "region": 0.4323})
        self.assertEqual(value["eta_squared"]["wage_2023"], {"none": 0.4626, "region": 0.1301})
        contest["profiles"][0]["name"] = "Другой тип"
        with self.assertRaises(ValueError):
            atlas.read_external_municipal(ROOT / "reports/external-municipal", contest)
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("D.external_municipal", text)
        self.assertIn("Автор: Егор Щеренко", text)


@unittest.skipUnless(HAS_PRIVATE_DATA, "private processed data are not present in a clean clone")
class ResearchAtlasIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = atlas.read_panel(PANEL, GRAPH)

    def test_panel_contract_preserves_real_complete_cohort(self):
        panel = self.panel
        self.assertEqual(len(panel["entities"]), 2016)
        self.assertEqual(len(panel["periods"]), 24)
        self.assertEqual(panel["periods"], sorted(panel["periods"]))
        self.assertTrue(all(len(entity["totals"]) == 24 for entity in panel["entities"]))
        self.assertTrue(all(len(entity["annual_features"]) == 5 for entity in panel["entities"]))
        self.assertTrue(all(value > 0 for entity in panel["entities"] for value in entity["totals"]))

    def test_pilot_is_exact_december_kmeans_join(self):
        ids = {entity["id"] for entity in self.panel["entities"]}
        pilot = atlas.read_pilot(PARTITIONS, METRICS, ids)
        self.assertEqual(pilot["period"], "2023-12-01")
        self.assertEqual(len(pilot["labels"]), 2016)
        self.assertEqual(sum(pilot["cluster_sizes"].values()), 2016)
        self.assertEqual(pilot["metric"]["k"], 8)
        self.assertAlmostEqual(pilot["metric"]["SW"], 0.21260261826094157)

    def test_incomplete_validation_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "frozen_prototypes.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                atlas.read_validation(path, {e["id"] for e in self.panel["entities"]}, set(self.panel["periods"]))

    def test_build_is_self_contained_and_uses_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            args = argparse.Namespace(panel=PANEL, graph_report=GRAPH, partitions=PARTITIONS, metrics=METRICS,
                                      reference_run=None, validation_run=None, stability_run=None,
                                      template=TEMPLATE, output=output, uncompressed=True)
            atlas.build(args)
            rendered = output.read_text(encoding="utf-8")
        self.assertNotIn("/*__ATLAS_DATA__*/null", rendered)
        self.assertIn('"entities":2016', rendered)
        self.assertIn('"status":"pilot"', rendered)
        self.assertNotIn(":Infinity", rendered)
        self.assertIn("https://creativecommons.org/licenses/by-sa/4.0", rendered)
        self.assertIn("https://www.openstreetmap.org/copyright", rendered)


    @unittest.skipUnless((EDGE_RULES / "edge_rules.json").exists() and VALIDATION.exists(), "edge-rule study is not published")
    def test_edge_rule_payload_joins_cohort_and_stays_compact(self):
        order = [entity["id"] for entity in self.panel["entities"]]
        validation = atlas.read_validation(VALIDATION, set(order), set(self.panel["periods"]))
        value = atlas.read_edge_rules(EDGE_RULES, order, self.panel["periods"], validation)
        self.assertEqual([r["rule"] for r in value["rules"]], list(atlas.EDGE_RULES))
        self.assertEqual(len(value["jaccard"]), 6)
        self.assertTrue(all(value["jaccard"][i][i] == 1 for i in range(6)))
        self.assertTrue(all(len(p["labels"]) == 2016 and sum(g["n"] for g in p["groups"]) == 2016 for p in value["partitions"].values()))
        self.assertTrue(all(row["MQ"] is not None for row in value["icvi"]))
        reference = next(row for row in value["icvi"] if row["partition"] == "kmeans_k4")
        self.assertAlmostEqual(reference["SW"], 0.240237, places=6)
        agreement = value["icvi_agreement"]
        self.assertEqual(agreement["keys"], list(atlas.ICVI_KEYS))
        self.assertEqual((agreement["all"]["n"], agreement["k4"]["n"]), (len(value["icvi"]), sum(r["k"] == 4 for r in value["icvi"])))
        self.assertTrue(all(len(agreement[s]["rho"]) == 7 and agreement[s]["rho"][i][i] == 1 for s in ("all", "k4") for i in range(7)))
        self.assertIn("kmeans_k4", agreement["pareto_k4_sw_mq"]["front"])
        for name in ("fixed_profiles", "fixed_profiles_deseasonalized", "temporal_leiden"):
            block = value["lifecycle"][name]
            self.assertTrue(all(sum(t["sizes"][m] for t in block["tracks"]) == 2016 for m in range(24)))
        self.assertEqual(value["lifecycle"]["fixed_profiles"]["counts"]["split"] + value["lifecycle"]["fixed_profiles"]["counts"]["merge"], 0)
        self.assertNotIn("events", json.dumps(value["lifecycle"]))
        self.assertLess(len(atlas.encode_payload(value)), 60_000)
        groups = atlas.read_temporal_groups(ROOT / "reports/temporal-groups", order, self.panel["periods"], validation)
        self.assertTrue(all(sum(g["sizes"][m] for g in groups["groups"]) == 2016 for m in range(24)))
        self.assertTrue(all(sum(n for _, _, n in step) == 2016 for step in groups["flows"]))
        self.assertEqual(sum(1 for g in groups["groups"] if g.get("name")), 7)
        self.assertEqual(len(groups["stable"]), 7)
        self.assertTrue(all(len(row) == 2016 for row in groups["monthly"]))
        self.assertTrue(all(groups["monthly"][m].count(str(j)) == groups["groups"][k]["sizes"][m]
                            for m in range(24) for j, k in enumerate(groups["stable"])))

    def test_network_core_and_temporal_quality_payloads(self):
        order = [entity["id"] for entity in self.panel["entities"]]
        core = atlas.read_network_core(ROOT / "reports/network-core", order)
        self.assertEqual([g["n"] for g in core["groups"]], [1391, 376, 249])
        self.assertEqual(len(core["labels"]), 2016)
        self.assertEqual(core["comparison"][0]["id"], "selected")
        quality = atlas.read_temporal_quality(ROOT / "reports/temporal-quality", self.panel["periods"])
        self.assertEqual([row["omega"] for row in quality["tradeoff"] if row["selected"]], [quality["selected"]])
        self.assertTrue(all(len(v) == 24 for series in quality["series"].values() for v in series.values()))
        self.assertIsNone(quality["series"]["leiden"]["MQ_next"][-1])


if __name__ == "__main__":
    unittest.main()