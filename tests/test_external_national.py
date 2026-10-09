from __future__ import annotations

import unittest
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from sbercluster.external_national import name_key, number_status, parse_passport, select_unique, total_sector
from scripts.prepare_external_national import attach_observations
from scripts.validate_external_national import blocked_predictions, holm, residualized_effect, within_region
from scripts.download_external_national import download_one
from scripts.download_external_national_archive import download
from scripts.validate_external_national_robustness import pairwise_region_loss


def source_html(*, value="12345.6", year=2023, unit="рубль", duplicate=False):
    annual = f'<tr><td class="prizn" style="PADDING-LEFT: 20pt;">январь-декабрь</td><td>{unit}</td><td>{value}</td></tr>'
    return (f'<html><div>Паспорт муниципального образования</div><div>БД ПМО Тест</div><div>Городские округа</div><div>Городской округ</div><div>Город Майкоп</div><div>за {year} год</div><table>'
            '<tr><td class="pok">Среднемесячная заработная плата работников организаций (без субъектов малого предпринимательства) с 2017 г.</td><td></td><td></td></tr>'
            '<tr><td class="prizn" style="PADDING-LEFT: 10pt;">Всего по обследуемым видам экономической деятельности</td><td></td><td></td></tr>'
            '<tr><td class="prizn" style="PADDING-LEFT: 20pt;">январь-март</td><td>рубль</td><td>9</td></tr>'
            + annual + (annual if duplicate else '') + '</table></html>').encode('utf-8')


class ParserTests(unittest.TestCase):
    def test_annual_total_not_quarter(self):
        p = parse_passport(source_html(), 2023)
        self.assertEqual(p.source_name, "Город Майкоп")
        self.assertEqual(select_unique(p.observations, "wage", total_sector), (12345.6, "observed"))

    def test_wrong_period_is_excluded(self):
        self.assertEqual(parse_passport(source_html(year=2024), 2023).status, "wrong_or_missing_year")

    def test_absent_table_is_not_zero(self):
        p = parse_passport('<html>Файл таблицы не найден</html>'.encode('utf-8'), 2023)
        self.assertEqual(p.status, "table_absent")
        self.assertEqual(p.observations, ())

    def test_unit_mismatch_is_excluded(self):
        p = parse_passport(source_html(unit="человек"), 2023)
        self.assertEqual(select_unique(p.observations, "wage", total_sector), (None, "unexpected_unit"))

    def test_conflicting_rows_are_not_averaged(self):
        p = parse_passport(source_html(duplicate=True), 2023)
        self.assertEqual(select_unique(p.observations, "wage", total_sector), (None, "duplicate_or_conflicting_source_rows"))

    def test_suppression_is_not_zero(self):
        for text in ["...", "…", "ND", "UD", "", "-"]:
            self.assertIsNone(number_status(text)[0])
        self.assertEqual(number_status("0"), (0.0, "observed"))

    def test_names_only_normalize_explicit_type_and_dates(self):
        self.assertEqual(name_key("Бежецкий муниципальный район (с 2024 г. преобразован в Бежецкий муниципальный округ)"), name_key("Бежецкий муниципальный район"))
        self.assertNotEqual(name_key("Зиминское"), name_key("Зима"))


class JoinTests(unittest.TestCase):
    def test_wrong_source_name_or_code_validity_cannot_enter_analysis(self):
        registry = pd.DataFrame([{"entity_id": "tid_1", "source_oktmo8": "79701000", "oktmo": "79-701-000-000",
                                  "municipal_district_name": "городской округ город Майкоп", "municipal_district_name_short": "Майкоп",
                                  "region_code": 1, "region_name": "Республика Адыгея"}])
        source = pd.DataFrame([{"oktmo": "79701000", "municipality": "Город Майкоп", "region_name": "Республика Адыгея",
                                "oktmo_year_from": "2010", "oktmo_year_to": "2025", "year": "2023",
                                "mun_level": "Муниципальное образование верхнего уровня", "kind": "wage",
                                "indicator_unit": "Рубль", "indicator_value": "45000", "conflicting_key": False,
                                "oktmo_history": "Без изменений"}])
        self.assertTrue(attach_observations(registry, source).join_verified.iloc[0])
        source.loc[0, "oktmo_year_to"] = "2013"
        self.assertFalse(attach_observations(registry, source).join_verified.iloc[0])
        source.loc[0, "oktmo_year_to"] = "2025"
        source.loc[0, "municipality"] = "Другой город"
        self.assertFalse(attach_observations(registry, source).join_verified.iloc[0])

    def test_exact_code_without_match_cannot_assign_name(self):
        registry = pd.DataFrame([{"entity_id": "tid_1", "source_oktmo8": "79701000", "oktmo": "79-701-000-000",
                                  "municipal_district_name": "городской округ город Майкоп", "municipal_district_name_short": "Майкоп",
                                  "region_code": 1, "region_name": "Республика Адыгея"}])
        source = pd.DataFrame([{"oktmo": "79703000", "municipality": "Город Майкоп", "region_name": "Республика Адыгея",
                                "oktmo_year_from": "2010", "oktmo_year_to": "2025", "year": "2023",
                                "mun_level": "Муниципальное образование верхнего уровня", "kind": "wage",
                                "indicator_unit": "Рубль", "indicator_value": "45000", "conflicting_key": False,
                                "oktmo_history": "Без изменений"}])
        self.assertTrue(attach_observations(registry, source).empty)


class StatisticalTests(unittest.TestCase):
    def test_held_out_targets_never_affect_held_out_predictions(self):
        rng = np.random.default_rng(7)
        region = np.repeat(np.arange(4), 8)
        x = rng.normal(size=(32, 3))
        y = x[:, 0] + rng.normal(size=32)
        before = blocked_predictions(y, {"controls": x}, region)["controls"]
        changed = y.copy()
        changed[region == 0] += 1000000
        after = blocked_predictions(changed, {"controls": x}, region)["controls"]
        np.testing.assert_array_equal(before[region == 0], after[region == 0])

    def test_region_only_constant_is_not_cluster_effect(self):
        region = np.repeat(np.arange(4), 8)
        y = region.astype(float) + .1
        labels = np.tile([0, 1], 16)
        effect = residualized_effect(y, np.ones((32, 1)), np.eye(2)[labels], region)[0]
        self.assertIsNone(effect)

    def test_effect_invariant_to_outcome_units(self):
        rng = np.random.default_rng(9)
        region = np.repeat(np.arange(4), 20)
        controls = rng.normal(size=(80, 2))
        labels = np.tile(np.arange(4), 20)
        y = labels + controls[:, 0] + rng.normal(size=80)
        a = residualized_effect(y, controls, np.eye(4)[labels], region)[0]
        b = residualized_effect(y * 1e-9, controls * 1e8, np.eye(4)[labels], region)[0]
        self.assertAlmostEqual(a, b, places=10)

    def test_exact_float_transformed_cluster_controls_have_zero_added_effect(self):
        rng = np.random.default_rng(1903)
        n = 120
        regions = np.repeat(np.arange(4), 30)
        labels = np.tile(np.arange(4), 30)
        z = np.eye(4)[labels]
        transform = np.array([[1., .2, .3], [.3, 1.1, .7], [.4, .8, 1.3]])
        controls = np.column_stack([np.arange(n) * .017, z[:, 1:] @ transform])
        y = rng.normal(size=n)
        # Before the original-scale rank gate, normalized projection roundoff
        # created partialR²=0.002779... although controls exactly span all labels.
        for scale in [1e-8, 1., 1e8]:
            effect = residualized_effect(y, controls, z * scale, regions)[0]
            self.assertEqual(effect, 0.)

    def test_exact_indicator_controls_cannot_turn_projection_noise_into_effect(self):
        rng = np.random.default_rng(2718)
        n = 200
        labels = np.arange(n) % 4
        regions = np.repeat(np.arange(10), 20)
        z = np.eye(4)[labels]
        controls = np.column_stack([z, rng.normal(size=(n, 2))])
        self.assertEqual(residualized_effect(rng.normal(size=n), controls, z, regions)[0], 0.)

    def test_partly_confounded_addition_keeps_identifiable_signal(self):
        rng = np.random.default_rng(28)
        n = 200
        labels = np.arange(n) % 4
        regions = np.repeat(np.arange(10), 20)
        z = np.eye(4)[labels]
        controls = np.column_stack([z[:, 1:3] @ np.array([[1.1, .31], [.19, 1.6]]), rng.normal(size=n)])
        y = 2 * z[:, 3] + rng.normal(scale=.2, size=n)
        effect = residualized_effect(y, controls, z, regions)[0]
        self.assertGreater(effect, .8)

    def test_common_denominator_cancels_before_aggregation(self):
        rng = np.random.default_rng(11)
        logcats = rng.normal(size=(12, 5))
        logtotal = rng.normal(size=12)
        ratios = logcats - logtotal[:, None]
        np.testing.assert_allclose(ratios - ratios.mean(axis=1)[:, None], logcats - logcats.mean(axis=1)[:, None], atol=1e-14)

    def test_holm_is_monotone_and_bounded(self):
        self.assertEqual(holm([.03, .01, .5]), [.06, .03, .5])

    def test_within_constant_cannot_create_decimal_rounding_residual(self):
        region = np.repeat(np.arange(3), [17, 19, 23])
        self.assertTrue((within_region(np.full((59, 2), .1), region) == 0).all())

    def test_pairwise_loss_uses_evaluation_error_differences(self):
        y = np.array([1., 3., 4., 7.])
        prediction = np.array([.2, 2.5, 3., 5.])
        regions = np.array([0, 0, 1, 1])
        got = pairwise_region_loss(y, {"probe": prediction}, regions)[0]
        errors = y - prediction
        expected = ((errors[0] - errors[1])**2 + (errors[2] - errors[3])**2) / 2
        self.assertAlmostEqual(got["mean_within_region_pairwise_squared_error_difference"], expected)


class AcquisitionTests(unittest.TestCase):
    def test_primary_fetch_uses_standard_library_response_status(self):
        row = {"source_oktmo8": "01512000", "entity_id": "tid_1", "oktmo": "01-512-000-000",
               "region_code": 22, "municipal_district_name": "Залесовский муниципальный округ"}
        response = MagicMock()
        response.status = 200
        response.url = "https://rosstat.gov.ru/scripts/db_inet2/passport/table.aspx?opt=15120002023"
        response.read.side_effect = [b"bounded-source", b""]
        response.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as directory, patch("scripts.download_external_national.urllib.request.urlopen", return_value=response), patch("scripts.download_external_national.time.sleep"):
            result = download_one(row, 2023, Path(directory), False)
            self.assertEqual(result["http_status"], 200)
            self.assertEqual(result["sha256"], hashlib.sha256(b"bounded-source").hexdigest())
            self.assertTrue(result["tls_certificate_verified"])

    def test_archive_rejects_changed_existing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wage-iftochno.zip"
            path.write_bytes(b"changed")
            source = {"file": path.name, "url": "https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/pinned.zip", "sha256": "0" * 64}
            with self.assertRaises(FileExistsError):
                download(source, Path(directory))
            self.assertEqual(path.read_bytes(), b"changed")


if __name__ == "__main__":
    unittest.main()
