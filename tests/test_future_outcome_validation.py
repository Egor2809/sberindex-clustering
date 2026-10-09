"""Synthetic-only oracles; never open a live municipal outcome source."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from sbercluster import future_outcome_validation as future
from scripts import validate_future_outcome as cli


def membership(entity, code, admitted=True, flag=False, name=True, candidates=None):
    n = (1 if admitted or not name else 0) if candidates is None else candidates
    return {"entity_id": entity, "source_oktmo8": code, "source_candidate_count": n,
            "exact_year_validity": n == 1, "exact_name": name and n == 1,
            "exact_region_name": n == 1, "upper_level": n == 1,
            "metadata_identity_pass": admitted, "territorial_change_flag": flag}


def source_row(code, value, **updates):
    return {**future.TARGET, "oktmo": code, "municipality": "Synthetic municipality",
            "oktmo_year_from": "2023", "oktmo_year_to": "2025",
            "indicator_value": value, **updates}


def metadata(rows):
    return {future.source_key(row): {k: v for k, v in row.items() if k != "indicator_value"}
            for row in rows}


def source_bytes(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter=";")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def compress(raw):
    encoder = zlib.compressobj(wbits=-15)
    return encoder.compress(raw) + encoder.flush()


class ForbiddenValue:
    def __str__(self):
        raise AssertionError("An excluded synthetic value was inspected")


class FutureOutcomeValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def prediction_fixture(self, n=5, regions=None):
        return future.FrozenPredictions(tuple(f"synthetic_{i}" for i in range(n)),
            np.array(regions if regions is not None else np.arange(n), dtype=int),
            np.zeros((n, len(future.PREDICTORS))))

    def test_unequal_region_size_primary_and_bootstrap_match_independent_oracle(self):
        values = np.zeros((4, 12))
        h75 = future.PREDICTORS.index(future.PRIMARY[0])
        continuous = future.PREDICTORS.index(future.PRIMARY[1])
        values[:, h75] = [0, 0, 0, 2]
        values[:, continuous] = [1, 1, 1, 0]
        result = future.paired_metrics(np.zeros(4), values, np.array([1, 1, 1, 2]))
        self.assertEqual(result["primary"]["difference"], 1.5)
        rows = {r["predictor"]: r for r in result["metrics"]}
        self.assertEqual(rows[future.PRIMARY[0]]["equal_region_mse"], 2)
        self.assertEqual(rows[future.PRIMARY[1]]["equal_region_mse"], .5)
        self.assertEqual(rows[future.PRIMARY[0]]["municipality_rmse"], 1)
        oracle_draws = np.random.default_rng(20261003).integers(2, size=(1999, 2))
        oracle = np.array([-1., 4.])[oracle_draws].mean(axis=1)
        np.testing.assert_array_equal(result["primary"]["conditional_paired_region_bootstrap_95"],
                                      np.quantile(oracle, [.025, .975]))
        reverse = rows[future.PRIMARY[1]]["comparisons"][future.PRIMARY[0]]
        np.testing.assert_array_equal(reverse["conditional_paired_region_bootstrap_95"],
            -np.array(result["primary"]["conditional_paired_region_bootstrap_95"])[::-1])

    def test_centered_variance_ignores_region_bias_and_singletons(self):
        y = np.array([1., 3., 5., 8., 11.])
        regions = np.array([1, 1, 2, 2, 3])
        values = np.zeros((5, 12))
        values[:, 1] = [100, 100, -90, -90, 900]
        result = future.paired_metrics(y, values, regions)
        a, b = result["metrics"][:2]
        self.assertEqual(a["regions_with_pairs"], 2)
        self.assertEqual(a["equal_region_centered_error_sample_variance"], b["equal_region_centered_error_sample_variance"])
        self.assertEqual(a["comparisons"][b["predictor"]]["conditional_centered_region_bootstrap_95"], [0., 0.])

    def test_all_observed_regions_singleton_has_null_sample_variance_and_intervals(self):
        values = np.zeros((3, 12))
        h75 = future.PREDICTORS.index(future.PRIMARY[0])
        continuous = future.PREDICTORS.index(future.PRIMARY[1])
        values[:, h75] = [1, 2, 3]
        values[:, continuous] = [0, 1, 1]
        result = future.paired_metrics(np.zeros(3), values, np.array([1, 2, 3]))
        self.assertEqual(result["primary"]["difference"], 4.)
        oracle_draws = np.random.default_rng(20261003).integers(3, size=(1999, 3))
        oracle = np.array([1., 3., 8.])[oracle_draws].mean(axis=1)
        np.testing.assert_array_equal(result["primary"]["conditional_paired_region_bootstrap_95"],
                                      np.quantile(oracle, [.025, .975]))
        self.assertEqual(result["centered_error_scope"]["eligible_regions"], 0)
        for row in result["metrics"]:
            self.assertEqual(row["regions_with_pairs"], 0)
            self.assertIsNone(row["equal_region_centered_error_sample_variance"])
            for comparison in row["comparisons"].values():
                self.assertIsNone(comparison["equal_region_centered_variance_difference"])
                self.assertIsNone(comparison["conditional_centered_region_bootstrap_95"])
        json.dumps(result, allow_nan=False)

    def test_shared_observed_cohort_audits_missing_regions_and_retains_history_flags(self):
        predictions = self.prediction_fixture(regions=[1, 1, 2, 3, 3])
        members = [membership("synthetic_0", "00000001", flag=True),
                   membership("synthetic_1", "00000002"),
                   membership("synthetic_2", "00000003"),
                   membership("synthetic_3", "00000004", admitted=False),
                   membership("synthetic_4", "00000005", admitted=False, name=False)]
        rows = [source_row("00000001", "100"), source_row("00000002", "0"),
                source_row("00000003", "..."), source_row("00000005", ForbiddenValue())]
        quarter = source_row("00000001", ForbiddenValue(), indicator_period="Январь-март")
        result = future.evaluate_records(predictions, tuple(members), metadata(rows), [quarter, *rows])
        self.assertEqual(len(result["audit"]), 5)
        self.assertEqual([r["outcome_status"] for r in result["audit"]],
            ["observed", "zero", "missing_or_suppressed", "identity_unmatched", "identity_name_failed"])
        self.assertEqual(result["coverage"]["entirely_unobserved_regions"], [2, 3])
        self.assertEqual(result["coverage"]["observed_history_flags"], 1)
        self.assertEqual(result["coverage"]["metadata_history_flags"], 1)
        self.assertEqual({(r["n"], r["regions"]) for r in result["metrics"]}, {(1, 1)})
        self.assertTrue(all(r["equal_region_centered_error_sample_variance"] is None for r in result["metrics"]))
        self.assertFalse(result["claim_limits"]["boundary_continuity_certified"])
        json.dumps(result, allow_nan=False)

    def test_wage_states_are_not_imputed_or_changed_into_log_zero(self):
        cases = [("", "missing_or_suppressed"), ("Х", "missing_or_suppressed"),
                 ("0", "zero"), ("-3", "negative"), ("NaN", "nonfinite"),
                 ("inf", "nonfinite"), ("not numeric", "invalid_numeric"),
                 ("CD", "source_marker_CD"), ("SD", "source_marker_SD"),
                 ("NR", "source_marker_NR"), ("—", "not_applicable_or_absent")]
        for token, expected in cases:
            with self.subTest(token=token):
                self.assertEqual(future.parse_wage(token), (expected, None))
        self.assertAlmostEqual(future.parse_wage("1\u00a0234,5")[1], np.log(1234.5))

    def test_all_missing_target_is_not_estimable_but_full_audit_survives(self):
        predictions = self.prediction_fixture(n=2, regions=[1, 2])
        members = tuple(membership(f"synthetic_{i}", f"{i+1:08d}") for i in range(2))
        rows = [source_row("00000001", ""), source_row("00000002", "-1")]
        result = future.evaluate_records(predictions, members, metadata(rows), rows)
        self.assertEqual(result["status"], "not_estimable")
        self.assertEqual(result["coverage"]["fixed_entities"], 2)
        self.assertEqual(result["coverage"]["observed_entities"], 0)
        self.assertIsNone(result["primary"]["difference"])
        self.assertEqual(len(result["metrics"]), 12)
        json.dumps(result, allow_nan=False)

    def test_nonfinite_or_missing_model_predictions_reject_before_source(self):
        for bad in [np.zeros((2, 11)), np.full((2, 12), np.nan)]:
            with self.assertRaises(ValueError):
                future.FrozenPredictions(("a", "b"), np.array([1, 2]), bad)
        with self.assertRaises(ValueError):
            future.FrozenPredictions(("a", "a"), np.array([1, 2]), np.zeros((2, 12)))
        with self.assertRaises(ValueError):
            future.FrozenPredictions(("a", "b"), np.array([1., 2.]), np.zeros((2, 12)))

    def test_membership_does_not_allow_silent_shrink_or_duplicate_identity(self):
        predictions = self.prediction_fixture(n=2)
        rows = [membership("synthetic_0", "00000001"), membership("synthetic_1", "00000002")]
        for wrong in [rows[:1], [rows[0], rows[0]], [rows[0], {**rows[1], "source_oktmo8": "00000001"}],
                      [rows[0], {**rows[1], "exact_name": False}]]:
            with self.subTest(wrong=wrong):
                with self.assertRaises(ValueError):
                    future.align_membership(wrong, predictions)
        with self.assertRaises(ValueError):
            future.align_membership(rows, predictions, expected_flags=1)
        aligned = future.align_membership(rows[::-1], predictions, expected_admitted=2, expected_flags=0)
        self.assertEqual([r["entity_id"] for r in aligned], list(predictions.ids))

    def test_duplicate_source_keys_reject_even_equal_values(self):
        row = source_row("00000001", "100")
        members = (membership("a", "00000001"),)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            future.collect_outcomes([row, row.copy()], members, metadata([row]))

    def test_endpoint_unit_indicator_year_and_absence_are_strict(self):
        row = source_row("00000001", "100")
        members = (membership("a", "00000001"),)
        for changes in [{"indicator_unit": "Тысяча рублей"}, {"year": "2024"},
                        {"indicator_code": "OTHER"}, {"indicator_period": "Январь-июнь"},
                        {"okved2": "Обрабатывающие производства"}]:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    future.collect_outcomes([{**row, **changes}], members, metadata([row]))

    def test_changed_identity_projection_or_missing_source_key_rejects(self):
        rows = [source_row("00000001", "100"), source_row("00000002", "200")]
        members = (membership("a", "00000001"), membership("b", "00000002"))
        with self.assertRaisesRegex(ValueError, "metadata"):
            future.collect_outcomes([{**rows[0], "municipality": "Changed"}, rows[1]], members, metadata(rows))
        with self.assertRaisesRegex(ValueError, "silent cohort shrink"):
            future.collect_outcomes(rows[:1], members, metadata(rows))

    def write_metadata(self, rows, name="metadata.csv"):
        path = self.directory / name
        projected = list(metadata(rows).values())
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(projected[0]))
            writer.writeheader()
            writer.writerows(projected)
        return path

    def test_metadata_file_rejects_values_and_duplicate_keys(self):
        row = source_row("00000001", "100")
        path = self.write_metadata([row])
        loaded = future.load_source_metadata(path, 1)
        self.assertEqual(loaded, metadata([row]))
        with path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=list(next(iter(loaded.values())))).writerow(next(iter(loaded.values())))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            future.load_source_metadata(path, 2)
        path.write_text("indicator_value\n100\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exclude"):
            future.load_source_metadata(path, 1)

    def test_prediction_loader_uses_twelve_columns_and_ignores_old_observed_cells(self):
        path = self.directory / "predictions.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["entity_id", "held_out_region", "observed2024", *future.PREDICTORS])
            writer.writeheader()
            writer.writerows([{"entity_id": "a", "held_out_region": "1", "observed2024": "unused nonnumeric",
                               **{name: 2 for name in future.PREDICTORS}},
                              {"entity_id": "b", "held_out_region": "2", "observed2024": "unused",
                               **{name: 3 for name in future.PREDICTORS}}])
        loaded = future.load_predictions(path, 2, 2)
        np.testing.assert_array_equal(loaded.values[:, 0], [2, 3])
        with self.assertRaises(ValueError):
            future.load_predictions(path, 3, 2)
        self.assertFalse(loaded.values.flags.writeable)

    def test_hash_pin_rejects_changed_bytes_and_unknown_contract(self):
        path = self.directory / "pin.txt"
        path.write_bytes(b"immutable synthetic evidence")
        expected = future.sha256_file(path)
        future.verify_file(path, expected)
        path.write_bytes(b"changed synthetic evidence")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            future.verify_file(path, expected)
        with self.assertRaises(ValueError):
            future.load_contract(path, "0" * 64)

    def test_streamed_deflate_handles_utf8_quoted_multiline_and_exact_crc(self):
        rows = [source_row("00000001", "100", municipality='Синтетическое; имя\n"тест"')]
        raw = source_bytes(rows)
        path = self.directory / "synthetic.deflate"
        path.write_bytes(compress(raw))
        actual = list(future.iter_source_records(path, len(raw), zlib.crc32(raw)))
        self.assertEqual(actual, rows)

    def test_deflate_rejects_broken_value_quotes_despite_valid_length_and_crc(self):
        fields = [*future.TARGET, "oktmo", "indicator_value"]
        header = ";".join(fields) + "\n"
        prefix = ";".join([*future.TARGET.values(), "00000001"]) + ";"
        path = self.directory / "synthetic-broken-quotes.deflate"
        for token in ['"100"0', '"100']:
            with self.subTest(token=token):
                raw = (header + prefix + token + "\n").encode("utf-8")
                path.write_bytes(compress(raw))
                with self.assertRaises(csv.Error):
                    list(future.iter_source_records(path, len(raw), zlib.crc32(raw)))

    def test_prediction_and_metadata_readers_reject_broken_quotes(self):
        prediction_fields = ["entity_id", "held_out_region", *future.PREDICTORS]
        prediction_prefix = ",".join(["synthetic_0", "1", *(["0"] * 11)]) + ","
        prediction_path = self.directory / "synthetic-broken-predictions.csv"
        metadata_fields = [*future.TARGET, "oktmo"]
        metadata_prefix = ",".join(future.TARGET.values()) + ","
        metadata_path = self.directory / "synthetic-broken-metadata.csv"
        for token in ['"100"0', '"100']:
            with self.subTest(token=token):
                prediction_path.write_text(
                    ",".join(prediction_fields) + "\n" + prediction_prefix + token + "\n",
                    encoding="utf-8")
                metadata_path.write_text(
                    ",".join(metadata_fields) + "\n" + metadata_prefix + token + "\n",
                    encoding="utf-8")
                with self.assertRaises(csv.Error):
                    future.load_predictions(prediction_path, 1, 1)
                with self.assertRaises(csv.Error):
                    future.load_source_metadata(metadata_path, 1)

    def test_metadata_rejects_duplicate_headers_before_key_overwrite(self):
        fields = [*future.TARGET, "oktmo", "oktmo"]
        row = [*future.TARGET.values(), "00000001", "00000001"]
        path = self.directory / "synthetic-duplicate-header.csv"
        path.write_text(",".join(fields) + "\n" + ",".join(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unique source metadata headers"):
            future.load_source_metadata(path, 1)

    def test_metadata_rejects_missing_or_extra_unused_fields(self):
        fields = [*future.TARGET, "oktmo", "comment"]
        selected = [*future.TARGET.values(), "00000001"]
        path = self.directory / "synthetic-metadata-width.csv"
        for row in [selected, [*selected, "synthetic comment", "extra unused field"]]:
            with self.subTest(width=len(row)):
                path.write_text(",".join(fields) + "\n" + ",".join(row) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Malformed source metadata CSV row"):
                    future.load_source_metadata(path, 1)

    def test_predictions_reject_missing_or_extra_unused_historical_fields(self):
        fields = ["entity_id", "held_out_region", *future.PREDICTORS, "observed2024"]
        selected = ["synthetic_0", "1", *(["0"] * 12)]
        path = self.directory / "synthetic-prediction-width.csv"
        for row in [selected, [*selected, "unused historical token", "extra unused field"]]:
            with self.subTest(width=len(row)):
                path.write_text(",".join(fields) + "\n" + ",".join(row) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Malformed prediction CSV row"):
                    future.load_predictions(path, 1, 1)

    def test_deflate_rejects_crc_length_truncation_and_trailing_bytes(self):
        raw = source_bytes([source_row("00000001", "100")])
        packed = compress(raw)
        path = self.directory / "synthetic.deflate"
        for data, size, crc in [(packed, len(raw), 0), (packed, len(raw) + 1, zlib.crc32(raw)),
                                (packed, len(raw) - 1, zlib.crc32(raw)),
                                (packed[:-1], len(raw), zlib.crc32(raw)),
                                (packed + b"trailing", len(raw), zlib.crc32(raw))]:
            with self.subTest(size=size, bytes=len(data), crc=crc):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    list(future.iter_source_records(path, size, crc))

    def test_synthetic_file_pipeline_pairing_and_output_manifest(self):
        rows = [source_row("00000001", "100"), source_row("00000002", "...")]
        raw = source_bytes(rows)
        path = self.directory / "synthetic.deflate"
        path.write_bytes(compress(raw))
        predictions = self.prediction_fixture(n=2, regions=[1, 2])
        members = tuple(membership(f"synthetic_{i}", f"{i+1:08d}", flag=i == 0) for i in range(2))
        result = future.evaluate_records(predictions, members, metadata(rows),
            future.iter_source_records(path, len(raw), zlib.crc32(raw)))
        output = self.directory / "result"
        cli.write_artifacts(result, output)
        manifest = json.loads((output / "manifest.json").read_text())
        for name, pin in manifest["files"].items():
            future.verify_file(output / name, pin)
        with (output / "audit.csv").open(encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)
        with (output / "metrics.csv").open(encoding="utf-8") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)
        with self.assertRaises(ValueError):
            cli.write_artifacts(result, output)

    def test_cli_check_never_calls_value_reader_or_evaluator(self):
        predictions = self.prediction_fixture(n=1)
        members = (membership("synthetic_0", "00000001", flag=True),)
        verified = future.VerifiedInputs(Path("synthetic-contract"), {}, {}, predictions, members, {})
        args = ["--contract", "synthetic", "--predictions", "synthetic", "--membership", "synthetic",
                "--source-metadata", "synthetic", "--source-member", "synthetic"]
        with patch.object(future, "verify_inputs", return_value=verified), \
             patch.object(future, "evaluate_verified", side_effect=AssertionError("values read")), \
             patch.object(future, "iter_source_records", side_effect=AssertionError("values read")), \
             patch("sys.stdout", new_callable=io.StringIO) as output:
            cli.main(args)
        record = json.loads(output.getvalue())
        self.assertFalse(record["outcome_values_parsed"])
        self.assertEqual(record["metadata_admitted"], 1)

    def test_bootstrap_is_deterministic_and_all_intervals_are_paired(self):
        values = np.tile(np.arange(12, dtype=float), (6, 1))
        y = np.arange(6, dtype=float)
        regions = np.array([1, 1, 2, 2, 3, 3])
        first = future.paired_metrics(y, values, regions)
        second = future.paired_metrics(y, values, regions)
        self.assertEqual(first, second)
        for row in first["metrics"]:
            self.assertEqual(row["comparisons"][row["predictor"]]["conditional_paired_region_bootstrap_95"], [0., 0.])

    def scientific_bundle(self):
        """All bytes, IDs and wages here are newly invented synthetic fixtures."""
        rows = [source_row("00000001", "100"), source_row("00000002", "200")]
        paths = {role: self.directory / name for role, name in
                 {"predictions": "synthetic-predictions.csv", "membership": "synthetic-membership.json",
                  "source_metadata": "synthetic-metadata.csv", "source_member_deflate": "synthetic.deflate"}.items()}
        with paths["predictions"].open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["entity_id", "held_out_region", *future.PREDICTORS])
            writer.writeheader()
            writer.writerows([{"entity_id": f"synthetic_{i}", "held_out_region": i+1,
                               **{name: 0 for name in future.PREDICTORS}} for i in range(2)])
        members = [membership(f"synthetic_{i}", f"{i+1:08d}", flag=i == 0) for i in range(2)]
        paths["membership"].write_text(json.dumps(members), encoding="utf-8")
        paths["source_metadata"] = self.write_metadata(rows, name="synthetic-metadata.csv")
        raw = source_bytes(rows)
        paths["source_member_deflate"].write_bytes(compress(raw))
        contract = {"provenance": {"fixed_protocol_sha256": future.PROTOCOL_SHA256,
                                   "metadata_admission_sha256": future.ADMISSION_SHA256},
                    "target": {"indicator": "Y48423007", "year": "2025", "period": "Январь-декабрь",
                               "unit": "Рубль", "sector": future.TARGET["okved2"]},
                    "predictors": list(future.PREDICTORS),
                    "primary": {"minuend": future.PRIMARY[0], "subtrahend": future.PRIMARY[1]},
                    "bootstrap": {"replicates": 1999, "seed": 20261003},
                    "cohort": {"fixed_entities": 2, "fixed_regions": 2, "metadata_admitted": 2,
                               "metadata_history_flags": 1, "source_annual_total_rows": 2},
                    "pins": {role: future.sha256_file(path) for role, path in paths.items()},
                    "source_member": {"compressed_bytes": paths["source_member_deflate"].stat().st_size,
                                      "uncompressed_bytes": len(raw), "crc32": zlib.crc32(raw)}}
        config = self.directory / "synthetic-contract.json"
        config.write_text(json.dumps(contract, ensure_ascii=False), encoding="utf-8")
        return config, paths, contract

    def test_scientific_bundle_checks_pins_before_values_and_reloads_frozen_membership(self):
        config, paths, _ = self.scientific_bundle()
        synthetic_pin = future.sha256_file(config)
        with patch.object(future, "CONTRACT_SHA256", synthetic_pin):
            with patch.object(future, "iter_source_records", side_effect=AssertionError("premature values")):
                verified = future.verify_inputs(config, paths, synthetic_pin)
            verified.membership[0]["territorial_change_flag"] = False
            verified.predictions.values.setflags(write=True)
            verified.predictions.values[:, 0] = 9000
            result = future.evaluate_verified(verified)
        self.assertEqual(result["coverage"]["metadata_history_flags"], 1)
        self.assertEqual(result["coverage"]["observed_history_flags"], 1)
        self.assertEqual(result["metrics"][0]["equal_region_mse"], result["metrics"][1]["equal_region_mse"])
        self.assertEqual(result["coverage"]["observed_entities"], 2)

    def test_byte_drift_after_check_stops_before_value_reader(self):
        config, paths, _ = self.scientific_bundle()
        synthetic_pin = future.sha256_file(config)
        with patch.object(future, "CONTRACT_SHA256", synthetic_pin):
            verified = future.verify_inputs(config, paths, synthetic_pin)
            with paths["predictions"].open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with patch.object(future, "iter_source_records", side_effect=AssertionError("premature values")) as reader:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    future.evaluate_verified(verified)
                reader.assert_not_called()

    def test_contract_semantics_reject_target_predictor_contrast_and_bootstrap_changes(self):
        config, _, contract = self.scientific_bundle()
        for section, field, wrong in [("target", "period", "Январь-июнь"),
                                      ("primary", "minuend", "linear_huber75"),
                                      ("bootstrap", "replicates", 2000),
                                      ("provenance", "fixed_protocol_sha256", "0" * 64)]:
            changed = copy.deepcopy(contract)
            changed[section][field] = wrong
            config.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
            synthetic_pin = future.sha256_file(config)
            with patch.object(future, "CONTRACT_SHA256", synthetic_pin):
                with self.assertRaises(ValueError):
                    future.load_contract(config, synthetic_pin)
        changed = copy.deepcopy(contract)
        changed["predictors"] = list(reversed(future.PREDICTORS))
        config.write_text(json.dumps(changed), encoding="utf-8")
        synthetic_pin = future.sha256_file(config)
        with patch.object(future, "CONTRACT_SHA256", synthetic_pin):
            with self.assertRaises(ValueError):
                future.load_contract(config, synthetic_pin)


if __name__ == "__main__":
    unittest.main()
