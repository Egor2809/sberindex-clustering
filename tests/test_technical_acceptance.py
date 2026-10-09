"""Failure controls for the offline technical acceptance command."""
import contextlib
import io
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.verify_technical_core import (AcceptanceError, SetupError, PACKAGES, contained_path,
    digest, main, read_json, verify_package, verify_reference_packets, run_cli)


class TechnicalAcceptance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = "reports/fixture"
        self.directory = self.root / self.package
        self.directory.mkdir(parents=True)
        self.payload = self.directory / "result.json"
        self.payload.write_text('{"result":1}\n', encoding="utf-8")
        self.manifest = self.directory / "manifest.json"

    def save_manifest(self, document):
        self.manifest.write_text(json.dumps(document), encoding="utf-8")

    def simple_manifest(self):
        self.save_manifest({"files_sha256": {"result.json": digest(self.payload)}})

    def test_three_existing_manifest_schemas(self):
        for document in (
            {"files_sha256": {"result.json": digest(self.payload)}},
            {"payload": {"result.json": {"sha256": digest(self.payload), "bytes": self.payload.stat().st_size}}},
            {"files": [{"path": self.package + "/result.json", "sha256": digest(self.payload), "bytes": self.payload.stat().st_size}]},
        ):
            self.save_manifest(document)
            result = verify_package(self.root, self.package)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["payload_files"], 1)

    def test_changed_payload_is_rejected(self):
        self.simple_manifest()
        self.payload.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(AcceptanceError, "hash/size mismatch"):
            verify_package(self.root, self.package)

    def test_missing_payload_is_rejected(self):
        self.simple_manifest()
        self.payload.unlink()
        with self.assertRaisesRegex(AcceptanceError, "Missing payload"):
            verify_package(self.root, self.package)

    def test_unlisted_payload_is_rejected(self):
        self.simple_manifest()
        (self.directory / "extra.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaisesRegex(AcceptanceError, "Unlisted payload"):
            verify_package(self.root, self.package)

    def test_unsafe_manifest_paths_are_rejected_on_any_host(self):
        for name in ("../outside", "/outside", "C:/outside", "a\\b", "./result.json", "a//b"):
            with self.subTest(path=name), self.assertRaises(AcceptanceError):
                contained_path(self.directory, name)

    def test_duplicate_payload_and_invalid_digest_are_rejected(self):
        item = {"path": self.package + "/result.json", "sha256": digest(self.payload)}
        self.save_manifest({"files": [item, item]})
        with self.assertRaisesRegex(AcceptanceError, "Duplicate manifest path"):
            verify_package(self.root, self.package)
        self.save_manifest({"files_sha256": {"result.json": "f" * 65}})
        with self.assertRaisesRegex(AcceptanceError, "SHA256"):
            verify_package(self.root, self.package)

    def test_duplicate_json_fields_and_nonfinite_values_are_rejected(self):
        for raw in ('{"files":[],"files":[]}', '{"value":NaN}'):
            self.manifest.write_text(raw, encoding="utf-8")
            with self.assertRaises(AcceptanceError):
                read_json(self.manifest)

    def test_schema_and_empty_manifest_are_rejected(self):
        for document in ({"files_sha256": []}, {"files_sha256": {}}):
            self.save_manifest(document)
            with self.assertRaises(AcceptanceError):
                verify_package(self.root, self.package)

    def test_current_documentation_and_source_drift_are_observations(self):
        self.save_manifest({"files_sha256": {"result.json": digest(self.payload)},
                            "documentation_sha256": {"docs/updated.md": "a" * 64},
                            "current_source_sha256": {"sbercluster/updated.py": "b" * 64}})
        result = verify_package(self.root, self.package)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual([r["status"] for r in result["current_snapshot"]],
                         ["CURRENT_DOCUMENTATION_DRIFT", "CURRENT_SOURCE_DRIFT"])

    def make_core_fixture(self):
        for package in PACKAGES:
            directory = self.root / package
            directory.mkdir(parents=True)
            (directory / "result.json").write_bytes(self.payload.read_bytes())
            (directory / "manifest.json").write_text(json.dumps({"files_sha256": {"result.json": digest(self.payload)}}), encoding="utf-8")

    def test_integrity_only_does_not_run_predictions_or_claim_scientific_acceptance(self):
        self.make_core_fixture()
        output = self.root / "acceptance.json"
        with patch("scripts.verify_technical_core.verify_predictions", side_effect=AssertionError("must not run")), contextlib.redirect_stdout(io.StringIO()):
            code = main(["--root", str(self.root), "--output", str(output), "--integrity-only"])
        report = read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PASS_INTEGRITY_ONLY")
        self.assertEqual(report["predictions"]["status"], "NOT_RUN")
        self.assertEqual(report["scientific_status"], "not_assessed")

    def test_payload_change_during_prediction_is_detected(self):
        self.make_core_fixture()
        def change_payload(_root):
            (self.root / PACKAGES[0] / "result.json").write_text("changed")
            return {"status": "PASS"}
        output = self.root / "acceptance.json"
        with patch("scripts.verify_technical_core.verify_predictions", side_effect=change_payload), contextlib.redirect_stdout(io.StringIO()):
            code = main(["--root", str(self.root), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(read_json(output)["status"], "FAIL")
        self.assertIn("hash/size mismatch", read_json(output)["error"])

    def test_existing_or_inside_package_output_is_refused(self):
        for output in (self.payload, self.root / PACKAGES[0] / "new-report.json"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["--root", str(self.root), "--output", str(output), "--integrity-only"])
            self.assertEqual(caught.exception.code, 2)
        self.assertEqual(self.payload.read_text(), '{"result":1}\n')

    def packet_fixture(self, mutation=None):
        def seal(document):
            document.pop("content_sha256", None)
            document["content_sha256"] = hashlib.sha256(json.dumps(document, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            return document
        ids = ["tid_1", "tid_2"]
        metadata = {"content_sha256": "a" * 64, "entity_ids": ids, "regions": ["1", "2"]}
        packet = {"revision": "paired_reference_packet_v1", "reference_lock_sha256": metadata["content_sha256"],
                  "reference_n": 2, "observed_valid_n": 2, "coverage": 1., "missing_ids": [], "invalid_ids": [],
                  "observed_ids_sha256": hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
                  "measurement_epsilon": 0., "measurement_error_known": False, "territory_transition_status": "unresolved",
                  "correction_lower": [0.] * 5, "correction_upper": [0.] * 5,
                  "lower_unbounded": [False] * 5, "upper_unbounded": [False] * 5,
                  "representative_correction": [0.] * 5, "point_paired_median": [0.] * 5,
                  "status": "complete_reference", "regional_coverage": {region: {"reference_n": 1, "observed_valid_n": 1, "coverage": 1.} for region in ["1", "2"]}}
        directory = self.root / "packets"
        directory.mkdir(exist_ok=True)
        manifest = {}
        for month in range(1, 13):
            period = f"2024-{month:02d}-01"
            current = copy.deepcopy(packet)
            current["period"] = period
            if mutation is not None:
                mutation(current)
            path = directory / (period + ".json")
            path.write_text(json.dumps(seal(current)), encoding="utf-8")
            manifest[period] = {"path": path.name, "sha256": digest(path)}
        return directory, metadata, manifest

    def test_public_reference_context_validates_twelve_well_formed_packets(self):
        self.assertEqual(verify_reference_packets(*self.packet_fixture()), 12)

    def test_resealed_packet_schema_revision_dimensions_and_intervals_are_rejected(self):
        mutations = [lambda packet: packet.update(revision="unsupported_revision"),
                     lambda packet: packet.pop("observed_valid_n"),
                     lambda packet: packet.update(correction_lower=[0.] * 4),
                     lambda packet: packet.update(correction_lower=[1.] * 5)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaisesRegex(AcceptanceError, "packet contract differs"):
                verify_reference_packets(*self.packet_fixture(mutation))

    def child_fixture(self):
        (self.root / "acceptance_child_fixture.py").write_text(
            "from pathlib import Path\nimport sys\nmode=sys.argv[1]\n"
            "if mode=='dependency': raise ModuleNotFoundError('synthetic missing dependency')\n"
            "if mode=='import': raise ImportError('synthetic failed import')\n"
            "if mode=='unexpected': raise ValueError('Unrelated invalid input')\n"
            "if mode=='existing': Path(sys.argv[2]).open('xb')\n"
            "if mode=='write': Path(sys.argv[2]).write_bytes(b'forbidden result')\n"
            "raise ValueError('Pinned file SHA256 differs: synthetic fixture')\n", encoding="utf-8")
        return "acceptance_child_fixture"

    def test_actual_missing_dependency_is_setup_error_for_success_and_refusal_controls(self):
        module = self.child_fixture()
        for mode in ("dependency", "import"):
            for control in ({}, {"expected_refusal": ("ValueError", "Pinned file SHA256 differs:"), "output": self.root / "absent.json"}):
                with self.subTest(mode=mode, control=bool(control)), self.assertRaises(SetupError):
                    run_cli(self.root, [module, mode], **control)
        with self.assertRaises(SetupError):
            run_cli(self.root, ["unavailable_acceptance_module"])

    def test_actual_refusals_require_declared_reason_and_output_policy(self):
        module = self.child_fixture()
        absent = self.root / "absent.json"
        result = run_cli(self.root, [module, "expected"], expected_refusal=("ValueError", "Pinned file SHA256 differs:"), output=absent)
        self.assertEqual(result["outcome"], "EXPECTED_REFUSAL")
        self.assertFalse(absent.exists())
        result = run_cli(self.root, [module, "existing", str(self.payload)], expected_refusal=("FileExistsError", "File exists"),
                         output=self.payload, preserve_sha256=digest(self.payload))
        self.assertEqual(result["output_policy"], "preserved_exact_bytes")
        with self.assertRaisesRegex(AcceptanceError, "declared reason"):
            run_cli(self.root, [module, "unexpected"], expected_refusal=("ValueError", "Pinned file SHA256 differs:"), output=absent)
        with self.assertRaisesRegex(AcceptanceError, "published an output"):
            run_cli(self.root, [module, "write", str(absent)], expected_refusal=("ValueError", "Pinned file SHA256 differs:"), output=absent)

    def test_actual_child_setup_failure_returns_setup_error_exit_two(self):
        self.make_core_fixture()
        module = self.child_fixture()
        def child_failure(_root):
            return run_cli(self.root, [module, "dependency"])
        output = self.root / "setup-error.json"
        with patch("scripts.verify_technical_core.verify_predictions", side_effect=child_failure), contextlib.redirect_stdout(io.StringIO()):
            code = main(["--root", str(self.root), "--output", str(output)])
        report = read_json(output)
        self.assertEqual((code, report["exit_code"], report["status"]), (2, 2, "SETUP_ERROR"))
        self.assertIn("ModuleNotFoundError", report["failed_cli"]["stderr"])

    def test_actual_unexpected_child_failure_returns_contract_failure_exit_one(self):
        self.make_core_fixture()
        module = self.child_fixture()
        def child_failure(_root):
            return run_cli(self.root, [module, "unexpected"], expected_refusal=("ValueError", "Pinned file SHA256 differs:"), output=self.root / "absent.json")
        output = self.root / "contract-error.json"
        with patch("scripts.verify_technical_core.verify_predictions", side_effect=child_failure), contextlib.redirect_stdout(io.StringIO()):
            code = main(["--root", str(self.root), "--output", str(output)])
        report = read_json(output)
        self.assertEqual((code, report["exit_code"], report["status"]), (1, 1, "FAIL"))
        self.assertIn("Unrelated invalid input", report["failed_cli"]["stderr"])


if __name__ == "__main__":
    unittest.main()
