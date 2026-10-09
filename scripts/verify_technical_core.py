"""Offline integrity and prediction acceptance of the shipped technical core.

No source download, private panel, fitting, or scientific re-evaluation is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "reports/model-frontier-2026-10-03",
    "reports/temporal-core-2026-10-03",
    "reports/economic-generalization-2026-10-03",
    "reports/core-round3-review-2026-10-03",
)


class AcceptanceError(ValueError):
    """A supplied artifact or executed prediction violates its contract."""


class SetupError(RuntimeError):
    """The child command could not execute the requested check."""


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AcceptanceError("Duplicate JSON field: " + key)
            result[key] = value
        return result
    def reject(value):
        raise AcceptanceError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique, parse_constant=reject)


def contained_path(base, relative):
    """Require a canonical relative path, including on a different host OS."""
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise AcceptanceError("Unsafe manifest path: " + repr(relative))
    base = Path(base).resolve()
    path = (base / relative).resolve()
    if base not in path.parents:
        raise AcceptanceError("Manifest path escapes its root: " + relative)
    return path


def _expected_hash(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
        raise AcceptanceError("Expected a lowercase SHA256 digest")
    return value


def verify_package(root, package):
    """Check every immutable payload; report current documentation separately."""
    root = Path(root).resolve()
    directory = contained_path(root, package)
    manifest_path = directory / "manifest.json"
    manifest_hash = digest(manifest_path)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise AcceptanceError("Expected an object manifest: " + package)
    if "files_sha256" in manifest and isinstance(manifest["files_sha256"], dict):
        entries = [(name, value, None) for name, value in manifest["files_sha256"].items()]
        repository_paths = False
    elif "payload" in manifest and isinstance(manifest["payload"], dict):
        entries = [(name, item["sha256"], item.get("bytes")) for name, item in manifest["payload"].items()]
        repository_paths = False
    elif "files" in manifest and isinstance(manifest["files"], list):
        entries = [(item["path"], item["sha256"], item.get("bytes")) for item in manifest["files"]]
        repository_paths = True
    else:
        raise AcceptanceError("Unsupported manifest schema: " + package)
    if not entries:
        raise AcceptanceError("Empty payload manifest: " + package)
    seen, payload, drift, total_bytes = set(), set(), [], 0
    for name, expected, size in entries:
        expected = _expected_hash(expected)
        path = contained_path(root if repository_paths else directory, name)
        if path in seen:
            raise AcceptanceError("Duplicate manifest path: " + name)
        seen.add(path)
        if size is not None and (type(size) is not int or size < 0):
            raise AcceptanceError("Invalid byte count: " + name)
        if directory not in path.parents:
            if not name.startswith("docs/"):
                raise AcceptanceError("Payload is outside declared package: " + name)
            actual = digest(path) if path.is_file() else None
            drift.append({"path": name, "expected_sha256": expected, "actual_sha256": actual,
                          "status": "MATCH" if actual == expected else "CURRENT_DOCUMENTATION_DRIFT"})
            continue
        if path == manifest_path:
            raise AcceptanceError("A manifest cannot hash itself")
        if not path.is_file():
            raise AcceptanceError("Missing payload: " + str(path.relative_to(root)))
        actual = digest(path)
        if actual != expected or (size is not None and path.stat().st_size != size):
            raise AcceptanceError("Payload hash/size mismatch: " + str(path.relative_to(root)))
        payload.add(path)
        total_bytes += path.stat().st_size
    actual_payload = {p.resolve() for p in directory.rglob("*")
                      if p.is_file() and p != manifest_path and "__pycache__" not in p.parts}
    if payload != actual_payload:
        missing = sorted(str(p.relative_to(root)) for p in actual_payload - payload)
        raise AcceptanceError("Unlisted payload files: " + ", ".join(missing[:8]))
    for group, label in (("documentation_sha256", "CURRENT_DOCUMENTATION_DRIFT"),
                         ("current_source_sha256", "CURRENT_SOURCE_DRIFT")):
        for name, expected in manifest.get(group, {}).items():
            path = contained_path(root, name)
            actual = digest(path) if path.is_file() else None
            drift.append({"path": name, "expected_sha256": _expected_hash(expected),
                          "actual_sha256": actual, "status": "MATCH" if actual == expected else label})
    if digest(manifest_path) != manifest_hash:
        raise AcceptanceError("Manifest changed during verification: " + package)
    return {"package": package, "manifest_sha256": manifest_hash, "status": "PASS",
            "payload_files": len(payload), "payload_bytes": total_bytes, "current_snapshot": drift}


def verify_integrity(root, packages=PACKAGES):
    return [verify_package(root, package) for package in packages]


def verify_reference_packets(directory, metadata, packet_manifest):
    """Use the prediction validator with the publicly supplied ID/region context.

    Public metadata omits baseline_features/frozen_artifact. Therefore this
    verifies packet structure, seals, membership/coverage and correction bounds,
    but cannot recompute paired corrections from the private source rows.
    """
    from sbercluster.dual_profiles import validate_reference_packet
    ids, regions = metadata["entity_ids"], metadata["regions"]
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or any(not isinstance(value, str) or not value for value in ids)
            or not isinstance(regions, list) or len(regions) != len(ids)):
        raise AcceptanceError("Malformed public reference ID/region context")
    _expected_hash(metadata["content_sha256"])
    if set(packet_manifest) != {f"2024-{month:02d}-01" for month in range(1, 13)}:
        raise AcceptanceError("Expected all twelve shipped2024 reference packets")
    for period, item in packet_manifest.items():
        path = contained_path(directory, item["path"])
        if digest(path) != _expected_hash(item["sha256"]):
            raise AcceptanceError("Shipped packet fingerprint differs: " + period)
        try:
            validate_reference_packet(read_json(path), metadata, period)
        except (ValueError, KeyError, TypeError) as error:
            raise AcceptanceError("Shipped packet contract differs: " + period + ": " + str(error)) from error
    return len(packet_manifest)


def run_cli(root, arguments, *, expected_refusal=None, output=None, preserve_sha256=None):
    """Require success or a declared exception/reason and output policy.

    Missing dependencies/modules, launcher failures and timeouts are setup
    errors even when the caller is testing a deliberate refusal.
    """
    if expected_refusal is not None and output is None:
        raise SetupError("A refusal control requires an output preservation/absence policy")
    output = Path(output) if output is not None else None
    if expected_refusal is not None:
        if preserve_sha256 is None and output.exists():
            raise SetupError("Refusal control requires a previously absent output")
        if preserve_sha256 is not None and (not output.is_file() or digest(output) != preserve_sha256):
            raise SetupError("Refusal control existing-output snapshot differs")
    env = dict(os.environ)
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS")})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    options = {"cwd": str(root), "env": env, "capture_output": True, "text": True, "timeout": 45}
    if os.name == "nt":
        options["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    try:
        completed = subprocess.run([sys.executable, "-m", *arguments], **options)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SetupError("Prediction CLI launcher/timeout: " + str(error)) from error
    record = {"arguments": arguments, "exit_code": completed.returncode,
              "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
    def fail(exception, message):
        error = exception(message)
        error.cli_result = record
        raise error
    if completed.returncode != 0 and (re.search(r"(?:ModuleNotFoundError|ImportError):", completed.stderr)
            or "No module named " in completed.stderr
            or (completed.returncode == 2 and "usage:" in completed.stderr)):
        fail(SetupError, "Prediction CLI dependency/invocation failure")
    if expected_refusal is None:
        if completed.returncode != 0:
            fail(AcceptanceError, "Prediction CLI unexpected failure")
        record["outcome"] = "SUCCESS"
        return record
    exception_name, reason = expected_refusal
    final_line = completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else ""
    if completed.returncode != 1 or not final_line.startswith(exception_name + ": ") or reason not in final_line:
        fail(AcceptanceError, "Prediction CLI did not refuse for the declared reason")
    if preserve_sha256 is None:
        if output.exists():
            fail(AcceptanceError, "Refused prediction published an output")
    elif not output.is_file() or digest(output) != preserve_sha256:
        fail(AcceptanceError, "Refused prediction changed its existing output")
    record["outcome"] = "EXPECTED_REFUSAL"
    record["expected_exception"] = exception_name
    record["expected_reason"] = reason
    record["output_policy"] = "absent" if preserve_sha256 is None else "preserved_exact_bytes"
    return record


def verify_predictions(root):
    """Replay shipped profiles and exercise actual CLIs on synthetic raw rows."""
    root = Path(root).resolve()
    code_paths = sorted((root / "sbercluster").glob("*.py")) + [root / "scripts" / (name + ".py")
        for name in ("predict_frozen", "predict_frontier", "predict_temporal", "predict_dual")]
    current_source = {p.relative_to(root).as_posix(): digest(p) for p in code_paths}
    sys.path.insert(0, str(root))
    import numpy as np
    import pandas as pd
    import sbercluster
    from sbercluster.io import CATEGORIES, TOTAL
    from sbercluster.selection_artifacts import load_artifact, predict_annual_panel
    from sbercluster.selection_frontier import predict_frontier
    from sbercluster.temporal_profiles import predict_temporal_panel
    from sbercluster.dual_profiles import build_reference_packet, predict_dual_panel
    if root not in Path(sbercluster.__file__).resolve().parents:
        raise RuntimeError("Imported prediction code does not belong to --root")
    frontier = root / PACKAGES[0]
    temporal = root / PACKAGES[1]
    model_hashes = read_json(frontier / "models/manifest.json")["files_sha256"]
    labels = read_json(frontier / "constrained/labels.json")
    x = np.load(frontier / "constrained/features.npy", allow_pickle=False)
    if (x.shape != (len(labels["ids"]), 5) or len(set(labels["ids"])) != len(x)
            or not np.isfinite(x).all()):
        raise AcceptanceError("Shipped annual profile keys/shape are inconsistent")
    models, replay = {}, {}
    for name in ("historical_kmeans4", "sw_constrained_ordinary", "sw_constrained_huber75"):
        models[name] = load_artifact(frontier / "models" / (name + ".json"), model_hashes[name + ".json"])
        actual = predict_frontier(x, models[name]["rule"])
        if not np.array_equal(actual, np.asarray(labels["labels"][name])):
            raise AcceptanceError("Shipped annual labels differ: " + name)
        replay[name] = len(actual)
    packet_manifest = read_json(temporal / "paired-reference/packet_manifest.json")["packets"]
    metadata = read_json(temporal / "paired-reference/reference_metadata.json")
    packet_count = verify_reference_packets(temporal / "paired-reference", metadata, packet_manifest)
    # Deliberately artificial rows, generated from stored centers; no source panel is reconstructed.
    scaler = models["historical_kmeans4"]["feature_scaler"]
    centers = np.asarray(models["historical_kmeans4"]["rule"]["centers"])
    rows = []
    for year in (2023, 2024):
        for month in range(1, 13):
            for index, point in enumerate(centers):
                ratio = np.exp(np.asarray(scaler["ratio_center"]) + np.sqrt(5) * point * np.asarray(scaler["ratio_iqr"]))
                rows.append({"entity_id": f"tid_{900000 + index}", "territory_id": str(900000 + index),
                             "region_code": "99", "period": f"{year}-{month:02d}-01", TOTAL: 1000.,
                             **dict(zip(CATEGORIES, 1000. * ratio))})
    panel = pd.DataFrame(rows)
    calls = []
    def cli(arguments, **control):
        calls.append(run_cli(root, arguments, **control))
    with tempfile.TemporaryDirectory(prefix="sbercluster-acceptance-") as temporary:
        work = Path(temporary)
        panel_path = work / "synthetic-panel.csv"
        panel.to_csv(panel_path, index=False)
        # Compare against the rows the CLI actually consumes, including CSV float parsing.
        panel = pd.read_csv(panel_path, dtype={"entity_id": str, "period": str,
                                              "territory_id": str, "region_code": str})
        for name, artifact in models.items():
            output = work / name
            arguments = ["scripts.predict_frontier", "--model", str(frontier / "models" / (name + ".json")),
                         "--expected-sha256", model_hashes[name + ".json"], "--panel", str(panel_path),
                         "--year", "2024", "--output", str(output)]
            cli(arguments)
            actual = pd.read_csv(output / "labels.csv", dtype={"entity_id": str})
            pd.testing.assert_frame_equal(actual, predict_annual_panel(panel, artifact, 2024), check_dtype=False)
            if read_json(output / "status.json")["status"] != "completed":
                raise AcceptanceError("Frontier CLI did not complete")
        temporal_hashes = read_json(temporal / "manifest.json")["files_sha256"]
        for name in ("calendar_only", "calendar_reliability_half", "calendar_reliability_full"):
            model = temporal / "models" / (name + ".json")
            output = work / name
            cli(["scripts.predict_temporal", "--model", str(model), "--expected-sha256",
                 temporal_hashes["models/" + name + ".json"], "--panel", str(panel_path),
                 "--year", "2024", "--output", str(output)])
            actual = pd.read_csv(output / "assignments.csv", dtype={"entity_id": str})
            pd.testing.assert_frame_equal(actual, predict_temporal_panel(panel, read_json(model), 2024),
                                          check_dtype=False, rtol=1e-12, atol=1e-12)
            if read_json(output / "provenance.json")["fitting_executed"] is not False:
                raise AcceptanceError("Temporal CLI claims fitting")
        old = root / "reports/experiments/2026-09-23-v2"
        old_hash = read_json(old / "SHA256.json")["validation/frozen_prototypes.json"]
        for aggregation in ("monthly", "annual"):
            output = work / ("frozen-" + aggregation)
            cli(["scripts.predict_frozen", "--model", str(old / "validation/frozen_prototypes.json"),
                 "--expected-model-sha256", old_hash, "--input", str(panel_path),
                 "--aggregation", aggregation, "--output", str(output)])
            actual = pd.read_csv(output / "assignments.csv")
            if len(actual) != (96 if aggregation == "monthly" else 8):
                raise AcceptanceError("Frozen CLI output row count differs")
            expected = actual.entity_id.map({f"tid_{900000+i}": i for i in range(4)})
            if not np.array_equal(actual.cluster, expected):
                raise AcceptanceError("Frozen CLI did not assign its historical centers")
        reference_path = work / "reference.json"
        cli(["scripts.predict_dual", "lock", "--model", str(frontier / "models/historical_kmeans4.json"),
             "--expected-model-sha256", model_hashes["historical_kmeans4.json"], "--panel", str(panel_path),
             "--output", str(reference_path)])
        reference = read_json(reference_path)
        packets = {f"2024-{m:02d}-01": build_reference_packet(panel, reference, f"2024-{m:02d}-01")
                   for m in range(1, 13)}
        packet_path = work / "cli-packet.json"
        cli(["scripts.predict_dual", "packet", "--reference", str(reference_path),
             "--expected-reference-sha256", digest(reference_path), "--panel", str(panel_path),
             "--period", "2024-01-01", "--output", str(packet_path)])
        cli_packet = read_json(packet_path)
        for key, value in packets["2024-01-01"].items():
            if key not in ("source_provenance", "content_sha256") and cli_packet[key] != value:
                raise AcceptanceError("Dual packet CLI differs: " + key)
        entries = {}
        for period, packet in packets.items():
            path = work / (period + ".json")
            path.write_text(json.dumps(packet, allow_nan=False), encoding="utf-8")
            entries[period] = {"path": path.name, "sha256": digest(path)}
        packets_path = work / "packets.json"
        packets_path.write_text(json.dumps({"packets": entries}), encoding="utf-8")
        output = work / "dual-predictions.json"
        dual_args = ["scripts.predict_dual", "predict", "--reference", str(reference_path),
                     "--expected-reference-sha256", digest(reference_path), "--panel", str(panel_path),
                     "--year", "2024", "--packet-manifest", str(packets_path),
                     "--expected-packet-manifest-sha256", digest(packets_path), "--output", str(output)]
        cli(dual_args)
        actual = pd.DataFrame(read_json(output)["records"])
        expected = predict_dual_panel(panel, reference, 2024, packets)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
        output_hash = digest(output)
        cli(dual_args, expected_refusal=("FileExistsError", "File exists"), output=output, preserve_sha256=output_hash)
        wrong_hash = dual_args.copy()
        wrong_hash[wrong_hash.index("--expected-reference-sha256") + 1] = "0" * 64
        wrong_hash[-1] = str(work / "rejected.json")
        cli(wrong_hash, expected_refusal=("ValueError", "Pinned file SHA256 differs:"), output=wrong_hash[-1])
    if current_source != {p.relative_to(root).as_posix(): digest(p) for p in code_paths}:
        raise AcceptanceError("Current prediction source changed during verification")
    return {"status": "PASS", "annual_exact_label_replay": replay, "shipped_reference_packets": packet_count,
            "synthetic_input_rows": len(panel), "cli_calls": calls, "current_source_sha256": current_source,
            "packet_validation": "canonical validator with public ID/region/seal context; private paired corrections not recomputed",
            "scope": "saved-profile replay and synthetic CLI interoperability; not raw-data reconstruction"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Checkout containing code and shipped report packages")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report; existing files are refused")
    parser.add_argument("--integrity-only", action="store_true", help="Standard-library payload check; predictions are explicitly NOT_RUN")
    args = parser.parse_args(argv)
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() or any((root / name).resolve() in output.parents for name in PACKAGES):
        parser.error("Use a new report file outside immutable evidence packages")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ[key] = "1"
    started = time.monotonic()
    report = {"schema_version": 1, "root": str(root), "fitting_executed": False,
              "raw_data_used": False, "scientific_status": "not_assessed", "status": "FAIL"}
    code = 1
    try:
        report["integrity"] = verify_integrity(root)
        report["predictions"] = {"status": "NOT_RUN", "reason": "--integrity-only"} if args.integrity_only else verify_predictions(root)
        # Recheck payloads, not just manifests: CLI readers must share this evidence snapshot.
        after = verify_integrity(root)
        if [r["manifest_sha256"] for r in after] != [r["manifest_sha256"] for r in report["integrity"]]:
            raise AcceptanceError("Evidence manifest changed during prediction checks")
        report["payload_snapshot_stable"] = True
        report["status"] = "PASS_INTEGRITY_ONLY" if args.integrity_only else "PASS"
        code = 0
    except (ImportError, RuntimeError, subprocess.TimeoutExpired) as error:
        report["status"] = "SETUP_ERROR"
        report["error"] = type(error).__name__ + ": " + str(error)
        if hasattr(error, "cli_result"):
            report["failed_cli"] = error.cli_result
        code = 2
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, AssertionError) as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        if hasattr(error, "cli_result"):
            report["failed_cli"] = error.cli_result
    report["elapsed_seconds"] = time.monotonic() - started
    report["exit_code"] = code
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except OSError as error:
        print(json.dumps({"status": "SETUP_ERROR", "error": "Cannot create report: " + str(error)}))
        return 2
    print(json.dumps({"status": report["status"], "exit_code": code, "output": str(output),
                      "error": report.get("error"), "elapsed_seconds": round(report["elapsed_seconds"], 3)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
