"""Hash-pinned retrospective annual endpoint validation of stored predictions.

There is no estimator fitting here. Identifier admission is distinct from
certification of longitudinal geographic boundaries. All source outcomes used
by every predictor come from one shared observed cohort.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Iterable, Mapping
import zlib

import numpy as np


CONTRACT_SHA256 = "1954c73a4c2d9fb90b7ade073ca647919a873fa838fb67b6ebacb7b9f37f84fc"
PROTOCOL_SHA256 = "18725c1f20bf19919e4eeb545c80669ceb1cc708ef9f5e39f4f1b577d8606872"
ADMISSION_SHA256 = "379ceb7ca1328b034a5e1d86f0bfca77f56f53488dfa94b1e6156fdbde85061c"
PREDICTORS = tuple(f"{learner}_{feature}" for learner in
                   ("linear", "quadratic", "histogram") for feature in
                   ("controls", "continuous5", "kmeans4", "huber75"))
PRIMARY = ("histogram_huber75", "histogram_continuous5")
TARGET = {"indicator_code": "Y48423007", "year": "2025",
          "indicator_period": "Январь-декабрь", "indicator_unit": "Рубль",
          "okved2": "Всего по обследуемым видам экономической деятельности"}
BOOTSTRAP = 1999
SEED = 20261003
IDENTITY_CHECKS = ("exact_year_validity", "exact_name", "exact_region_name", "upper_level")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256_file(path) != expected:
        raise ValueError(f"SHA256 mismatch: {Path(path).name}")


def load_contract(path: Path, expected_sha256: str = CONTRACT_SHA256) -> dict:
    """The scientific contract itself has an external byte pin."""
    if expected_sha256 != CONTRACT_SHA256:
        raise ValueError("This endpoint implementation requires its frozen contract pin")
    verify_file(path, expected_sha256)
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract["provenance"]["fixed_protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("Protocol provenance changed")
    if contract["provenance"]["metadata_admission_sha256"] != ADMISSION_SHA256:
        raise ValueError("Metadata provenance changed")
    if tuple(contract["predictors"]) != PREDICTORS:
        raise ValueError("Frozen prediction columns changed")
    target = contract["target"]
    if target != {"indicator": TARGET["indicator_code"], "year": TARGET["year"],
                  "period": TARGET["indicator_period"], "unit": TARGET["indicator_unit"],
                  "sector": TARGET["okved2"]}:
        raise ValueError("Fixed annual target changed")
    if (contract["primary"]["minuend"], contract["primary"]["subtrahend"]) != PRIMARY:
        raise ValueError("Primary contrast changed")
    if contract["bootstrap"]["replicates"] != BOOTSTRAP or contract["bootstrap"]["seed"] != SEED:
        raise ValueError("Conditional bootstrap specification changed")
    return contract


@dataclass(frozen=True)
class FrozenPredictions:
    ids: tuple[str, ...]
    regions: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        if not self.ids or len(set(self.ids)) != len(self.ids) or any(not s for s in self.ids):
            raise ValueError("Nonempty unique stable prediction IDs required")
        regions = np.asarray(self.regions)
        values = np.asarray(self.values, dtype=float)
        if regions.shape != (len(self.ids),) or regions.dtype.kind not in "iu":
            raise ValueError("Every frozen ID requires an integer region")
        if values.shape != (len(self.ids), len(PREDICTORS)) or not np.isfinite(values).all():
            raise ValueError("All twelve predictions must be finite for every frozen ID")
        regions = regions.copy()
        values = values.copy()
        regions.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "values", values)


def load_predictions(path: Path, expected_count: int, expected_regions: int) -> FrozenPredictions:
    ids, regions, values = [], [], []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        required = {"entity_id", "held_out_region", *PREDICTORS}
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Unique prediction headers required")
        if not required.issubset(reader.fieldnames):
            raise ValueError("Missing fixed prediction column")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed prediction CSV row")
            ids.append(row["entity_id"])
            if not re.fullmatch(r"[0-9]+", row["held_out_region"]):
                raise ValueError("Invalid frozen region")
            regions.append(int(row["held_out_region"]))
            # Historical observed2023/observed2024 columns are not converted or used.
            values.append([float(row[name]) for name in PREDICTORS])
    frozen = FrozenPredictions(tuple(ids), np.array(regions, dtype=int), np.array(values))
    if len(ids) != expected_count or len(set(regions)) != expected_regions:
        raise ValueError("Frozen cohort size or region count changed")
    return frozen


def align_membership(rows: list[dict], predictions: FrozenPredictions,
                     expected_admitted: int | None = None,
                     expected_flags: int | None = None) -> tuple[dict, ...]:
    """Align identifiers without any outcome-based exclusions or name matching."""
    by_id = {}
    for original in rows:
        row = dict(original)
        entity = row["entity_id"]
        if entity in by_id:
            raise ValueError("Duplicate membership ID")
        if not isinstance(row["source_oktmo8"], str) or not re.fullmatch(r"\d{8}", row["source_oktmo8"]):
            raise ValueError("Exact eight-digit source key required")
        n = row["source_candidate_count"]
        if type(n) is not int or n < 0:
            raise ValueError("Invalid source candidate count")
        for key in (*IDENTITY_CHECKS, "metadata_identity_pass", "territorial_change_flag"):
            if type(row[key]) is not bool:
                raise ValueError("Identity flags must be booleans")
        admitted = n == 1 and all(row[key] for key in IDENTITY_CHECKS)
        if admitted != row["metadata_identity_pass"]:
            raise ValueError("Inconsistent frozen identifier admission")
        by_id[entity] = row
    if set(by_id) != set(predictions.ids):
        raise ValueError("Membership must retain every frozen prediction ID exactly")
    aligned = tuple(by_id[entity] for entity in predictions.ids)
    codes = [row["source_oktmo8"] for row in aligned]
    if len(set(codes)) != len(codes):
        raise ValueError("Source keys cannot assign two frozen IDs")
    admitted = sum(row["metadata_identity_pass"] for row in aligned)
    flagged = sum(row["metadata_identity_pass"] and row["territorial_change_flag"] for row in aligned)
    if expected_admitted is not None and admitted != expected_admitted:
        raise ValueError("Metadata-admitted membership changed")
    if expected_flags is not None and flagged != expected_flags:
        raise ValueError("Frozen territorial flags changed")
    return aligned


def source_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(row[name] for name in ("indicator_code", "year", "indicator_period", "okved2", "oktmo"))


def load_source_metadata(path: Path, expected_count: int) -> dict[tuple[str, ...], dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames or any("value" in c.casefold() for c in reader.fieldnames):
            raise ValueError("Source metadata must exclude measurement columns")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Unique source metadata headers required")
        if not {*TARGET, "oktmo"}.issubset(reader.fieldnames):
            raise ValueError("Required annual metadata columns absent")
        rows = {}
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed source metadata CSV row")
            if any(row[name] != value for name, value in TARGET.items()):
                raise ValueError("Frozen metadata contains a different endpoint")
            key = source_key(row)
            if key in rows:
                raise ValueError("Duplicate source metadata key; no averaging")
            rows[key] = row
    if len(rows) != expected_count:
        raise ValueError("Annual source metadata count changed")
    return rows


@dataclass(frozen=True)
class VerifiedInputs:
    contract_path: Path
    contract: dict
    paths: dict[str, Path]
    predictions: FrozenPredictions
    membership: tuple[dict, ...]
    source_metadata: dict[tuple[str, ...], dict]


def verify_inputs(contract_path: Path, paths: Mapping[str, Path],
                  contract_sha256: str = CONTRACT_SHA256) -> VerifiedInputs:
    """Verify all scientific input byte pins before opening source values."""
    contract = load_contract(contract_path, contract_sha256)
    if set(paths) != set(contract["pins"]):
        raise ValueError("Supply each pinned scientific input exactly once")
    clean_paths = {role: Path(path) for role, path in paths.items()}
    for role, path in clean_paths.items():
        verify_file(path, contract["pins"][role])
    if clean_paths["source_member_deflate"].stat().st_size != contract["source_member"]["compressed_bytes"]:
        raise ValueError("Compressed member length changed")
    counts = contract["cohort"]
    predictions = load_predictions(clean_paths["predictions"], counts["fixed_entities"], counts["fixed_regions"])
    membership = align_membership(json.loads(clean_paths["membership"].read_text(encoding="utf-8")),
                                  predictions, counts["metadata_admitted"], counts["metadata_history_flags"])
    metadata = load_source_metadata(clean_paths["source_metadata"], counts["source_annual_total_rows"])
    for row in membership:
        key = (TARGET["indicator_code"], TARGET["year"], TARGET["indicator_period"],
               TARGET["okved2"], row["source_oktmo8"])
        if row["source_candidate_count"] != int(key in metadata):
            raise ValueError("Source candidate count differs from frozen metadata")
    return VerifiedInputs(Path(contract_path), contract, clean_paths, predictions, membership, metadata)


class _DeflateReader(io.RawIOBase):
    """Bounded raw-DEFLATE reader with completion CRC/length checks."""
    def __init__(self, path: Path, expected_length: int, expected_crc: int):
        super().__init__()
        self.stream = Path(path).open("rb")
        self.decompressor = zlib.decompressobj(-15)
        self.pending = b""
        self.length = 0
        self.crc = 0
        self.expected_length = expected_length
        self.expected_crc = expected_crc
        self.complete = False

    def readable(self):
        return True

    def readinto(self, buffer):
        if self.complete:
            return 0
        while True:
            if self.decompressor.eof:
                if self.pending or self.decompressor.unused_data or self.stream.read(1):
                    raise ValueError("Trailing compressed member bytes")
                if self.length != self.expected_length or self.crc != self.expected_crc:
                    raise ValueError("Source member length or CRC mismatch")
                self.complete = True
                return 0
            if not self.pending:
                self.pending = self.stream.read(65536)
                if not self.pending:
                    raise ValueError("Truncated DEFLATE member")
            chunk = self.decompressor.decompress(self.pending, len(buffer))
            self.pending = self.decompressor.unconsumed_tail
            if chunk:
                self.length += len(chunk)
                if self.length > self.expected_length:
                    raise ValueError("Source member exceeds pinned uncompressed size")
                self.crc = zlib.crc32(chunk, self.crc)
                buffer[:len(chunk)] = chunk
                return len(chunk)

    def close(self):
        self.stream.close()
        super().close()


def iter_source_records(path: Path, expected_length: int, expected_crc: int) -> Iterable[dict]:
    """Read the cached member, not a network URL. Fully consume to verify CRC."""
    raw = _DeflateReader(path, expected_length, expected_crc)
    with io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";", strict=True)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("Unique source headers required")
        if not {*TARGET, "oktmo", "indicator_value"}.issubset(fields):
            raise ValueError("Required source schema absent")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed source CSV row")
            yield row
    if not raw.complete:
        raise ValueError("Source member was not completely verified")


def parse_wage(token: str) -> tuple[str, float | None]:
    text = str(token).strip()
    if text.casefold() in {"", "...", "…", "x", "х", "na", "n/a", "null", "none"}:
        return "missing_or_suppressed", None
    if text.upper() in {"SD", "UD", "ND", "CD", "NR"}:
        # Preserve the source marker instead of asserting a numeric value or
        # silently collapsing an unknown/conflicting source code into zero.
        return f"source_marker_{text.upper()}", None
    if text in {"-", "–", "—"}:
        return "not_applicable_or_absent", None
    try:
        wage = float(re.sub(r"\s+", "", text).replace(",", "."))
    except ValueError:
        return "invalid_numeric", None
    if not np.isfinite(wage):
        return "nonfinite", None
    if wage == 0:
        return "zero", None
    if wage < 0:
        return "negative", None
    return "observed", float(np.log(wage))


def collect_outcomes(records: Iterable[Mapping[str, str]], membership: tuple[dict, ...],
                     metadata: Mapping[tuple[str, ...], dict]) -> dict[str, tuple[str, float | None]]:
    admitted = {r["source_oktmo8"] for r in membership if r["metadata_identity_pass"]}
    seen, outcomes = set(), {}
    for row in records:
        if row["indicator_code"] != TARGET["indicator_code"] or row["year"] != TARGET["year"]:
            raise ValueError("Wrong indicator or year in the pinned source member")
        if row["indicator_period"] != TARGET["indicator_period"] or row["okved2"] != TARGET["okved2"]:
            continue  # No quarter fallback; these values are not converted or used.
        if row["indicator_unit"] != TARGET["indicator_unit"]:
            raise ValueError("Annual source unit differs from frozen rubles")
        key = source_key(row)
        if key in seen:
            raise ValueError("Duplicate annual source key; no averaging")
        seen.add(key)
        expected = metadata.get(key)
        if expected is None or any(row.get(name) != value for name, value in expected.items()):
            raise ValueError("Annual identity metadata differs from its byte-pinned projection")
        if row["oktmo"] in admitted:
            outcomes[row["oktmo"]] = parse_wage(row["indicator_value"])
    if not seen:
        raise ValueError("Exact annual endpoint absent; no alternative period allowed")
    if seen != set(metadata):
        raise ValueError("Annual source keys differ from frozen metadata; no silent cohort shrink")
    return outcomes


def paired_metrics(y, predictions, regions, bootstrap: int = BOOTSTRAP, seed: int = SEED) -> dict:
    """Equal region weights, shared paired draws, and an explicit signed primary."""
    y = np.asarray(y, dtype=float)
    values = np.asarray(predictions, dtype=float)
    regions = np.asarray(regions)
    if y.ndim != 1 or values.shape != (len(y), len(PREDICTORS)) or regions.shape != y.shape:
        raise ValueError("Aligned outcomes, twelve predictions and regions required")
    if not np.isfinite(y).all() or not np.isfinite(values).all() or regions.dtype.kind not in "iu":
        raise ValueError("All models must share finite observations and integer regions")
    if type(bootstrap) is not int or bootstrap < 1:
        raise ValueError("Positive conditional bootstrap count required")
    groups = [np.flatnonzero(regions == region) for region in np.unique(regions)]
    metrics, primary = [], {"minuend": PRIMARY[0], "subtrahend": PRIMARY[1],
        "metric": "equal_region_log_mse", "negative_favors": PRIMARY[0],
        "difference": None, "conditional_paired_region_bootstrap_95": None}
    if not groups:
        return {"status": "not_estimable", "primary": primary,
                "centered_error_scope": {"definition": "Mean sample variance over observed regions with at least two entities", "eligible_regions": 0},
                "metrics": [{"predictor": name, "n": 0, "regions": 0,
                             "municipality_rmse": None, "municipality_mae": None,
                             "equal_region_mse": None,
                             "equal_region_centered_error_sample_variance": None,
                             "regions_with_pairs": 0, "comparisons": {}} for name in PREDICTORS]}
    error = y[:, None] - values
    regional = np.array([np.mean(error[ix] ** 2, axis=0) for ix in groups])
    pair_groups = [ix for ix in groups if len(ix) >= 2]
    centered = np.array([np.var(error[ix], axis=0, ddof=1) for ix in pair_groups])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(groups), size=(bootstrap, len(groups)))
    boot_loss = regional[draws].mean(axis=1)
    if pair_groups:
        pair_draws = rng.integers(len(pair_groups), size=(bootstrap, len(pair_groups)))
        boot_centered = centered[pair_draws].mean(axis=1)
    else:
        boot_centered = None
    mean_loss = regional.mean(axis=0)
    for i, name in enumerate(PREDICTORS):
        comparisons = {}
        for j, reference in enumerate(PREDICTORS):
            comparisons[reference] = {
                "equal_region_mse_difference": float(mean_loss[i] - mean_loss[j]),
                "conditional_paired_region_bootstrap_95": np.quantile(boot_loss[:, i] - boot_loss[:, j], [.025, .975]).tolist(),
                "equal_region_centered_variance_difference": float(centered[:, i].mean() - centered[:, j].mean()) if pair_groups else None,
                "conditional_centered_region_bootstrap_95": np.quantile(boot_centered[:, i] - boot_centered[:, j], [.025, .975]).tolist() if pair_groups else None}
        metrics.append({"predictor": name, "n": len(y), "regions": len(groups),
                        "municipality_rmse": float(np.sqrt(np.mean(error[:, i] ** 2))),
                        "municipality_mae": float(np.mean(np.abs(error[:, i]))),
                        "equal_region_mse": float(mean_loss[i]),
                        "equal_region_centered_error_sample_variance": float(centered[:, i].mean()) if pair_groups else None,
                        "regions_with_pairs": len(pair_groups), "comparisons": comparisons})
    i, j = (PREDICTORS.index(name) for name in PRIMARY)
    primary["difference"] = float(mean_loss[i] - mean_loss[j])
    primary["conditional_paired_region_bootstrap_95"] = np.quantile(boot_loss[:, i] - boot_loss[:, j], [.025, .975]).tolist()
    return {"status": "observed_cohort_only", "primary": primary, "metrics": metrics,
            "centered_error_scope": {"definition": "Mean sample variance over observed regions with at least two entities", "eligible_regions": len(pair_groups)}}


def evaluate_records(predictions: FrozenPredictions, membership: tuple[dict, ...],
                     metadata: Mapping[tuple[str, ...], dict], records: Iterable[Mapping[str, str]],
                     bootstrap: int = BOOTSTRAP, seed: int = SEED) -> dict:
    membership = align_membership(list(membership), predictions)
    outcomes = collect_outcomes(records, membership, metadata)
    audit, observed = [], []
    for i, row in enumerate(membership):
        if row["metadata_identity_pass"]:
            status, wage = outcomes[row["source_oktmo8"]]
        elif row["source_candidate_count"] == 0:
            status, wage = "identity_unmatched", None
        elif not row["exact_name"]:
            status, wage = "identity_name_failed", None
        else:
            status, wage = "identity_rejected", None
        audit.append({"entity_id": predictions.ids[i], "held_out_region": int(predictions.regions[i]),
                      "source_oktmo8": row["source_oktmo8"], "metadata_identity_pass": row["metadata_identity_pass"],
                      "territorial_change_flag": row["territorial_change_flag"],
                      "identity_checks_failed": [k for k in IDENTITY_CHECKS if not row[k]],
                      "source_candidate_count": row["source_candidate_count"],
                      "outcome_status": status, "log_wage": wage})
        if status == "observed":
            observed.append(i)
    indices = np.array(observed, dtype=int)
    y = np.array([audit[i]["log_wage"] for i in observed], dtype=float)
    result = paired_metrics(y, predictions.values[indices], predictions.regions[indices], bootstrap, seed)
    status_counts = {status: sum(r["outcome_status"] == status for r in audit)
                     for status in sorted({r["outcome_status"] for r in audit})}
    region_audit = [{"region": int(region), "fixed_entities": int(np.sum(predictions.regions == region)),
                     "observed_entities": sum(a["held_out_region"] == int(region) and a["outcome_status"] == "observed" for a in audit)}
                    for region in np.unique(predictions.regions)]
    result.update({"audit": audit, "region_audit": region_audit,
                   "coverage": {"fixed_entities": len(audit), "metadata_admitted": sum(r["metadata_identity_pass"] for r in audit),
                                "observed_entities": len(observed), "status_counts": status_counts,
                                "fixed_regions": len(region_audit), "observed_regions": len(np.unique(predictions.regions[indices])),
                                "entirely_unobserved_regions": [r["region"] for r in region_audit if not r["observed_entities"]],
                                "metadata_history_flags": sum(r["metadata_identity_pass"] and r["territorial_change_flag"] for r in audit),
                                "observed_history_flags": sum(r["outcome_status"] == "observed" and r["territorial_change_flag"] for r in audit)},
                   "uncertainty": {"replicates": bootstrap, "seed": seed, "unit": "paired observed regions",
                                   "interpretation": "Descriptive conditional intervals; overlapping fits and spatial dependence prevent a generic95%coverage claim."},
                   "claim_limits": {"full_cohort_effect_identified": False, "boundary_continuity_certified": False,
                                    "calendar_prospective_forecast": False, "production_typology_identified": False,
                                    "unknown_log_wage_losses_have_finite_bounds": False}})
    return result


def evaluate_verified(inputs: VerifiedInputs) -> dict:
    # Reload byte-pinned inputs rather than trusting caller-mutated in-memory
    # arrays or membership dictionaries after initial verification.
    inputs = verify_inputs(inputs.contract_path, inputs.paths, CONTRACT_SHA256)
    member = inputs.contract["source_member"]
    result = evaluate_records(inputs.predictions, inputs.membership, inputs.source_metadata,
                              iter_source_records(inputs.paths["source_member_deflate"], member["uncompressed_bytes"], member["crc32"]))
    for role, path in inputs.paths.items():
        verify_file(path, inputs.contract["pins"][role])
    result["provenance"] = {"contract_sha256": CONTRACT_SHA256,
                            **inputs.contract["provenance"], "input_sha256": dict(inputs.contract["pins"])}
    return result
