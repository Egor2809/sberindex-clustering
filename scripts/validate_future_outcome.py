"""Check scientific input pins or evaluate a stored annual endpoint once."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sbercluster import future_outcome_validation as validation


def write_artifacts(result: dict, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty evaluation output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "audit.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = list(result["audit"][0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["audit"]:
            writer.writerow({**row, "identity_checks_failed": json.dumps(row["identity_checks_failed"])})
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [key for key in result["metrics"][0] if key != "comparisons"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["metrics"])
    payloads = ("evaluation.json", "audit.csv", "metrics.csv")
    (output / "manifest.json").write_text(json.dumps(
        {"schema": 1, "files": {name: validation.sha256_file(output / name) for name in payloads}},
        indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", default=validation.CONTRACT_SHA256)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--membership", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--source-member", type=Path, required=True, help="Cached raw-DEFLATE member")
    parser.add_argument("--stage", choices=("check", "score"), default="check")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.stage == "score":
        if args.output is None:
            parser.error("--output is required for score")
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Evaluation output must be new or empty")
    paths = {"predictions": args.predictions, "membership": args.membership,
             "source_metadata": args.source_metadata, "source_member_deflate": args.source_member}
    inputs = validation.verify_inputs(args.contract, paths, args.contract_sha256)
    if args.stage == "check":
        print(json.dumps({"status": "scientific_input_pins_verified", "outcome_values_parsed": False,
                          "fixed_entities": len(inputs.predictions.ids),
                          "metadata_admitted": sum(r["metadata_identity_pass"] for r in inputs.membership),
                          "history_flags": sum(r["metadata_identity_pass"] and r["territorial_change_flag"] for r in inputs.membership),
                          "boundary_continuity_certified": False,
                          "contract_sha256": validation.CONTRACT_SHA256}, ensure_ascii=False))
        return
    result = validation.evaluate_verified(inputs)
    result["code_sha256"] = {
        "sbercluster/future_outcome_validation.py": validation.sha256_file(Path(validation.__file__)),
        "scripts/validate_future_outcome.py": validation.sha256_file(Path(__file__))}
    result["runtime"] = {"python": sys.version.split()[0], "numpy": np.__version__}
    write_artifacts(result, args.output)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"],
                      "primary": result["primary"], "output": str(args.output.resolve())},
                     ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
