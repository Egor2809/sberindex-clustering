"""Acquire immutable Rosstat PMO municipality passports for a frozen cohort.

The registry is joined by stable territory ID at the requested year. Rosstat's
public passport query uses the eight-digit municipal OKTMO, with leading zeros
omitted, followed by the year. Every response (including absent tables) is kept
and hashed. No name matching assigns identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://rosstat.gov.ru/scripts/db_inet2/passport/table.aspx"
MAX_BYTES = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort(year: int, root: Path = ROOT) -> pd.DataFrame:
    labels = pd.read_csv(root / "reports/experiments/2026-09-23-v2/validation/reference_assignments.csv")
    registry = pd.read_excel(root / "artifacts/sources/acquisition/t_dict_municipal_districts.xlsx")
    registry = registry.loc[(registry.year_from <= year) & (year < registry.year_to)].copy()
    registry["entity_id"] = "tid_" + registry.territory_id.astype(int).astype(str)
    if labels.entity_id.duplicated().any() or registry.entity_id.duplicated().any():
        raise ValueError("Duplicate stable ID or year-valid registry version")
    merged = labels.merge(registry, on="entity_id", how="left", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Frozen cohort has missing registry version")
    merged["source_oktmo8"] = merged.oktmo.map(lambda x: re.sub(r"\D", "", str(x))[:8])
    if not merged.source_oktmo8.str.fullmatch(r"\d{8}").all():
        raise ValueError("Invalid municipal OKTMO")
    if merged.source_oktmo8.duplicated().any():
        raise ValueError("Eight-digit OKTMO is not one-to-one in frozen cohort")
    return merged.drop(columns="_merge").sort_values(["region_code", "territory_id"])


def download_one(row: dict, year: int, directory: Path, allow_unverified_tls: bool) -> dict:
    code = row["source_oktmo8"]
    target = directory / f"{code}-{year}.html"
    url = f"{BASE}?opt={int(code)}{year}"
    record = {"entity_id": row["entity_id"], "source_oktmo8": code,
              "registry_oktmo": row["oktmo"], "region_code": int(row["region_code"]),
              "registry_name": row["municipal_district_name"], "year": year,
              "url": url, "file": target.name,
              "tls_certificate_verified": not allow_unverified_tls}
    if target.exists():
        record.update(status="cached", bytes=target.stat().st_size, sha256=sha256(target))
        return record
    for attempt in range(2):
        temporary = target.with_suffix(".part")
        try:
            context = ssl._create_unverified_context() if allow_unverified_tls else ssl.create_default_context()
            request = urllib.request.Request(url, headers={"User-Agent": "SberAI-public-statistics-audit/1.0"})
            with urllib.request.urlopen(request, timeout=25, context=context) as response:
                if response.url.split("/")[2] != "rosstat.gov.ru":
                    raise ValueError("Unexpected response host")
                size = 0
                with temporary.open("wb") as stream:
                    while True:
                        block = response.read(65536)
                        if not block:
                            break
                        size += len(block)
                        if size > MAX_BYTES:
                            raise ValueError("Passport exceeds size bound")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(target)
                record.update(status="downloaded", http_status=response.status,
                              bytes=size, sha256=sha256(target),
                              retrieved_at=datetime.now(timezone.utc).isoformat(), attempts=attempt + 1)
                return record
        except (urllib.error.URLError, ValueError, OSError) as exc:
            temporary.unlink(missing_ok=True)
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}", attempts=attempt + 1)
            if attempt == 0:
                time.sleep(1)
        finally:
            time.sleep(0.2)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=[2023, 2024], default=2023)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sources/external-national-2026-10-03")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--allow-unverified-tls", action="store_true",
                        help="Explicit public-data transport exception, recorded per response; never changes global TLS settings")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = cohort(args.year).to_dict("records")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("Positive limit required")
        rows = rows[:args.limit]
    manifest_path = args.output / f"acquisition-{args.year}.jsonl"
    done = 0
    successful = 0
    start = time.monotonic()
    with manifest_path.open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=args.workers) as executor:
        for record in executor.map(lambda row: download_one(row, args.year, args.output, args.allow_unverified_tls), rows):
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            done += 1
            successful += record["status"] != "failed"
            if done % 100 == 0 or done == len(rows):
                print(json.dumps({"done": done, "total": len(rows), "successful_responses": successful,
                                  "elapsed_seconds": round(time.monotonic() - start, 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
