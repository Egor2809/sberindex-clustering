"""Restore the three pinned public Rosstat-derived IfTochno archives.

Official source measurements: Rosstat PMO. Table transformation and archive
publisher: IfTochno, CC BY4.0. HTTPS certificates are verified for these archives.
Files are size-bounded, SHA-256 checked and never silently overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from scripts.download_external_national import ROOT, sha256


def download(source: dict, output: Path) -> str:
    filename = source["file"]
    if Path(filename).name != filename or not re.fullmatch(r"[a-z]+-iftochno\.zip", filename):
        raise ValueError("Unsafe archive filename")
    parsed = urllib.parse.urlsplit(source["url"])
    if parsed.scheme != "https" or parsed.hostname != "storage.yandexcloud.net" or not parsed.path.startswith("/tochno-st-catalog/Rosstat/"):
        raise ValueError("Unexpected public archive URL")
    expected = source["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid pinned archive hash")
    size_bound = 180 * 1024 * 1024
    target = output / filename
    if target.exists():
        if sha256(target) != expected:
            raise FileExistsError(f"Refusing to overwrite mismatching source: {target}")
        return f"reused {filename}"
    output.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output, prefix=f".{filename}.", suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            request = urllib.request.Request(source["url"], headers={"User-Agent": "SberAI-reproducibility/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if urllib.parse.urlsplit(response.url).hostname != parsed.hostname:
                    raise ValueError("Unexpected redirect host")
                total = 0
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > size_bound:
                        raise ValueError("Source archive exceeds bound")
                    stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if sha256(temporary) != expected:
            raise ValueError(f"Source hash changed; review required: {filename}")
        temporary.replace(target)
        temporary = None
        return f"downloaded {filename}"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=ROOT / "reports/external-national-2026-10-03/source-audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sources/external-national-2026-10-03")
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    sources = audit["archive_sources"]
    if {source["kind"] for source in sources} != {"population", "headcount", "wage"} or len(sources) != 3:
        raise ValueError("Expected exactly three pinned sources")
    for source in sources:
        print(download(source, args.output), flush=True)
    # Preparation records the same pinned acquisition manifest in a new cache.
    (args.output / "archive-sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
