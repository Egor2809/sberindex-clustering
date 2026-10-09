"""Download the small official Rosstat workbooks used by external round 2.

Expected URLs, filenames and SHA-256 digests come from the checked-in source
audit.  Existing files are reused only when their digest matches; a mismatching
file is never overwritten.  Downloads are size-limited and atomically renamed
after digest verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


MAX_SOURCE_BYTES = 5 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoded_https_url(raw_url: str) -> str:
    parts = urllib.parse.urlsplit(raw_url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError(f"Expected an HTTPS source URL, got {raw_url!r}")
    if not parts.hostname.casefold().endswith(".rosstat.gov.ru"):
        raise ValueError(f"Unexpected source host: {parts.hostname!r}")
    path = urllib.parse.quote(parts.path, safe="/%:@!$&'()*+,;=-._~")
    query = urllib.parse.quote(parts.query, safe="%=&/:?@!$'()*+,;-._~")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, query, ""))


def validated_sources(audit_path: Path) -> list[dict[str, str]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sources = audit.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"No sources in {audit_path}")
    result: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for source in sources:
        filename = str(source.get("file", ""))
        digest = str(source.get("sha256", "")).casefold()
        url = str(source.get("url", ""))
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Unsafe source filename: {filename!r}")
        if filename in seen_names:
            raise ValueError(f"Duplicate source filename: {filename!r}")
        if not SHA256_RE.fullmatch(digest):
            raise ValueError(f"Invalid SHA-256 for {filename!r}: {digest!r}")
        result.append(
            {"file": filename, "sha256": digest, "url": encoded_https_url(url)}
        )
        seen_names.add(filename)
    return result


def download_one(source: dict[str, str], output_dir: Path) -> str:
    destination = output_dir / source["file"]
    if destination.exists():
        actual = sha256_file(destination)
        if actual != source["sha256"]:
            raise FileExistsError(
                f"Refusing to overwrite {destination}: SHA-256 {actual} does not "
                f"match expected {source['sha256']}"
            )
        return f"reused {destination.name} ({destination.stat().st_size} bytes)"

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=output_dir,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = urllib.request.Request(
                source["url"], headers={"User-Agent": "SberAI-reproducibility/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_SOURCE_BYTES:
                    raise ValueError(
                        f"Source exceeds {MAX_SOURCE_BYTES} bytes: {source['file']}"
                    )
                total = 0
                while True:
                    block = response.read(64 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > MAX_SOURCE_BYTES:
                        raise ValueError(
                            f"Source exceeds {MAX_SOURCE_BYTES} bytes: {source['file']}"
                        )
                    temporary.write(block)
            temporary.flush()
            os.fsync(temporary.fileno())

        actual = sha256_file(temporary_path)
        if actual != source["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {source['file']}: {actual} != {source['sha256']}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return f"downloaded {destination.name} ({destination.stat().st_size} bytes)"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=root / "reports/external-round2/source_mapping_audit.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "artifacts/sources/round2"
    )
    args = parser.parse_args()

    for source in validated_sources(args.audit):
        print(download_one(source, args.output_dir))


if __name__ == "__main__":
    main()
