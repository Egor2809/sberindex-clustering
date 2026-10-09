"""Download open municipal statistics used by the external municipal check.

Sources, filenames and SHA-256 digests come from the checked-in manifest.
Whole files are downloaded with a size limit.  Large tochno.st archives are not
downloaded: the required CSV member is streamed through HTTP range requests,
its decompressed SHA-256 is verified, and only the manifest-listed rows are
kept.  Existing outputs are reused only when their digest matches; mismatching
files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import ssl
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


MAX_SOURCE_BYTES = 8 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_HOSTS = ("rosstat.gov.ru", "storage.yandexcloud.net")
USER_AGENT = "SberAI-reproducibility/1.0"


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
    host = parts.hostname.casefold()
    if not any(host == allowed or host.endswith("." + allowed) for allowed in ALLOWED_HOSTS):
        raise ValueError(f"Unexpected source host: {parts.hostname!r}")
    path = urllib.parse.quote(parts.path, safe="/%:@!$&'()*+,;=-._~")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def validated_sources(manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"No sources in {manifest_path}")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in sources:
        filename = str(source.get("file", ""))
        if not filename or Path(filename).name != filename or filename in seen:
            raise ValueError(f"Unsafe or duplicate source filename: {filename!r}")
        digests = [str(source.get("sha256", "")).casefold()]
        if "member" in source:
            digests.append(str(source.get("member_sha256", "")).casefold())
        if not all(SHA256_RE.fullmatch(value) for value in digests):
            raise ValueError(f"Invalid SHA-256 for {filename!r}")
        item = dict(source)
        item["url"] = encoded_https_url(str(source.get("url", "")))
        item["sha256"] = digests[0]
        if "member" in source:
            item["member_sha256"] = digests[1]
        result.append(item)
        seen.add(filename)
    return result


def ssl_context(cafile: Path | None) -> ssl.SSLContext:
    context = ssl.create_default_context()
    if cafile is not None:
        context.load_verify_locations(cafile=str(cafile))
    return context


def open_url(url: str, context: ssl.SSLContext, headers: dict[str, str] | None = None, method: str = "GET"):
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})}, method=method
    )
    return urllib.request.urlopen(request, timeout=120, context=context)


class RangeReader(io.RawIOBase):
    def __init__(self, url: str, context: ssl.SSLContext) -> None:
        self.url = url
        self.context = context
        self.position = 0
        with open_url(url, context, method="HEAD") as response:
            if response.headers.get("Accept-Ranges", "").casefold() != "bytes":
                raise ValueError(f"Server does not accept byte ranges: {url}")
            self.size = int(response.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size:
            return 0
        end = min(self.size, self.position + len(buffer)) - 1
        with open_url(self.url, self.context, {"Range": f"bytes={self.position}-{end}"}) as response:
            if response.status != 206:
                raise ValueError(f"Expected partial content, got HTTP {response.status}")
            block = response.read()
        buffer[: len(block)] = block
        self.position += len(block)
        return len(block)


def keep_row(row: dict[str, str], filters: dict[str, list[str]]) -> bool:
    return all(row.get(column) in values for column, values in filters.items())


def write_member_extract(source: dict[str, object], stream, context: ssl.SSLContext) -> None:
    filters = {str(k): [str(v) for v in values] for k, values in dict(source.get("filters", {})).items()}
    member_digest = hashlib.sha256()
    archive = zipfile.ZipFile(io.BufferedReader(RangeReader(str(source["url"]), context), 1 << 20))
    with archive.open(str(source["member"])) as member:
        header_line = member.readline()
        member_digest.update(header_line)
        header = next(csv.reader([header_line.decode("utf-8-sig")], delimiter=";"))
        missing = set(filters) - set(header)
        if missing:
            raise ValueError(f"Filter columns absent in {source['member']}: {sorted(missing)}")
        stream.write(header_line)
        for line in member:
            member_digest.update(line)
            values = next(csv.reader([line.decode("utf-8")], delimiter=";"))
            if keep_row(dict(zip(header, values)), filters):
                stream.write(line)
    actual = member_digest.hexdigest()
    if actual != source["member_sha256"]:
        raise ValueError(f"Member SHA-256 mismatch for {source['member']}: {actual}")


def write_whole_file(source: dict[str, object], stream, context: ssl.SSLContext) -> None:
    limit = int(source.get("max_bytes", MAX_SOURCE_BYTES))
    with open_url(str(source["url"]), context) as response:
        total = 0
        while True:
            block = response.read(64 * 1024)
            if not block:
                break
            total += len(block)
            if total > limit:
                raise ValueError(f"Source exceeds {limit} bytes: {source['file']}")
            stream.write(block)


def download_one(source: dict[str, object], output_dir: Path, context: ssl.SSLContext) -> str:
    destination = output_dir / str(source["file"])
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
            mode="wb", prefix=f".{destination.name}.", suffix=".part", dir=output_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            if "member" in source:
                write_member_extract(source, temporary, context)
            else:
                write_whole_file(source, temporary, context)
            temporary.flush()
            os.fsync(temporary.fileno())
        actual = sha256_file(temporary_path)
        if actual != source["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {source['file']}: {actual} != {source['sha256']}")
        os.replace(temporary_path, destination)
        temporary_path = None
        return f"downloaded {destination.name} ({destination.stat().st_size} bytes)"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "reports/external-municipal/sources.json")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts/sources/municipal")
    parser.add_argument(
        "--cafile",
        type=Path,
        help="PEM bundle with Russian Trusted Root/Sub CA when the system store lacks it",
    )
    args = parser.parse_args()
    context = ssl_context(args.cafile)
    for source in validated_sources(args.manifest):
        print(download_one(source, args.output_dir, context))


if __name__ == "__main__":
    main()
