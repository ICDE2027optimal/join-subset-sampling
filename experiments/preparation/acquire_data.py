#!/usr/bin/env python3
"""Acquire the pinned STATS or JOB source bytes and verify their hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ARTIFACT = CONFIGS.parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_bytes: int, expected_sha256: str) -> dict:
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"Source size mismatch for {path.name}: {size} != {expected_bytes}")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise ValueError(f"Source SHA256 mismatch for {path.name}: {digest}")
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest}


def download(url: str, path: Path, expected_bytes: int, expected_sha256: str) -> dict:
    if path.exists():
        return verify(path, expected_bytes, expected_sha256)
    temporary = path.with_suffix(path.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "join-sampling-minimal-artifact/1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as output:
        shutil.copyfileobj(response, output, length=1 << 20)
    record = verify(temporary, expected_bytes, expected_sha256)
    temporary.replace(path)
    record["path"] = str(path.resolve())
    return record


def source_items(dataset: str, config: dict) -> list[dict]:
    return config["files"] if dataset in ("stats", "newbranch5") else config["tables"]


def verify_sources(dataset: str, root: Path, config: dict) -> list[dict]:
    return [verify(root / (item["name"] if dataset in ("stats", "newbranch5") else item["member"]),
                   item["expected_bytes"], item["expected_sha256"])
            for item in source_items(dataset, config)]


def acquire_stats(output: Path, config: dict) -> list[dict]:
    records = []
    for item in config["files"]:
        url = config["raw_base_url"] + item["repo_path"]
        record = download(url, output / item["name"], item["expected_bytes"], item["expected_sha256"])
        record["url"] = url
        records.append(record)
        print(json.dumps({"status": "verified", "file": item["name"]}), flush=True)
    return records


def acquire_job(output: Path, config: dict, archive: Path | None) -> tuple[list[dict], dict]:
    info = config["archive"]
    if archive is None:
        archive = output / "imdb.tgz"
        archive_record = download(info["url"], archive, info["bytes"], info["sha256"])
    else:
        archive_record = verify(archive, info["bytes"], info["sha256"])
    wanted = {item["member"] for item in config["tables"]} | {info["schema_member"]}
    seen = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            basename = Path(member.name).name
            if basename not in wanted:
                continue
            if basename in seen or not member.isfile():
                raise ValueError(f"Unexpected archive member for {basename}")
            seen.add(basename)
            destination = output / basename
            if destination.exists():
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError(f"Archive member has no file content: {member.name}")
            temporary = destination.with_suffix(destination.suffix + ".extract")
            with source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, length=1 << 20)
            temporary.replace(destination)
    if seen != wanted:
        raise ValueError(f"Archive is missing selected members: {sorted(wanted - seen)}")
    return verify_sources("job", output, config), archive_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("stats", "job", "newbranch5", "job17a"))
    parser.add_argument("--output", type=Path, help="Source directory, defaults to data/stats or data/job by source family")
    parser.add_argument("--archive", type=Path, help="An existing JOB imdb.tgz archive")
    parser.add_argument("--check-only", action="store_true", help="Verify an existing source directory")
    args = parser.parse_args()
    if args.archive is not None and args.dataset not in ("job", "job17a"):
        parser.error("--archive applies to JOB")
    config_path = CONFIGS / f"{args.dataset}_sources.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    family = "stats" if args.dataset in ("stats", "newbranch5") else "job"
    output = (args.output or ARTIFACT / "data" / family).resolve()
    archive_record = None
    if args.check_only:
        records = verify_sources(args.dataset, output, config)
    else:
        output.mkdir(parents=True, exist_ok=True)
        if family == "stats":
            records = acquire_stats(output, config)
        else:
            records, archive_record = acquire_job(output, config, args.archive)
    payload = {"schema": "minimal-dataset-acquisition-v1", "dataset": args.dataset,
               "status": "pass", "source_config_sha256": sha256_file(config_path),
               "files": records, "archive": archive_record}
    if not args.check_only:
        (output / "acquisition_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "dataset": args.dataset, "files": len(records), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
