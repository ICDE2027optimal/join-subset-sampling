#!/usr/bin/env python3
"""Prepare and validate the original or added STATS and JOB queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import acquire_data


HERE = Path(__file__).resolve().parent
CONFIGS = HERE.parent / "configs"
ARTIFACT = HERE.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("stats", "job", "newbranch5", "job17a"))
    parser.add_argument("--raw-dir", type=Path, help="Verified sources, defaults to data/stats or data/job by source family")
    parser.add_argument("--output", type=Path, required=True, help="A new preparation output directory")
    parser.add_argument("--duckdb-python", type=Path, help="JOB oracle interpreter, defaults to this Python")
    args = parser.parse_args()
    family = "stats" if args.dataset in ("stats", "newbranch5") else "job"
    raw_dir = (args.raw_dir or ARTIFACT / "data" / family).resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    sources = CONFIGS / f"{args.dataset}_sources.json"
    queries = CONFIGS / f"{args.dataset}_queries.json"
    config = json.loads(sources.read_text(encoding="utf-8"))
    acquire_data.verify_sources(args.dataset, raw_dir, config)
    output.mkdir(parents=True)
    if args.dataset == "job17a":
        sql = CONFIGS / "job17a.sql"
        protocol = json.loads((CONFIGS / "job17a_protocol.json").read_text(encoding="utf-8"))
        if acquire_data.sha256_file(sql) != protocol["sql_sha256"]:
            raise ValueError("JOB17a SQL identity disagrees with its protocol")
        shutil.copyfile(sql, output / "job17a.sql")
        oracle_python = str((args.duckdb_python or Path(sys.executable)).resolve())
        subprocess.run([sys.executable, str(HERE / "prepare_job17a.py"), str(output),
                        "--raw-dir", str(raw_dir), "--stage", "parse"], check=True)
        subprocess.run([oracle_python, str(HERE / "oracle_job17a.py"), str(output),
                        "--raw-dir", str(raw_dir)], check=True)
        subprocess.run([sys.executable, str(HERE / "prepare_job17a.py"), str(output),
                        "--raw-dir", str(raw_dir), "--stage", "prepare"], check=True)
        subprocess.run([oracle_python, str(HERE / "verify_job17a_serialized.py"), str(output)], check=True)
        print(json.dumps({"status": "pass", "dataset": args.dataset,
                          "prepared": str(output / "prepared"),
                          "validation": str(output / "validation.json")}))
        return 0
    command = [sys.executable, str(HERE / f"prepare_{family}.py"),
               "--raw-dir", str(raw_dir), "--source-manifest", str(sources),
               "--query-manifest", str(queries), "--output-dir", str(output / "prepared"),
               "--inventory-out", str(output / "source_inventory.json"),
               "--validation-out", str(output / "validation.json"),
               "--duckdb-temp-dir", str(output / "oracle")]
    if args.dataset == "newbranch5":
        command += ["--duckdb-helper", str(HERE / "oracle_newbranch5.py")]
    if args.dataset == "job":
        command += ["--stage", "prepare", "--duckdb-python",
                    str((args.duckdb_python or Path(sys.executable)).resolve()),
                    "--manifest-out", str(output / "preparation_manifest.json")]
    subprocess.run(command, check=True)
    if args.dataset == "job":
        verify = list(command)
        verify[verify.index("--stage") + 1] = "verify-moments"
        verify += ["--moment-validation-out", str(output / "serialized_moment_validation.json")]
        subprocess.run(verify, check=True)
    print(json.dumps({"status": "pass", "dataset": args.dataset,
                      "prepared": str(output / "prepared"),
                      "validation": str(output / "validation.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
