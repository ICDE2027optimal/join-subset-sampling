#!/usr/bin/env python3
"""Prepare a selected controlled group with the paper's generator."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

import controlled_generator as generator


EXPERIMENTS = Path(__file__).resolve().parents[1]
CONFIG = EXPERIMENTS / "configs" / "controlled_groups.json"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def prepare(group: str, output_root: Path) -> dict:
    config = load_config()
    selection = config["groups"][group]
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Choose a new preparation directory: {output_root}")
    prepared_root = output_root / "prepared"
    prepared_root.mkdir(parents=True)
    started = time.perf_counter()
    definitions = generator.case_definitions()
    if group == "smoke":
        cases = [{
            "case_index": 0, "directory": "smoke_dense_k4",
            "family": "dense-uniform-scope", "name": "smoke-dense-k4",
            "key_count": 4, "dense_p": 1 / 16,
            "requested_labels": "1over16", "schedule_class": "dense",
        }]
    else:
        cases = [definitions[index] for index in selection["case_indices"]]
    tiny = generator.tiny_oracle_checks()
    records = []
    for case in cases:
        result = generator.build_case(case, prepared_root)
        if group == "smoke":
            result["queries"] = selection["queries"]
            result["warmup"] = selection["warmup"]
            case_file = prepared_root / case["directory"] / "cases.tsv"
            with case_file.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream, delimiter="\t")
                fields = list(reader.fieldnames or ())
                rows = list(reader)
            rows[0]["queries"] = str(result["queries"])
            rows[0]["warmup"] = str(result["warmup"])
            with case_file.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fields, delimiter="\t", lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        result["paper_case"] = group != "smoke"
        result["repeats"] = selection["repeats"]
        records.append(result)
        print(json.dumps({"status": "prepared", "case_index": case["case_index"],
                          "name": case["name"]}), flush=True)
    files = [generator.hashed_file(path, output_root)
             for path in sorted(prepared_root.rglob("*")) if path.is_file()]
    manifest = {
        "schema": "minimal-controlled-preparation-v1", "group": group,
        "paper_schedule": group != "smoke", "cases": records,
        "case_order": [case["directory"] for case in records],
        "prepared_files": files, "tiny_oracles": tiny,
        "root_near_tie_relative_delta": "1e-12",
        "normalization": "relation0 multiplied by min(1,100/mu0) for size and degree families",
        "sources": [generator.hashed_file(Path(generator.__file__)),
                    generator.hashed_file(Path(__file__)), generator.hashed_file(CONFIG)],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(),
                        "longdouble_bits": np.finfo(np.longdouble).bits,
                        "exp2_backend": "libm.exp2" if generator._C_EXP2 else "python_power"},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output_root / "preparation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=load_config()["groups"], default="smoke")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    if args.self_test_only:
        print(json.dumps({"status": "pass", "tiny_oracles": generator.tiny_oracle_checks()}))
        return 0
    if args.output_root is None:
        parser.error("--output-root is required for preparation")
    result = prepare(args.group, args.output_root)
    print(json.dumps({"status": "pass", "cases": len(result["cases"]),
                      "manifest": str(args.output_root / "preparation_manifest.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
