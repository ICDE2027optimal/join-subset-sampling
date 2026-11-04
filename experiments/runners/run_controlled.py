#!/usr/bin/env python3
"""Prepare, plan and optionally execute a bounded controlled experiment group."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time


ARTIFACT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ARTIFACT / "experiments" / "preparation"))
import prepare_controlled
import controlled_generator
from controlled_generator import hashed_file, sha256_file


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def driver_command(driver: Path, case: dict, method: str, seed: int, label: str) -> list[str]:
    return [str(driver), "--query-dir", case["query_directory"],
            "--weight-set", case["weight_set"], "--root", str(case["chosen_root"]),
            "--method", method, "--seed", str(seed),
            "--warmup", str(case["warmup"]), "--queries", str(case["queries"]),
            "--expected-j", str(case["join_size"]), "--expected-mu", case["actual_mu"],
            "--expected-maxp", case["max_p"], "--expected-sump2", case["sum_p2"],
            "--case-label", label]


def build_plan(args: argparse.Namespace, preparation: dict, config: dict) -> dict:
    processes = []
    exclusions = []
    cases = preparation["cases"]
    primary = config["primary"]
    paired = config["paired"]

    def add(case: dict, method: str, repeat: int, batch: str, seed_index: int) -> None:
        protocol = primary if batch == "primary" else paired
        seed = int(protocol["seed_base_hex"], 16) + seed_index * 0x10000 + repeat * 0x100
        seed += protocol["method_index"][method]
        sequence = len(processes)
        label = f"{sequence:04d}_c{case['case_index']:03d}_rep{repeat}_{method.replace('-', '_')}"
        driver = args.explicit_driver if method == "materialized-explicit" else args.driver
        processes.append({
            "sequence": sequence, "process_id": label, "case_index": case["case_index"],
            "repeat": repeat, "method": method, "batch": batch,
            "scope": "paired-materialized-explicit" if batch == "paired" else (
                "scope" if method in ("rank", "scan") else "primary"),
            "seed_index": seed_index, "seed": seed, "seed_hex": f"0x{seed:016x}",
            "cpu": args.cpu, "memory_bytes": args.memory_bytes,
            "timeout_seconds": args.timeout_seconds,
            "relative_directory": (Path("processes") / label).as_posix(),
            "command": driver_command(driver, case, method, seed, label),
        })

    if args.batch in ("primary", "all"):
        for case in cases:
            for repeat in range(case["repeats"]):
                methods = primary["methods"]
                order = methods[repeat % len(methods):] + methods[:repeat % len(methods)]
                for method in order:
                    add(case, method, repeat, "primary", case["case_index"])
            applicable = [method for method in ("rank", "scan") if case[f"{method}_eligible"]]
            for repeat in range(case["repeats"]):
                for method in applicable if repeat % 2 == 0 else list(reversed(applicable)):
                    add(case, method, repeat, "primary", case["case_index"])
            for method in ("rank", "scan"):
                if method not in applicable:
                    exclusions.append({"case_index": case["case_index"], "method": method,
                                       "reason": "requires dense constant weights and J<=2^53"
                                       if method == "rank" else "requires J<=1000000"})
    if args.batch in ("paired", "all"):
        for case in cases:
            if case["join_size"] > paired["join_size_cap"]:
                exclusions.append({"case_index": case["case_index"],
                                   "method": "materialized-explicit", "reason": "requires J<=25000000"})
                continue
            if preparation["paper_schedule"]:
                seed_index = paired["supplement_controlled_offset"] + paired["controlled_source_indices"].index(case["case_index"])
            else:
                seed_index = 0
            for repeat in range(case["repeats"]):
                order = paired["methods"] if repeat % 2 == 0 else list(reversed(paired["methods"]))
                for method in order:
                    add(case, method, repeat, "paired", seed_index)
    if not processes:
        raise ValueError("The selected group has no eligible processes for this batch")
    return {
        "schema": "minimal-controlled-run-manifest-v1", "created_at_utc": utc_now(),
        "group": args.group, "batch": args.batch, "paper_schedule": preparation["paper_schedule"],
        "paths": {"output": str(args.output), "run_dir": str(args.output / "run"),
                  "preparation_manifest": str(args.output / "preparation" / "preparation_manifest.json")},
        "protocol": {"repeats": config["groups"][args.group]["repeats"],
                     "primary": primary, "paired": paired,
                     "process_timeout_seconds": args.timeout_seconds,
                     "address_space_bytes": args.memory_bytes, "cpu": args.cpu,
                     "timeout_policy": "censor and continue", "unexpected_exit_policy": "record and stop"},
        "hashes": {"files": [hashed_file(path) for path in
                              {Path(__file__), Path(prepare_controlled.__file__),
                               Path(controlled_generator.__file__),
                               prepare_controlled.CONFIG,
                               *[Path(p["command"][0]) for p in processes]}],
                   "prepared_files": preparation["prepared_files"]},
        "cases": cases, "processes": processes, "exclusions": exclusions,
    }


def child_limits(cpu: int, memory_bytes: int) -> None:
    import resource
    os.setsid()
    os.sched_setaffinity(0, {cpu})
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))


def kill_group(child: subprocess.Popen) -> None:
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)


def parse_time(path: Path) -> dict:
    result = {"maximum_rss_kib": None, "file": str(path)}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            prefix = "\tMaximum resident set size (kbytes): "
            if line.startswith(prefix):
                result["maximum_rss_kib"] = int(line[len(prefix):])
    return result


def validate_record(record: dict, process: dict, case: dict) -> str | None:
    checks = {"schema": "join-formal-driver-v1", "mode": "measurement",
              "method": process["method"], "seed": process["seed"],
              "root": case["chosen_root"], "prepared_expected_j": case["join_size"],
              "warmup_queries": case["warmup"], "measured_queries": case["queries"]}
    for field, expected in checks.items():
        if record.get(field) != expected:
            return f"Driver record field {field} disagrees with the plan"
    for field, expected in (("prepared_expected_mu", "actual_mu"),
                            ("prepared_expected_maxp", "max_p"),
                            ("prepared_expected_sump2", "sum_p2")):
        try:
            matches = math.isclose(float(record[field]), float(case[expected]), rel_tol=1e-9, abs_tol=1e-15)
        except (KeyError, TypeError, ValueError):
            matches = False
        if not matches:
            return f"Driver record field {field} disagrees with the preparation"
    return None


def execute_one(process: dict, case: dict, run_dir: Path, time_binary: Path) -> dict:
    process_dir = run_dir / process["relative_directory"]
    process_dir.mkdir(parents=True)
    stdout_path = process_dir / "stdout.jsonl"
    stderr_path = process_dir / "stderr.txt"
    time_path = process_dir / "gnu_time.txt"
    launch = [str(time_binary), "-v", "-o", str(time_path), *process["command"]]
    start = utc_now()
    monotonic = time.monotonic()
    timed_out = False
    exit_code = None
    launch_error = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            child = subprocess.Popen(
                launch, stdout=stdout, stderr=stderr,
                preexec_fn=lambda: child_limits(process["cpu"], process["memory_bytes"]),
                env={**os.environ, "LC_ALL": "C", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            try:
                exit_code = child.wait(timeout=process["timeout_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_group(child)
                exit_code = child.returncode
        except (OSError, subprocess.SubprocessError) as error:
            launch_error = str(error)
    elapsed = time.monotonic() - monotonic
    driver_record = None
    error = launch_error
    status = "launch_error" if launch_error else "completed"
    if timed_out:
        status, error = "timeout", "Whole-process timeout reached"
    elif exit_code != 0 and not launch_error:
        status, error = "unexpected_exit", f"Process exited with code {exit_code}"
    elif status == "completed":
        try:
            lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != 1:
                raise ValueError(f"Expected one JSON line, received {len(lines)}")
            parsed = json.loads(lines[0])
            error = validate_record(parsed, process, case)
            if error:
                status = "metadata_mismatch"
            else:
                driver_record = parsed
        except (ValueError, TypeError) as problem:
            status, error = "metadata_mismatch", str(problem)
    return {"schema": "minimal-controlled-process-record-v1", **process,
            "status": status, "censored": status == "timeout", "error": error,
            "exit_code": exit_code, "timed_out": timed_out,
            "start_timestamp_utc": start, "end_timestamp_utc": utc_now(),
            "wall_seconds_runner": elapsed, "launch_command": launch,
            "stdout_file": str(stdout_path), "stderr_file": str(stderr_path),
            "gnu_time": parse_time(time_path), "driver_record": driver_record}


def execute(plan: dict, time_binary: Path) -> int:
    run_dir = Path(plan["paths"]["run_dir"])
    run_dir.mkdir(exist_ok=False)
    for expected in plan["hashes"]["files"]:
        if sha256_file(Path(expected["path"])) != expected["sha256"]:
            raise RuntimeError(f"Source or executable changed: {expected['path']}")
    preparation_root = Path(plan["paths"]["preparation_manifest"]).parent
    for expected in plan["hashes"]["prepared_files"]:
        if sha256_file(preparation_root / expected["path"]) != expected["sha256"]:
            raise RuntimeError(f"Prepared input changed: {expected['path']}")
    write_json(run_dir / "manifest.json", plan)
    write_json(run_dir / "execution_environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "time_binary": hashed_file(time_binary), "started_at_utc": utc_now()})
    counts = {"completed": 0, "timeout": 0}
    stop = None
    cases = {case["case_index"]: case for case in plan["cases"]}
    with (run_dir / "progress.jsonl").open("w", encoding="utf-8") as progress:
        for process in plan["processes"]:
            record = execute_one(process, cases[process["case_index"]], run_dir, time_binary)
            write_json(run_dir / process["relative_directory"] / "record.json", record)
            progress.write(json.dumps(record, sort_keys=True) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
            print(json.dumps({"process": process["process_id"], "status": record["status"]}), flush=True)
            if record["status"] in counts:
                counts[record["status"]] += 1
            else:
                stop = record["process_id"]
                break
    summary = {"schema": "minimal-controlled-run-summary-v1",
               "status": "complete" if stop is None else "aborted",
               "planned_processes": len(plan["processes"]),
               "completed_processes": counts["completed"], "timeout_processes": counts["timeout"],
               "stopped_process": stop, "finished_at_utc": utc_now()}
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if stop is None else 2


def main() -> int:
    config = prepare_controlled.load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=config["groups"], default="smoke")
    parser.add_argument("--batch", choices=("primary", "paired", "all"), default="primary",
                        help="The paired batch reruns Candidate with Materialized using the supplement seeds")
    parser.add_argument("--output", type=Path, default=ARTIFACT / "runs" / "controlled-smoke")
    parser.add_argument("--driver", type=Path, default=ARTIFACT / "build" / "formal_driver")
    parser.add_argument("--explicit-driver", type=Path, default=ARTIFACT / "build" / "materialized_explicit_driver")
    parser.add_argument("--execute", action="store_true", help="Execute after preparation and planning")
    parser.add_argument("--cpu", type=int, help="Logical CPU, defaults to the first allowed CPU")
    parser.add_argument("--timeout-seconds", type=float, default=config["limits"]["process_timeout_seconds"])
    parser.add_argument("--memory-bytes", type=int, default=config["limits"]["address_space_bytes"])
    parser.add_argument("--time-binary", type=Path, default=Path("/usr/bin/time"))
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or args.memory_bytes <= 0:
        parser.error("Resource limits must be positive")
    args.output = args.output.resolve()
    args.driver = args.driver.resolve()
    args.explicit_driver = args.explicit_driver.resolve()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if args.execute:
        if not sys.platform.startswith("linux"):
            parser.error("Execution uses Linux CPU affinity, RLIMIT_AS and GNU time. Preparation and planning also run on other systems.")
        if not args.time_binary.is_file():
            parser.error("GNU time is required. Set --time-binary to its path")
        allowed = sorted(os.sched_getaffinity(0))
        args.cpu = allowed[0] if args.cpu is None else args.cpu
        if args.cpu not in allowed:
            parser.error(f"CPU must be in the allowed affinity set: {allowed}")
    elif args.cpu is None:
        args.cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    required = [args.driver] if args.batch == "primary" else [args.driver, args.explicit_driver]
    for driver in required:
        if not driver.is_file():
            parser.error(f"Build the driver or pass its path: {driver}")
    args.output.mkdir(parents=True)
    preparation = prepare_controlled.prepare(args.group, args.output / "preparation")
    plan = build_plan(args, preparation, config)
    write_json(args.output / "manifest.json", plan)
    print(json.dumps({"status": "planned", "manifest": str(args.output / "manifest.json"),
                      "cases": len(plan["cases"]), "processes": len(plan["processes"])}), flush=True)
    return execute(plan, args.time_binary.resolve()) if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
