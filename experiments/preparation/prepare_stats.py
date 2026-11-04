#!/usr/bin/env python3
"""Prepare the fixed STATS snapshot for the formal join-sampling driver.

The script has three deliberately separate checks:
  * Python's csv module parses, filters, and writes the undirected TSV format.
  * an integer tree DP counts the prepared join without enumerating it.
  * DuckDB parses the original CSV files and executes the fixed SQL directly.

All timestamps are timezone-free source values represented as integer seconds
from 1970-01-01 in the proleptic Gregorian calendar.  Empty CSV fields and
``\\N`` are SQL NULL.  A row with NULL in any incident equality-join key can
never occur in the full join and is excluded from prepared relation files.
"""

from __future__ import annotations

import argparse
import csv
import functools
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


FORMAT = "stats-undirected-v1"
PREPARATION_VERSION = "prepare-stats-v1"
ROOT_NEAR_TIE_RELATIVE_DELTA = np.longdouble("1e-12")
NULL_MARKERS = {"", "\\N"}
EPOCH = datetime(1970, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--query-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--inventory-out", required=True, type=Path)
    parser.add_argument("--validation-out", required=True, type=Path)
    parser.add_argument("--duckdb-temp-dir", required=True, type=Path)
    parser.add_argument("--source-download-seconds", type=float)
    parser.add_argument("--duckdb-helper", type=Path)
    parser.add_argument("--skip-duckdb", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def timestamp_seconds(text: str) -> int:
    value = datetime.fromisoformat(text)
    if value.tzinfo is not None:
        raise ValueError(f"timestamp unexpectedly has timezone: {text!r}")
    if value.microsecond != 0:
        raise ValueError(f"fixed STATS snapshot is expected to use whole-second timestamps: {text!r}")
    delta = value - EPOCH
    return delta.days * 86400 + delta.seconds


def parse_scalar(text: str, kind: str) -> Optional[int]:
    if text in NULL_MARKERS:
        return None
    if kind == "int":
        return int(text)
    if kind == "timestamp":
        return timestamp_seconds(text)
    raise ValueError(f"unknown source type {kind!r}")


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(block.count(b"\n") for block in iter(lambda: handle.read(1 << 20), b""))


def read_table(path: Path, columns: Mapping[str, str]) -> Tuple[List[Dict[str, Optional[int]]], Dict[str, Any]]:
    rows: List[Dict[str, Optional[int]]] = []
    null_counts = {column: 0 for column in columns}
    parse_counts = {column: 0 for column in columns}
    seen_ids = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(columns):
            raise ValueError(
                f"{path.name}: header {reader.fieldnames!r} != manifest {list(columns)!r}"
            )
        for physical_row, raw in enumerate(reader, start=2):
            parsed: Dict[str, Optional[int]] = {}
            for column, kind in columns.items():
                try:
                    parsed[column] = parse_scalar(raw[column], kind)
                    parse_counts[column] += 1
                except Exception as exc:
                    raise ValueError(
                        f"{path.name}:{physical_row} cannot parse {column}={raw[column]!r} as {kind}"
                    ) from exc
                if parsed[column] is None:
                    null_counts[column] += 1
            tuple_id = parsed.get("Id")
            if tuple_id is None:
                raise ValueError(f"{path.name}:{physical_row} has NULL Id")
            if tuple_id in seen_ids:
                raise ValueError(f"{path.name}:{physical_row} duplicates Id {tuple_id}")
            seen_ids.add(tuple_id)
            rows.append(parsed)
    return rows, {
        "row_count": len(rows),
        "null_counts": null_counts,
        "parse_counts": parse_counts,
        "unique_id_count": len(seen_ids),
    }


def build_source_inventory(
    raw_dir: Path, source_manifest: Mapping[str, Any]
) -> Tuple[Dict[str, List[Dict[str, Optional[int]]]], Dict[str, Any]]:
    tables: Dict[str, List[Dict[str, Optional[int]]]] = {}
    files_out = []
    base_url = source_manifest["raw_base_url"]
    for item in source_manifest["files"]:
        path = raw_dir / item["name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        byte_count = path.stat().st_size
        expected = item.get("expected_bytes")
        if expected is not None and byte_count != expected:
            raise ValueError(f"{path.name}: bytes {byte_count} != expected {expected}")
        record: Dict[str, Any] = {
            "name": item["name"],
            "repo_path": item["repo_path"],
            "url": base_url + item["repo_path"],
            "role": item["role"],
            "bytes": byte_count,
            "sha256": sha256_file(path),
        }
        if item["role"] == "table":
            rows, parsed = read_table(path, item["columns"])
            tables[item["table"]] = rows
            record.update(parsed)
            record["table"] = item["table"]
            record["columns"] = item["columns"]
        else:
            record["line_count"] = count_lines(path)
        files_out.append(record)
    timestamp_valid_count = sum(
        record["row_count"] - record["null_counts"][column]
        for record in files_out if record["role"] == "table"
        for column, kind in record["columns"].items() if kind == "timestamp"
    )
    return tables, {
        "format": "stats-source-inventory-v1",
        "preparation_version": PREPARATION_VERSION,
        "repository": source_manifest["repository"],
        "commit": source_manifest["commit"],
        "timestamp_precision_audit": {
            "valid_timestamp_values": timestamp_valid_count,
            "fractional_values": 0,
            "whole_second_domain_asserted_by_parser": True
        },
        "files": files_out,
    }


def nearest_rank_q99(values: Sequence[int]) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(max(value, 0) for value in values)
    rank = math.ceil(0.99 * len(ordered)) - 1
    return ordered[rank]


def discover_feature_specs(query_manifest: Mapping[str, Any]) -> Dict[Tuple[str, str, str], None]:
    specs: Dict[Tuple[str, str, str], None] = {}
    for query in query_manifest["queries"]:
        for relation in query["relations"]:
            for field in ("single_anchor_feature", "product_real_feature"):
                feature = relation.get(field)
                if feature:
                    specs[(relation["table"], feature["column"], feature["kind"])] = None
    return specs


def compute_feature_summaries(
    tables: Mapping[str, Sequence[Mapping[str, Optional[int]]]],
    query_manifest: Mapping[str, Any],
) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for table, column, kind in discover_feature_specs(query_manifest):
        valid = [row[column] for row in tables[table] if row[column] is not None]
        key = f"{table}.{column}.{kind}"
        if kind == "numeric_importance":
            summaries[key] = {
                "kind": kind,
                "table": table,
                "column": column,
                "base_row_count": len(tables[table]),
                "valid_count": len(valid),
                "missing_count": len(tables[table]) - len(valid),
                "q99_nearest_rank": nearest_rank_q99([int(value) for value in valid]),
            }
        elif kind == "date_recency":
            summaries[key] = {
                "kind": kind,
                "table": table,
                "column": column,
                "base_row_count": len(tables[table]),
                "valid_count": len(valid),
                "missing_count": len(tables[table]) - len(valid),
                "min_seconds": min(valid) if valid else None,
                "max_seconds": max(valid) if valid else None,
                "calendar_rule": "timezone-free proleptic-Gregorian seconds since 1970-01-01",
            }
        else:
            raise ValueError(f"unknown feature kind {kind}")
    return summaries


def feature_weight(
    row: Mapping[str, Optional[int]],
    table: str,
    feature: Mapping[str, str],
    summaries: Mapping[str, Mapping[str, Any]],
    epsilon: float,
) -> float:
    column = feature["column"]
    kind = feature["kind"]
    value = row[column]
    if value is None:
        return epsilon
    summary = summaries[f"{table}.{column}.{kind}"]
    if kind == "date_recency":
        low = summary["min_seconds"]
        high = summary["max_seconds"]
        if low is None or high is None:
            return epsilon
        if low == high:
            return 1.0
        return epsilon + (1.0 - epsilon) * (float(value - low) / float(high - low))
    if kind == "numeric_importance":
        x = max(int(value), 0)
        q99 = summary["q99_nearest_rank"]
        if q99 is None:
            return epsilon
        if q99 == 0:
            return 1.0 if x > 0 else epsilon
        ratio = math.log1p(x) / math.log1p(q99)
        return epsilon + (1.0 - epsilon) * min(1.0, ratio)
    raise ValueError(f"unknown feature kind {kind}")


OPS = {
    "=": lambda lhs, rhs: lhs == rhs,
    ">=": lambda lhs, rhs: lhs >= rhs,
    "<=": lambda lhs, rhs: lhs <= rhs,
    ">": lambda lhs, rhs: lhs > rhs,
    "<": lambda lhs, rhs: lhs < rhs,
}


def filter_rhs(value: Any, kind: str) -> int:
    if kind == "timestamp":
        return timestamp_seconds(str(value))
    if kind == "int":
        return int(value)
    raise ValueError(kind)


def passes_filters(
    row: Mapping[str, Optional[int]], filters: Sequence[Mapping[str, Any]], column_types: Mapping[str, str]
) -> bool:
    for condition in filters:
        lhs = row[condition["column"]]
        if lhs is None:
            return False
        rhs = filter_rhs(condition["value"], column_types[condition["column"]])
        if not OPS[condition["op"]](lhs, rhs):
            return False
    return True


def incident_edges(query: Mapping[str, Any]) -> Dict[int, List[Mapping[str, Any]]]:
    result = {relation["id"]: [] for relation in query["relations"]}
    for edge in query["edges"]:
        result[edge["left"]].append(edge)
        result[edge["right"]].append(edge)
    for edges in result.values():
        edges.sort(key=lambda edge: edge["id"])
    return result


def endpoint_columns(edge: Mapping[str, Any], relation_id: int) -> Sequence[str]:
    if edge["left"] == relation_id:
        return edge["left_columns"]
    if edge["right"] == relation_id:
        return edge["right_columns"]
    raise ValueError(f"relation {relation_id} is not incident to edge {edge['id']}")


def endpoint_other(edge: Mapping[str, Any], relation_id: int) -> int:
    if edge["left"] == relation_id:
        return edge["right"]
    if edge["right"] == relation_id:
        return edge["left"]
    raise ValueError


def raw_key(row: Mapping[str, Optional[int]], columns: Sequence[str]) -> Optional[Tuple[int, ...]]:
    values = tuple(row[column] for column in columns)
    if any(value is None for value in values):
        return None
    return tuple(int(value) for value in values if value is not None)


def prepare_query_rows(
    query: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Optional[int]]]],
    table_types: Mapping[str, Mapping[str, str]],
) -> Tuple[List[List[Dict[str, Any]]], List[Dict[str, Any]], Dict[int, int]]:
    incident = incident_edges(query)
    rows_by_relation: List[List[Dict[str, Any]]] = [[] for _ in query["relations"]]
    relation_stats: List[Dict[str, Any]] = []
    for relation in query["relations"]:
        rid = relation["id"]
        if rid != len(relation_stats):
            raise ValueError(f"{query['id']}: relation IDs must be dense and ordered")
        filtered = []
        for source_row in tables[relation["table"]]:
            if passes_filters(source_row, relation["filters"], table_types[relation["table"]]):
                filtered.append(source_row)
        eligible = []
        null_join_rows = 0
        for source_row in filtered:
            if any(raw_key(source_row, endpoint_columns(edge, rid)) is None for edge in incident[rid]):
                null_join_rows += 1
                continue
            eligible.append({"source": source_row, "tuple_id": int(source_row["Id"]), "edge_keys": {}})
        rows_by_relation[rid] = eligible
        relation_stats.append({
            "relation_id": rid,
            "alias": relation["alias"],
            "table": relation["table"],
            "base_rows": len(tables[relation["table"]]),
            "filter_pass_rows": len(filtered),
            "null_join_rows_dropped": null_join_rows,
            "prepared_rows": len(eligible),
        })

    key_cardinalities: Dict[int, int] = {}
    for edge in query["edges"]:
        left = edge["left"]
        right = edge["right"]
        values = set()
        for row in rows_by_relation[left]:
            values.add(raw_key(row["source"], edge["left_columns"]))
        for row in rows_by_relation[right]:
            values.add(raw_key(row["source"], edge["right_columns"]))
        if None in values:
            raise AssertionError("NULL key survived eligibility filtering")
        ordered = sorted(values)
        interner = {value: index + 1 for index, value in enumerate(ordered)}
        key_cardinalities[edge["id"]] = len(interner)
        for row in rows_by_relation[left]:
            row["edge_keys"][edge["id"]] = interner[raw_key(row["source"], edge["left_columns"])]
        for row in rows_by_relation[right]:
            row["edge_keys"][edge["id"]] = interner[raw_key(row["source"], edge["right_columns"])]
    return rows_by_relation, relation_stats, key_cardinalities


def tree_neighbors(query: Mapping[str, Any]) -> Dict[int, List[Tuple[int, int]]]:
    neighbors = {relation["id"]: [] for relation in query["relations"]}
    for edge in query["edges"]:
        neighbors[edge["left"]].append((edge["right"], edge["id"]))
        neighbors[edge["right"]].append((edge["left"], edge["id"]))
    if len(query["edges"]) != len(query["relations"]) - 1:
        raise ValueError(f"{query['id']}: relation graph is not a tree by edge count")
    seen = set()
    stack = [query["relations"][0]["id"]]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(other for other, _ in neighbors[node] if other not in seen)
    if len(seen) != len(query["relations"]):
        raise ValueError(f"{query['id']}: relation graph is disconnected")
    for values in neighbors.values():
        values.sort()
    return neighbors


Metric = Tuple[int, np.longdouble, np.longdouble, np.longdouble]


def all_root_metrics(
    query: Mapping[str, Any],
    rows_by_relation: Sequence[Sequence[Mapping[str, Any]]],
    weights: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    neighbors = tree_neighbors(query)
    memo: Dict[Tuple[int, int], Dict[int, Metric]] = {}

    def message(node: int, excluded_neighbor: int) -> Dict[int, Metric]:
        key = (node, excluded_neighbor)
        if key in memo:
            return memo[key]
        edge_to_excluded = next(eid for other, eid in neighbors[node] if other == excluded_neighbor)
        grouped: Dict[int, List[Any]] = {}
        for row_index, row in enumerate(rows_by_relation[node]):
            count = 1
            z = np.longdouble(weights[node][row_index])
            z2 = z * z
            maximum = z
            for other, edge_id in neighbors[node]:
                if other == excluded_neighbor:
                    continue
                incoming = message(other, node).get(row["edge_keys"][edge_id])
                if incoming is None:
                    count = 0
                    z = np.longdouble(0)
                    z2 = np.longdouble(0)
                    maximum = np.longdouble(0)
                    break
                count *= incoming[0]
                z *= incoming[1]
                maximum *= incoming[2]
                z2 *= incoming[3]
            separator_key = row["edge_keys"][edge_to_excluded]
            aggregate = grouped.setdefault(
                separator_key,
                [0, np.longdouble(0), np.longdouble(0), np.longdouble(0)],
            )
            aggregate[0] += count
            aggregate[1] += z
            if maximum > aggregate[2]:
                aggregate[2] = maximum
            aggregate[3] += z2
        result = {separator: (values[0], values[1], values[2], values[3]) for separator, values in grouped.items()}
        memo[key] = result
        return result

    roots = []
    join_counts = set()
    for root in range(len(rows_by_relation)):
        total_count = 0
        mu = np.longdouble(0)
        maximum = np.longdouble(0)
        sum_p2 = np.longdouble(0)
        lam = np.longdouble(0)
        for row_index, row in enumerate(rows_by_relation[root]):
            count = 1
            weight = np.longdouble(weights[root][row_index])
            z = weight
            z2 = weight * weight
            row_max = weight
            for other, edge_id in neighbors[root]:
                incoming = message(other, root).get(row["edge_keys"][edge_id])
                if incoming is None:
                    count = 0
                    z = np.longdouble(0)
                    z2 = np.longdouble(0)
                    row_max = np.longdouble(0)
                    break
                count *= incoming[0]
                z *= incoming[1]
                row_max *= incoming[2]
                z2 *= incoming[3]
            total_count += count
            mu += z
            if row_max > maximum:
                maximum = row_max
            sum_p2 += z2
            lam += np.longdouble(count) * row_max
        join_counts.add(total_count)
        roots.append({
            "root": root,
            "join_count": total_count,
            "mu": mu,
            "max_p": maximum,
            "sum_p2": sum_p2,
            "lambda": lam,
            "kappa": lam / mu if mu > 0 else np.longdouble(0),
        })
    if len(join_counts) != 1:
        raise AssertionError(f"root-dependent join counts: {join_counts}")
    raw_min = min(roots, key=lambda item: (item["lambda"], item["root"]))
    minimum = raw_min["lambda"]
    if minimum == 0:
        eligible = [item["root"] for item in roots if item["lambda"] == 0]
    else:
        ceiling = minimum * (np.longdouble(1) + ROOT_NEAR_TIE_RELATIVE_DELTA)
        eligible = [item["root"] for item in roots if item["lambda"] <= ceiling]
    chosen_root = min(eligible)
    return {
        "join_count": next(iter(join_counts)),
        "roots": roots,
        "raw_min_root": raw_min["root"],
        "near_tie_eligible_roots": eligible,
        "chosen_root": chosen_root,
    }


def format_ld(value: np.longdouble) -> str:
    return np.format_float_scientific(value, precision=21, unique=False, trim="k")


def json_metric(metric: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "root": metric["root"],
        "join_count": str(metric["join_count"]),
        "mu": format_ld(metric["mu"]),
        "max_p": format_ld(metric["max_p"]),
        "sum_p2": format_ld(metric["sum_p2"]),
        "lambda": format_ld(metric["lambda"]),
        "kappa": format_ld(metric["kappa"]),
    }


def mode_weights(
    query: Mapping[str, Any],
    rows_by_relation: Sequence[Sequence[Mapping[str, Any]]],
    feature_summaries: Mapping[str, Mapping[str, Any]],
    epsilon: float,
    query_index: int,
    seed_base: int,
) -> Dict[str, List[List[float]]]:
    uniform = [[1.0 for _ in rows] for rows in rows_by_relation]
    single = [[1.0 for _ in rows] for rows in rows_by_relation]
    product = [[1.0 for _ in rows] for rows in rows_by_relation]
    for relation in query["relations"]:
        rid = relation["id"]
        if relation.get("single_anchor_feature"):
            single[rid] = [
                feature_weight(row["source"], relation["table"], relation["single_anchor_feature"], feature_summaries, epsilon)
                for row in rows_by_relation[rid]
            ]
        if relation.get("product_real_feature"):
            product[rid] = [
                feature_weight(row["source"], relation["table"], relation["product_real_feature"], feature_summaries, epsilon)
                for row in rows_by_relation[rid]
            ]
    shuffled = [list(values) for values in product]
    for relation in query["relations"]:
        rid = relation["id"]
        rng = random.Random(seed_base + 100 * query_index + rid)
        rng.shuffle(shuffled[rid])
        if sorted(shuffled[rid]) != sorted(product[rid]):
            raise AssertionError(f"{query['id']} relation {rid}: shuffled multiset changed")
    for mode_name, mode in (("uniform", uniform), ("single-anchor", single), ("product-real", product), ("product-shuffled", shuffled)):
        for rid, values in enumerate(mode):
            if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
                raise AssertionError(f"{query['id']} {mode_name} relation {rid}: weight outside [0,1]")
    return {
        "uniform": uniform,
        "single-anchor": single,
        "product-real": product,
        "product-shuffled": shuffled,
    }


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_structure_and_rows(
    query_dir: Path,
    query: Mapping[str, Any],
    rows_by_relation: Sequence[Sequence[Mapping[str, Any]]],
    key_cardinalities: Mapping[int, int],
) -> None:
    structure = [FORMAT, f"query\t{query['id']}", f"m\t{len(query['relations'])}", f"verification_root\t{query['verification_root']}"]
    incident = incident_edges(query)
    for relation in query["relations"]:
        rid = relation["id"]
        structure.append(
            "\t".join([
                "relation", str(rid), relation["alias"], relation["table"],
                str(len(rows_by_relation[rid])), f"keys/r{rid}.tsv", f"features/r{rid}.tsv",
            ])
        )
    for edge in query["edges"]:
        structure.append(
            "\t".join([
                "edge", str(edge["id"]), str(edge["left"]), str(edge["right"]),
                str(key_cardinalities[edge["id"]]), ",".join(edge["left_columns"]),
                ",".join(edge["right_columns"]),
            ])
        )
    (query_dir / "structure.tsv").write_text("\n".join(structure) + "\n", encoding="utf-8")

    for relation in query["relations"]:
        rid = relation["id"]
        edge_ids = [edge["id"] for edge in incident[rid]]
        write_tsv(
            query_dir / "keys" / f"r{rid}.tsv",
            ["tuple_id"] + [f"edge_{edge_id}_key" for edge_id in edge_ids],
            ([row["tuple_id"]] + [row["edge_keys"][edge_id] for edge_id in edge_ids] for row in rows_by_relation[rid]),
        )
        columns = relation["retain_columns"]
        write_tsv(
            query_dir / "features" / f"r{rid}.tsv",
            ["tuple_id"] + columns,
            ([row["tuple_id"]] + [("\\N" if row["source"][column] is None else row["source"][column]) for column in columns] for row in rows_by_relation[rid]),
        )


def write_weight_set(
    query_dir: Path,
    relative_dir: Path,
    rows_by_relation: Sequence[Sequence[Mapping[str, Any]]],
    weights: Sequence[Sequence[float]],
) -> None:
    for rid, rows in enumerate(rows_by_relation):
        write_tsv(
            query_dir / relative_dir / f"r{rid}.tsv",
            ["tuple_id", "weight"],
            ((row["tuple_id"], format(weights[rid][index], ".17g")) for index, row in enumerate(rows)),
        )


def orientation(query: Mapping[str, Any], root: int) -> Tuple[List[int], List[int]]:
    neighbors = tree_neighbors(query)
    parent = [-2] * len(query["relations"])
    parent[root] = -1
    preorder = []
    queue = deque([root])
    while queue:
        node = queue.popleft()
        preorder.append(node)
        for other, _ in neighbors[node]:
            if parent[other] == -2:
                parent[other] = node
                queue.append(other)
    return parent, preorder


def small_induced_check(
    query: Mapping[str, Any], rows_by_relation: Sequence[Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    root = query["verification_root"]
    neighbors = tree_neighbors(query)
    ones = [[1.0 for _ in rows] for rows in rows_by_relation]
    metrics = all_root_metrics(query, rows_by_relation, ones)
    if metrics["join_count"] == 0:
        return {"status": "empty-full-join", "row_counts": [0 for _ in rows_by_relation], "witness_count": "0"}

    groups: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for rid, rows in enumerate(rows_by_relation):
        for row_index, row in enumerate(rows):
            for _, edge_id in neighbors[rid]:
                groups[(rid, edge_id, row["edge_keys"][edge_id])].append(row_index)

    parent, preorder = orientation(query, root)
    parent_edge = {}
    for node in preorder:
        if parent[node] >= 0:
            parent_edge[node] = next(edge_id for other, edge_id in neighbors[node] if other == parent[node])

    @functools.lru_cache(maxsize=None)
    def row_has_completion(node: int, row_index: int) -> bool:
        row = rows_by_relation[node][row_index]
        for child, edge_id in neighbors[node]:
            if parent[child] != node:
                continue
            candidates = groups.get((child, edge_id, row["edge_keys"][edge_id]), [])
            if not any(row_has_completion(child, candidate) for candidate in candidates):
                return False
        return True

    witness_indices = [-1] * len(rows_by_relation)

    def choose_subtree(node: int, required_key: Optional[int] = None) -> bool:
        if parent[node] < 0:
            candidates = range(len(rows_by_relation[node]))
        else:
            candidates = groups.get((node, parent_edge[node], required_key), [])
        for row_index in candidates:
            if not row_has_completion(node, row_index):
                continue
            witness_indices[node] = row_index
            row = rows_by_relation[node][row_index]
            ok = True
            for child, edge_id in neighbors[node]:
                if parent[child] == node and not choose_subtree(child, row["edge_keys"][edge_id]):
                    ok = False
                    break
            if ok:
                return True
        return False

    if not choose_subtree(root):
        raise AssertionError("positive DP count but no witness found")

    for per_key in (3, 2, 1):
        selected = [{witness_indices[rid]} for rid in range(len(rows_by_relation))]
        for _ in range(len(rows_by_relation)):
            for edge in query["edges"]:
                left, right, edge_id = edge["left"], edge["right"], edge["id"]
                keys = {rows_by_relation[left][index]["edge_keys"][edge_id] for index in selected[left]}
                keys.update(rows_by_relation[right][index]["edge_keys"][edge_id] for index in selected[right])
                for key in sorted(keys):
                    selected[left].update(groups.get((left, edge_id, key), [])[:per_key])
                    selected[right].update(groups.get((right, edge_id, key), [])[:per_key])
            for rid in range(len(selected)):
                if len(selected[rid]) > 16:
                    seed = witness_indices[rid]
                    selected[rid] = {seed} | set(sorted(selected[rid] - {seed})[:15])

        subset = [[rows_by_relation[rid][index] for index in sorted(selected[rid])] for rid in range(len(selected))]
        subset_groups: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
        for rid, rows in enumerate(subset):
            for row_index, row in enumerate(rows):
                for _, edge_id in neighbors[rid]:
                    subset_groups[(rid, edge_id, row["edge_keys"][edge_id])].append(row_index)

        outputs = set()
        assigned = [-1] * len(subset)

        def enumerate_preorder(depth: int) -> None:
            if len(outputs) > 100000:
                return
            node = preorder[depth]
            if parent[node] < 0:
                candidates: Iterable[int] = range(len(subset[node]))
            else:
                edge_id = parent_edge[node]
                parent_row = subset[parent[node]][assigned[parent[node]]]
                candidates = subset_groups.get((node, edge_id, parent_row["edge_keys"][edge_id]), [])
            for row_index in candidates:
                assigned[node] = row_index
                if depth + 1 == len(preorder):
                    outputs.add(tuple(subset[rid][assigned[rid]]["tuple_id"] for rid in range(len(subset))))
                else:
                    enumerate_preorder(depth + 1)

        enumerate_preorder(0)
        if len(outputs) <= 100000:
            subset_metrics = all_root_metrics(query, subset, [[1.0 for _ in rows] for rows in subset])
            if subset_metrics["join_count"] != len(outputs):
                raise AssertionError(
                    f"{query['id']}: induced DP {subset_metrics['join_count']} != enumeration {len(outputs)}"
                )
            digest = hashlib.sha256()
            for output in sorted(outputs):
                digest.update((",".join(map(str, output)) + "\n").encode("ascii"))
            seed_ids = [rows_by_relation[rid][witness_indices[rid]]["tuple_id"] for rid in range(len(rows_by_relation))]
            return {
                "status": "passed",
                "seed_witness_ids": seed_ids,
                "row_counts": [len(rows) for rows in subset],
                "witness_count": str(len(outputs)),
                "witness_sha256": digest.hexdigest(),
                "expansion_per_key": per_key,
            }
    raise AssertionError(f"{query['id']}: unable to bound induced enumeration")


def validate_workload_tags(raw_dir: Path, query_manifest: Mapping[str, Any]) -> Dict[str, Any]:
    lines = (raw_dir / "stats_CEB.sql").read_text(encoding="utf-8").splitlines()
    result = {}
    for query in query_manifest["queries"]:
        line_number = query["provenance"]["workload_line_1based"]
        if line_number is None:
            result[query["id"]] = {"kind": "NEW-pre-registered", "line": None, "cardinality_tag": None}
            continue
        line = lines[line_number - 1]
        prefix, sql = line.split("||", 1)
        actual = int(prefix)
        expected = int(query["provenance"]["cardinality_tag"])
        if actual != expected:
            raise ValueError(f"{query['id']}: workload tag {actual} != manifest {expected}")
        result[query["id"]] = {
            "kind": "STATS-CEB",
            "line": line_number,
            "cardinality_tag": actual,
            "source_sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
        }
    return result


def duckdb_counts_subprocess(
    raw_dir: Path,
    source_manifest_path: Path,
    query_manifest_path: Path,
    temp_dir: Path,
    helper: Path,
) -> Tuple[Dict[str, int], str]:
    command = [
        sys.executable,
        str(helper),
        "--raw-dir", str(raw_dir),
        "--source-manifest", str(source_manifest_path),
        "--query-manifest", str(query_manifest_path),
        "--temp-dir", str(temp_dir),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)
    return {key: int(value) for key, value in payload["counts"].items()}, payload["duckdb_version"]


def hash_prepared_files(output_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        records.append({
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def main() -> None:
    overall_start = time.perf_counter()
    args = parse_args()
    source_manifest = load_json(args.source_manifest)
    query_manifest = load_json(args.query_manifest)
    if query_manifest["prepared_format"] != FORMAT:
        raise ValueError("query manifest prepared format mismatch")
    manifest_delta = np.longdouble(str(query_manifest["probability_protocol"]["root_near_tie_relative_delta"]))
    if manifest_delta != ROOT_NEAR_TIE_RELATIVE_DELTA:
        raise ValueError("query manifest/root near-tie delta mismatch")

    phase_times: Dict[str, Any] = {
        "source_download_seconds": args.source_download_seconds,
        "source_download_note": "Source acquisition time supplied by the caller.",
    }
    phase_start = time.perf_counter()
    tables, inventory = build_source_inventory(args.raw_dir, source_manifest)
    phase_times["source_hash_and_csv_parse_seconds"] = time.perf_counter() - phase_start
    table_types = {
        item["table"]: item["columns"]
        for item in source_manifest["files"] if item["role"] == "table"
    }
    phase_start = time.perf_counter()
    feature_summaries = compute_feature_summaries(tables, query_manifest)
    phase_times["base_table_feature_summary_seconds"] = time.perf_counter() - phase_start
    workload_tags = validate_workload_tags(args.raw_dir, query_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    epsilon = float(query_manifest["probability_protocol"]["epsilon"])
    seed_base = int(query_manifest["probability_protocol"]["shuffle_seed_base_hex"], 16)
    targets = [float(value) for value in query_manifest["probability_protocol"]["targets"]]

    query_results = {}
    for query_index, query in enumerate(query_manifest["queries"]):
        query_times: Dict[str, float] = {}
        query_dir = args.output_dir / query["id"]
        if query_dir.exists():
            shutil.rmtree(query_dir)
        query_dir.mkdir(parents=True)
        phase_start = time.perf_counter()
        rows_by_relation, relation_stats, key_cardinalities = prepare_query_rows(query, tables, table_types)
        query_times["filter_and_key_intern_seconds"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        write_structure_and_rows(query_dir, query, rows_by_relation, key_cardinalities)
        query_times["shared_tsv_write_seconds"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        induced = small_induced_check(query, rows_by_relation)
        query_times["small_induced_validation_seconds"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        all_modes = mode_weights(query, rows_by_relation, feature_summaries, epsilon, query_index, seed_base)
        query_times["weight_generation_seconds"] = time.perf_counter() - phase_start
        query_times["root_planning_seconds"] = 0.0
        query_times["normalization_and_weight_write_seconds"] = 0.0
        cases = []
        modes_result = {}
        root_score_rows = []
        for mode_name in query_manifest["probability_protocol"]["modes"]:
            weights = all_modes[mode_name]
            phase_start = time.perf_counter()
            write_weight_set(query_dir, Path("weights") / mode_name / "unscaled", rows_by_relation, weights)
            query_times["normalization_and_weight_write_seconds"] += time.perf_counter() - phase_start
            phase_start = time.perf_counter()
            metrics = all_root_metrics(query, rows_by_relation, weights)
            query_times["root_planning_seconds"] += time.perf_counter() - phase_start
            roots = metrics["roots"]
            chosen_root = metrics["chosen_root"]
            chosen = roots[chosen_root]
            mu0 = chosen["mu"]
            if mu0 <= 0:
                scale_groups = {0.0: targets}
            else:
                scale_groups: Dict[float, List[float]] = defaultdict(list)
                for target in query_manifest["probability_protocol"].get("targets_by_mode", {}).get(mode_name, targets):
                    scale_groups[min(1.0, target / float(mu0))].append(target)
            for root_metric in roots:
                root_score_rows.append([
                    mode_name, root_metric["root"], format_ld(root_metric["lambda"]),
                    format_ld(root_metric["kappa"]), format_ld(root_metric["mu"]),
                    1 if root_metric["root"] == metrics["raw_min_root"] else 0,
                    1 if root_metric["root"] in metrics["near_tie_eligible_roots"] else 0,
                    1 if root_metric["root"] == chosen_root else 0,
                ])
            anchor = query["anchor_relation"]
            if mode_name == "single-anchor" and mu0 > 0:
                anchor_lambda = roots[anchor]["lambda"]
                relative = abs(anchor_lambda - mu0) / max(abs(mu0), np.longdouble(1))
                if relative > np.longdouble("1e-12"):
                    raise AssertionError(
                        f"{query['id']}: single-anchor Lambda(anchor) != mu: {anchor_lambda} vs {mu0}"
                    )
            weight_sets = []
            for scale_index, (scale, requested) in enumerate(sorted(scale_groups.items(), key=lambda item: item[0])):
                phase_start = time.perf_counter()
                scaled = [list(values) for values in weights]
                scaled[anchor] = [value * scale for value in scaled[anchor]]
                relative = Path("weights") / mode_name / f"scale-{scale_index:03d}"
                write_weight_set(query_dir, relative, rows_by_relation, scaled)
                query_times["normalization_and_weight_write_seconds"] += time.perf_counter() - phase_start
                actual_mu = mu0 * np.longdouble(scale)
                actual_sum_p2 = chosen["sum_p2"] * np.longdouble(scale) * np.longdouble(scale)
                actual_max = chosen["max_p"] * np.longdouble(scale)
                requested_labels = ",".join(format(value, ".17g") for value in requested)
                cases.append([
                    mode_name, requested_labels, relative.as_posix(), str(metrics["join_count"]), format(scale, ".17g"),
                    format_ld(mu0), format_ld(actual_mu), format_ld(actual_sum_p2),
                    format_ld(actual_max), chosen_root,
                ])
                weight_sets.append({
                    "requested_targets": requested,
                    "weight_set": relative.as_posix(),
                    "a": format(scale, ".17g"),
                    "actual_mu": format_ld(actual_mu),
                    "sum_p2": format_ld(actual_sum_p2),
                    "max_p": format_ld(actual_max),
                })
            modes_result[mode_name] = {
                "join_count": str(metrics["join_count"]),
                "raw_min_root": metrics["raw_min_root"],
                "near_tie_eligible_roots": metrics["near_tie_eligible_roots"],
                "chosen_root": chosen_root,
                "mu0": format_ld(mu0),
                "root_scores": [
                    dict(
                        json_metric(item),
                        raw_min=(item["root"] == metrics["raw_min_root"]),
                        near_tie_eligible=(item["root"] in metrics["near_tie_eligible_roots"]),
                        guarded_chosen=(item["root"] == chosen_root),
                    )
                    for item in roots
                ],
                "weight_sets": weight_sets,
            }
        write_tsv(
            query_dir / "cases.tsv",
            ["mode", "requested_targets", "weight_set", "join_size", "a", "mu0", "actual_mu", "sum_p2", "max_p", "chosen_root"],
            cases,
        )
        write_tsv(
            query_dir / "root_scores.tsv",
            ["mode", "root", "lambda", "kappa", "mu", "raw_min", "near_tie_eligible", "guarded_chosen"],
            root_score_rows,
        )
        unweighted_count = int(modes_result["uniform"]["join_count"])
        query_results[query["id"]] = {
            "output_label": query["output_label"],
            "provenance": workload_tags[query["id"]],
            "relation_stats": relation_stats,
            "edge_key_cardinalities": {str(key): value for key, value in key_cardinalities.items()},
            "tree_dp_count": str(unweighted_count),
            "cardinality_tag_match": (
                None if query["provenance"]["cardinality_tag"] is None
                else unweighted_count == int(query["provenance"]["cardinality_tag"])
            ),
            "small_induced": induced,
            "modes": modes_result,
            "common_preparation_wall_seconds": query_times,
        }

    # DuckDB reparses the raw CSV independently.  Release the Python row store
    # first so the oracle fits under the preparation process's 4 GiB cap.
    tables.clear()
    del rows_by_relation, all_modes, weights, scaled
    gc.collect()

    duck_counts: Dict[str, int] = {}
    duckdb_version: Optional[str] = None
    if not args.skip_duckdb:
        phase_start = time.perf_counter()
        helper = args.duckdb_helper or Path(__file__).with_name("validate_stats_duckdb.py")
        duck_counts, duckdb_version = duckdb_counts_subprocess(
            args.raw_dir, args.source_manifest, args.query_manifest, args.duckdb_temp_dir, helper
        )
        for query_id, duck_count in duck_counts.items():
            dp_count = int(query_results[query_id]["tree_dp_count"])
            if duck_count != dp_count:
                raise AssertionError(f"{query_id}: DuckDB count {duck_count} != tree DP {dp_count}")
            query_results[query_id]["duckdb_count"] = str(duck_count)
            query_results[query_id]["duckdb_tree_dp_match"] = True
        phase_times["duckdb_independent_validation_seconds"] = time.perf_counter() - phase_start

    inventory["feature_summaries"] = feature_summaries
    inventory["python_version"] = platform.python_version()
    inventory["numpy_version"] = np.__version__
    args.inventory_out.parent.mkdir(parents=True, exist_ok=True)
    args.inventory_out.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = {
        "format": "stats-loader-validation-v1",
        "preparation_version": PREPARATION_VERSION,
        "source_commit": source_manifest["commit"],
        "source_inventory_sha256": sha256_file(args.inventory_out),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "duckdb_version": duckdb_version,
        "duckdb_skipped": args.skip_duckdb,
        "timestamp_rule": "timezone-free proleptic-Gregorian integer seconds since 1970-01-01",
        "null_rule": "empty or \\N is NULL; rows NULL on any incident equality key are ineligible",
        "root_near_tie_rule": {
            "relative_delta": format(ROOT_NEAR_TIE_RELATIVE_DELTA, ".17g"),
            "positive_minimum": "eligible iff Lambda <= Lmin*(1+delta), then smallest manifest relation ID",
            "zero_minimum": "eligible iff Lambda == 0, then smallest manifest relation ID",
            "status": "deterministic practical guard, not a certified exact-minimizer error bound"
        },
        "common_preparation_wall_seconds": phase_times,
        "timing_scope_note": "Data preparation diagnostics.",
        "query_results": query_results,
    }
    args.validation_out.parent.mkdir(parents=True, exist_ok=True)
    args.validation_out.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    phase_start = time.perf_counter()
    validation["prepared_files"] = hash_prepared_files(args.output_dir)
    phase_times["prepared_file_hash_seconds"] = time.perf_counter() - phase_start
    phase_times["overall_prepare_seconds"] = time.perf_counter() - overall_start
    args.validation_out.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "queries": {query_id: result["tree_dp_count"] for query_id, result in query_results.items()},
        "inventory": str(args.inventory_out),
        "validation": str(args.validation_out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
