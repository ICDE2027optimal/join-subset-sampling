#!/usr/bin/env python3
"""Inventory and prepare the fixed May-2013 IMDb snapshot for JOB 1a.

The inventory stage is intentionally independent of probability generation.  It
parses every selected archive member, checks the source schema assumptions and
records the exact domains needed to specify the JOB weight protocol.  The source
CSV files have no header.  An unquoted empty field is SQL NULL, whereas quoted
``""`` is a non-NULL empty string, following PostgreSQL COPY CSV semantics.
Double quotes mark quoted fields. Inside such a field backslash escapes a quote
or a backslash.  Outside a quoted field backslashes are literal IMDb data.
Python's ``csv`` module applies ``escapechar`` outside quotes too, so the parser
below implements the source convention explicitly.
"""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import deque
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PREPARATION_VERSION = "prepare-job-v1"
EXPECTED_PREPARED_FORMAT = "stats-undirected-v1"
ROOT_NEAR_TIE_RELATIVE_DELTA = np.longdouble("1e-12")
YEAR_NULL_SENTINEL = np.int32(-(1 << 31))
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--query-manifest", required=True, type=Path)
    parser.add_argument("--inventory-out", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validation-out", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--moment-validation-out", type=Path)
    parser.add_argument("--duckdb-python", type=Path)
    parser.add_argument("--duckdb-temp-dir", type=Path)
    parser.add_argument(
        "--stage", choices=("inventory", "oracle", "prepare", "finalize", "verify-moments"), default="inventory",
    )
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


def parse_int(text: Optional[str], *, nullable: bool, context: str) -> Optional[int]:
    if text is None:
        if nullable:
            return None
        raise ValueError(f"{context}: unexpected SQL NULL")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid integer {text!r}") from exc


def parse_job_csv_line(line: str, *, context: str) -> List[Optional[str]]:
    """Parse one physical JOB CSV row with escape active only within quotes."""
    fields: List[Optional[str]] = []
    length = len(line)
    position = 0
    while True:
        if position < length and line[position] == '"':
            position += 1
            pieces: List[str] = []
            while True:
                if position >= length:
                    raise ValueError(f"{context}: unterminated quoted field")
                char = line[position]
                if char == "\\" and position + 1 < length and line[position + 1] in {'"', "\\"}:
                    pieces.append(line[position + 1])
                    position += 2
                elif char == '"':
                    if position + 1 < length and line[position + 1] == '"':
                        pieces.append('"')
                        position += 2
                    else:
                        position += 1
                        break
                else:
                    pieces.append(char)
                    position += 1
            field = "".join(pieces)
            if position < length and line[position] != ",":
                raise ValueError(
                    f"{context}: expected comma after quoted field at byte {position}"
                )
        else:
            comma = line.find(",", position)
            end = length if comma < 0 else comma
            raw_field = line[position:end]
            field = None if raw_field == "" else raw_field
            position = end
        fields.append(field)
        if position >= length:
            break
        # The current byte is the delimiter checked or found above.
        position += 1
        if position == length:
            fields.append(None)
            break
    return fields


def csv_rows(path: Path, width: int) -> Iterable[Tuple[int, List[Optional[str]]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for physical_row, raw_line in enumerate(handle, start=1):
            if raw_line.endswith("\n"):
                raw_line = raw_line[:-1]
            if raw_line.endswith("\r"):
                raw_line = raw_line[:-1]
            row = parse_job_csv_line(raw_line, context=f"{path.name}:{physical_row}")
            if len(row) != width:
                raise ValueError(
                    f"{path.name}:{physical_row}: parsed {len(row)} fields, expected {width}: {row!r}"
                )
            yield physical_row, row


def summarize_table(
    raw_dir: Path, table: Mapping[str, Any], target_company_kind: str, target_info: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    path = raw_dir / table["member"]
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = int(table["bytes"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{path.name}: {path.stat().st_size} bytes != expected {expected_bytes}")

    columns = list(table["columns"])
    types = list(table["types"])
    null_counts = {column: 0 for column in columns}
    quoted_empty_counts = {column: 0 for column in columns}
    seen_ids = set()
    row_count = 0
    min_id: Optional[int] = None
    max_id: Optional[int] = None
    extra: Dict[str, Any] = {}

    selected_ids: List[int] = []
    note_nonnull = 0
    note_include = 0
    note_excluded_mgm = 0
    mc_filter_count = 0
    valid_year_count = 0
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    top_rank_rows = 0
    top_rank_parse_valid = 0
    top_rank_parse_invalid = 0
    top_rank_min: Optional[int] = None
    top_rank_max: Optional[int] = None

    name = table["name"]
    for physical_row, row in csv_rows(path, len(columns)):
        row_count += 1
        for column, value in zip(columns, row):
            if value is None:
                null_counts[column] += 1
            elif value == "":
                quoted_empty_counts[column] += 1

        row_id = parse_int(row[0], nullable=False, context=f"{path.name}:{physical_row}:id")
        assert row_id is not None
        if row_id in seen_ids:
            raise ValueError(f"{path.name}:{physical_row}: duplicate id {row_id}")
        seen_ids.add(row_id)
        min_id = row_id if min_id is None else min(min_id, row_id)
        max_id = row_id if max_id is None else max(max_id, row_id)

        # Validate every source column declared integer.  The original schema
        # determines which integer columns may be NULL.
        for position, (column, kind) in enumerate(zip(columns, types)):
            if kind == "int":
                nullable = not (
                    column == "id"
                    or (name == "movie_companies" and column in {"movie_id", "company_id", "company_type_id"})
                    or (name == "movie_info_idx" and column in {"movie_id", "info_type_id"})
                    or (name == "title" and column == "kind_id")
                )
                parse_int(
                    row[position], nullable=nullable,
                    context=f"{path.name}:{physical_row}:{column}",
                )

        if name == "company_type" and row[1] == target_company_kind:
            selected_ids.append(row_id)
        elif name == "info_type" and row[1] == target_info:
            selected_ids.append(row_id)
        elif name == "movie_companies":
            note = row[4]
            if note is not None:
                note_nonnull += 1
                include = "(co-production)" in note or "(presents)" in note
                excluded = "(as Metro-Goldwyn-Mayer Pictures)" in note
                note_include += int(include)
                note_excluded_mgm += int(excluded)
                # This exactly matches SQL three-valued logic because a NULL
                # note cannot make the WHERE conjunction TRUE.
                if include and not excluded:
                    mc_filter_count += 1
        elif name == "title":
            year = parse_int(
                row[4], nullable=True,
                context=f"{path.name}:{physical_row}:production_year",
            )
            if year is not None:
                valid_year_count += 1
                min_year = year if min_year is None else min(min_year, year)
                max_year = year if max_year is None else max(max_year, year)

    if name in {"company_type", "info_type"}:
        extra["filter_selected_ids"] = selected_ids
        extra["filter_selected_count"] = len(selected_ids)
    elif name == "movie_companies":
        extra.update(
            {
                "note_nonnull_count": note_nonnull,
                "note_include_or_count": note_include,
                "note_excluded_mgm_count": note_excluded_mgm,
                "job1a_note_filter_count": mc_filter_count,
            }
        )
    elif name == "title":
        extra["production_year_domain"] = {
            "valid_count": valid_year_count,
            "null_count": null_counts["production_year"],
            "min": min_year,
            "max": max_year,
        }

    record: Dict[str, Any] = {
        "table": name,
        "member": table["member"],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "columns": columns,
        "types": types,
        "row_count": row_count,
        "unique_id_count": len(seen_ids),
        "min_id": min_id,
        "max_id": max_id,
        "null_counts": null_counts,
        "quoted_empty_counts": quoted_empty_counts,
    }
    record.update(extra)
    return record, {"selected_ids": selected_ids}


def audit_top_rank(
    raw_dir: Path, table: Mapping[str, Any], target_info_type_ids: Sequence[int]
) -> Dict[str, Any]:
    path = raw_dir / table["member"]
    target = set(target_info_type_ids)
    count = valid = invalid = 0
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    invalid_examples: List[Dict[str, Any]] = []
    for physical_row, row in csv_rows(path, len(table["columns"])):
        info_type_id = parse_int(
            row[2], nullable=False,
            context=f"{path.name}:{physical_row}:info_type_id",
        )
        if info_type_id not in target:
            continue
        count += 1
        try:
            # Python int accepts surrounding whitespace, matching the stated
            # finite positive integral parse rule after whitespace trimming.
            if row[3] is None:
                raise ValueError("SQL NULL rank")
            rank = int(row[3].strip())
            if rank <= 0:
                raise ValueError("not positive")
        except (ValueError, OverflowError):
            invalid += 1
            if len(invalid_examples) < 10:
                invalid_examples.append({"id": int(row[0]), "info": row[3]})
            continue
        valid += 1
        minimum = rank if minimum is None else min(minimum, rank)
        maximum = rank if maximum is None else max(maximum, rank)
    return {
        "target_info_type_ids": sorted(target),
        "row_count": count,
        "positive_integral_count": valid,
        "invalid_count": invalid,
        "invalid_examples": invalid_examples,
        "min_rank": minimum,
        "max_rank": maximum,
    }


def inventory(
    raw_dir: Path, source_manifest: Mapping[str, Any], query_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    if query_manifest["prepared_format"] != EXPECTED_PREPARED_FORMAT:
        raise ValueError("prepared format drift")
    query = query_manifest["queries"][0]
    target_company_kind = query["filters"]["ct"][0]["value"]
    target_info = query["filters"]["it"][0]["value"]

    records: List[Dict[str, Any]] = []
    side: Dict[str, Dict[str, Any]] = {}
    by_name = {table["name"]: table for table in source_manifest["tables"]}
    for table in source_manifest["tables"]:
        record, table_side = summarize_table(
            raw_dir, table, target_company_kind, target_info
        )
        records.append(record)
        side[table["name"]] = table_side

    company_ids = side["company_type"]["selected_ids"]
    info_type_ids = side["info_type"]["selected_ids"]
    if len(company_ids) != 1:
        raise ValueError(f"expected one company_type target, got {company_ids}")
    if len(info_type_ids) != 1:
        raise ValueError(f"expected one info_type target, got {info_type_ids}")

    top_rank = audit_top_rank(raw_dir, by_name["movie_info_idx"], info_type_ids)
    return {
        "format": "job-source-inventory-v1",
        "preparation_version": PREPARATION_VERSION,
        "snapshot": source_manifest["snapshot"],
        "archive": source_manifest["archive"],
        "csv": source_manifest["csv"],
        "query": query["id"],
        "target_company_type_ids": company_ids,
        "target_info_type_ids": info_type_ids,
        "tables": records,
        "top_rank_domain": top_rank,
        "null_semantics": {
            "source_null": "unquoted empty CSV field",
            "quoted_empty": "non-NULL empty string",
            "join": "SQL equality never matches NULL",
            "movie_companies_note": "NULL makes the JOB1a WHERE conjunction non-TRUE",
        },
    }


def np_i64(values: array) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def np_i32(values: array) -> np.ndarray:
    return np.asarray(values, dtype=np.int32)


def build_prepared_rows(
    raw_dir: Path, source_manifest: Mapping[str, Any], query_manifest: Mapping[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load only compact integer columns for the fixed, local-filter-only query."""
    tables = {item["name"]: item for item in source_manifest["tables"]}
    query = query_manifest["queries"][0]
    target_kind = query["filters"]["ct"][0]["value"]
    target_info = query["filters"]["it"][0]["value"]

    ids: List[array] = [array("q") for _ in range(5)]
    raw_keys: List[Dict[int, array]] = [
        {0: array("q")},
        {0: array("q"), 1: array("q")},
        {1: array("q"), 2: array("q")},
        {2: array("q"), 3: array("q")},
        {3: array("q")},
    ]
    years = array("i")
    ranks = array("i")

    for _, row in csv_rows(raw_dir / "company_type.csv", 2):
        if row[1] != target_kind:
            continue
        row_id = parse_int(row[0], nullable=False, context="company_type.id")
        assert row_id is not None
        ids[0].append(row_id)
        raw_keys[0][0].append(row_id)

    for _, row in csv_rows(raw_dir / "movie_companies.csv", 5):
        note = row[4]
        if note is None:
            continue
        if "(as Metro-Goldwyn-Mayer Pictures)" in note:
            continue
        if "(co-production)" not in note and "(presents)" not in note:
            continue
        row_id = parse_int(row[0], nullable=False, context="movie_companies.id")
        movie_id = parse_int(row[1], nullable=False, context="movie_companies.movie_id")
        company_type_id = parse_int(row[3], nullable=False, context="movie_companies.company_type_id")
        assert row_id is not None and movie_id is not None and company_type_id is not None
        ids[1].append(row_id)
        raw_keys[1][0].append(company_type_id)
        raw_keys[1][1].append(movie_id)

    for _, row in csv_rows(raw_dir / "title.csv", 12):
        row_id = parse_int(row[0], nullable=False, context="title.id")
        year = parse_int(row[4], nullable=True, context="title.production_year")
        assert row_id is not None
        ids[2].append(row_id)
        raw_keys[2][1].append(row_id)
        raw_keys[2][2].append(row_id)
        years.append(int(YEAR_NULL_SENTINEL) if year is None else year)

    target_info_type_id = 112
    for _, row in csv_rows(raw_dir / "movie_info_idx.csv", 5):
        row_id = parse_int(row[0], nullable=False, context="movie_info_idx.id")
        movie_id = parse_int(row[1], nullable=False, context="movie_info_idx.movie_id")
        info_type_id = parse_int(row[2], nullable=False, context="movie_info_idx.info_type_id")
        assert row_id is not None and movie_id is not None and info_type_id is not None
        rank = 0
        if info_type_id == target_info_type_id and row[3] is not None:
            try:
                candidate = int(row[3].strip())
            except (ValueError, OverflowError):
                candidate = 0
            if candidate > 0:
                rank = candidate
        ids[3].append(row_id)
        raw_keys[3][2].append(movie_id)
        raw_keys[3][3].append(info_type_id)
        ranks.append(rank)

    for _, row in csv_rows(raw_dir / "info_type.csv", 2):
        if row[1] != target_info:
            continue
        row_id = parse_int(row[0], nullable=False, context="info_type.id")
        assert row_id is not None
        ids[4].append(row_id)
        raw_keys[4][3].append(row_id)

    stores: List[Dict[str, Any]] = []
    for rid in range(5):
        tuple_ids = np_i64(ids[rid])
        relation_raw_keys = {eid: np_i64(values) for eid, values in raw_keys[rid].items()}
        if any(len(values) != len(tuple_ids) for values in relation_raw_keys.values()):
            raise AssertionError(f"relation {rid}: key vector length mismatch")
        stores.append({"tuple_ids": tuple_ids, "raw_keys": relation_raw_keys, "edge_keys": {}})
    stores[2]["production_year"] = np_i32(years)
    stores[3]["top_rank"] = np_i32(ranks)
    if len(stores[2]["production_year"]) != len(stores[2]["tuple_ids"]):
        raise AssertionError("title feature length mismatch")
    if len(stores[3]["top_rank"]) != len(stores[3]["tuple_ids"]):
        raise AssertionError("movie_info_idx feature length mismatch")

    relation_stats = [
        {"relation_id": 0, "alias": "ct", "table": "company_type", "base_rows": 4,
         "filter_pass_rows": len(stores[0]["tuple_ids"]), "null_join_rows_dropped": 0,
         "prepared_rows": len(stores[0]["tuple_ids"])},
        {"relation_id": 1, "alias": "mc", "table": "movie_companies", "base_rows": 2609129,
         "filter_pass_rows": len(stores[1]["tuple_ids"]), "null_join_rows_dropped": 0,
         "prepared_rows": len(stores[1]["tuple_ids"])},
        {"relation_id": 2, "alias": "t", "table": "title", "base_rows": 2528312,
         "filter_pass_rows": len(stores[2]["tuple_ids"]), "null_join_rows_dropped": 0,
         "prepared_rows": len(stores[2]["tuple_ids"])},
        {"relation_id": 3, "alias": "mi_idx", "table": "movie_info_idx", "base_rows": 1380035,
         "filter_pass_rows": len(stores[3]["tuple_ids"]), "null_join_rows_dropped": 0,
         "prepared_rows": len(stores[3]["tuple_ids"])},
        {"relation_id": 4, "alias": "it", "table": "info_type", "base_rows": 113,
         "filter_pass_rows": len(stores[4]["tuple_ids"]), "null_join_rows_dropped": 0,
         "prepared_rows": len(stores[4]["tuple_ids"])},
    ]
    return stores, relation_stats


def intern_edge_keys(
    stores: Sequence[Dict[str, Any]], query: Mapping[str, Any]
) -> Dict[int, int]:
    cardinalities: Dict[int, int] = {}
    for edge in query["edges"]:
        eid = int(edge["id"])
        left = int(edge["left"])
        right = int(edge["right"])
        left_raw = stores[left]["raw_keys"][eid]
        right_raw = stores[right]["raw_keys"][eid]
        values = np.unique(np.concatenate((left_raw, right_raw)))
        left_encoded = np.searchsorted(values, left_raw).astype(np.int64) + 1
        right_encoded = np.searchsorted(values, right_raw).astype(np.int64) + 1
        if not np.array_equal(values[left_encoded - 1], left_raw):
            raise AssertionError(f"edge {eid}: left interning failed")
        if not np.array_equal(values[right_encoded - 1], right_raw):
            raise AssertionError(f"edge {eid}: right interning failed")
        stores[left]["edge_keys"][eid] = left_encoded
        stores[right]["edge_keys"][eid] = right_encoded
        cardinalities[eid] = int(len(values))
    for store in stores:
        store.pop("raw_keys", None)
    return cardinalities


def tree_neighbors(query: Mapping[str, Any]) -> Dict[int, List[Tuple[int, int]]]:
    neighbors = {int(relation["id"]): [] for relation in query["relations"]}
    for edge in query["edges"]:
        left, right, eid = int(edge["left"]), int(edge["right"]), int(edge["id"])
        neighbors[left].append((right, eid))
        neighbors[right].append((left, eid))
    if len(query["edges"]) != len(query["relations"]) - 1:
        raise ValueError("JOB relation graph is not a tree")
    for values in neighbors.values():
        values.sort()
    return neighbors


def exact_tree_count(
    stores: Sequence[Mapping[str, Any]], cardinalities: Mapping[int, int]
) -> int:
    """Exact integer DP specialized to ct--mc--t--mi_idx--it."""
    ct_support = np.zeros(cardinalities[0] + 1, dtype=np.uint64)
    np.add.at(ct_support, stores[0]["edge_keys"][0], np.uint64(1))
    mc_valid = ct_support[stores[1]["edge_keys"][0]]
    mc_by_movie = np.zeros(cardinalities[1] + 1, dtype=np.uint64)
    np.add.at(mc_by_movie, stores[1]["edge_keys"][1], mc_valid)

    it_support = np.zeros(cardinalities[3] + 1, dtype=np.uint64)
    np.add.at(it_support, stores[4]["edge_keys"][3], np.uint64(1))
    mi_valid = it_support[stores[3]["edge_keys"][3]]
    mi_by_movie = np.zeros(cardinalities[2] + 1, dtype=np.uint64)
    np.add.at(mi_by_movie, stores[3]["edge_keys"][2], mi_valid)

    left = mc_by_movie[stores[2]["edge_keys"][1]]
    right = mi_by_movie[stores[2]["edge_keys"][2]]
    return sum(int(a) * int(b) for a, b in zip(left, right))


MetricArrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def all_root_metrics_compact(
    stores: Sequence[Mapping[str, Any]], query: Mapping[str, Any],
    cardinalities: Mapping[int, int], weights: Sequence[np.ndarray], expected_count: int,
) -> Dict[str, Any]:
    neighbors = tree_neighbors(query)
    memo: Dict[Tuple[int, int], MetricArrays] = {}

    def message(node: int, excluded: int) -> MetricArrays:
        cache_key = (node, excluded)
        if cache_key in memo:
            return memo[cache_key]
        outgoing_eid = next(eid for other, eid in neighbors[node] if other == excluded)
        size = len(stores[node]["tuple_ids"])
        row_count = np.ones(size, dtype=np.uint64)
        row_sum = np.asarray(weights[node], dtype=np.longdouble).copy()
        row_max = row_sum.copy()
        row_sum2 = row_sum * row_sum
        for other, eid in neighbors[node]:
            if other == excluded:
                continue
            incoming = message(other, node)
            keys = stores[node]["edge_keys"][eid]
            factor_count = incoming[0][keys]
            overflow = (factor_count != 0) & (row_count > np.iinfo(np.uint64).max // np.maximum(factor_count, 1))
            if bool(np.any(overflow)):
                raise OverflowError("uint64 subtree count overflow")
            row_count *= factor_count
            row_sum *= incoming[1][keys]
            row_max *= incoming[2][keys]
            row_sum2 *= incoming[3][keys]
        out_size = cardinalities[outgoing_eid] + 1
        grouped_count = np.zeros(out_size, dtype=np.uint64)
        grouped_sum = np.zeros(out_size, dtype=np.longdouble)
        grouped_max = np.zeros(out_size, dtype=np.longdouble)
        grouped_sum2 = np.zeros(out_size, dtype=np.longdouble)
        out_keys = stores[node]["edge_keys"][outgoing_eid]
        np.add.at(grouped_count, out_keys, row_count)
        np.add.at(grouped_sum, out_keys, row_sum)
        np.maximum.at(grouped_max, out_keys, row_max)
        np.add.at(grouped_sum2, out_keys, row_sum2)
        result = (grouped_count, grouped_sum, grouped_max, grouped_sum2)
        memo[cache_key] = result
        return result

    roots: List[Dict[str, Any]] = []
    for root in range(len(stores)):
        size = len(stores[root]["tuple_ids"])
        row_count = np.ones(size, dtype=np.uint64)
        row_sum = np.asarray(weights[root], dtype=np.longdouble).copy()
        row_max = row_sum.copy()
        row_sum2 = row_sum * row_sum
        for other, eid in neighbors[root]:
            incoming = message(other, root)
            keys = stores[root]["edge_keys"][eid]
            factor_count = incoming[0][keys]
            overflow = (factor_count != 0) & (row_count > np.iinfo(np.uint64).max // np.maximum(factor_count, 1))
            if bool(np.any(overflow)):
                raise OverflowError("uint64 root count overflow")
            row_count *= factor_count
            row_sum *= incoming[1][keys]
            row_max *= incoming[2][keys]
            row_sum2 *= incoming[3][keys]
        total_count = int(np.sum(row_count, dtype=np.uint64))
        if total_count != expected_count:
            raise AssertionError(f"root {root}: compact DP count {total_count} != exact {expected_count}")
        mu = np.sum(row_sum, dtype=np.longdouble)
        maximum = np.max(row_max, initial=np.longdouble(0))
        sum_p2 = np.sum(row_sum2, dtype=np.longdouble)
        lam = np.sum(row_count.astype(np.longdouble) * row_max, dtype=np.longdouble)
        roots.append({
            "root": root, "join_count": total_count, "mu": mu, "max_p": maximum,
            "sum_p2": sum_p2, "lambda": lam,
            "kappa": lam / mu if mu > 0 else np.longdouble(0),
        })
    raw_min = min(roots, key=lambda item: (item["lambda"], item["root"]))
    minimum = raw_min["lambda"]
    if minimum == 0:
        eligible = [item["root"] for item in roots if item["lambda"] == 0]
    else:
        ceiling = minimum * (np.longdouble(1) + ROOT_NEAR_TIE_RELATIVE_DELTA)
        eligible = [item["root"] for item in roots if item["lambda"] <= ceiling]
    return {
        "join_count": expected_count,
        "roots": roots,
        "raw_min_root": raw_min["root"],
        "near_tie_eligible_roots": eligible,
        "chosen_root": min(eligible),
    }


def build_weight_modes(
    stores: Sequence[Mapping[str, Any]], query_manifest: Mapping[str, Any]
) -> Dict[str, List[np.ndarray]]:
    protocol = query_manifest["probability_protocol"]
    epsilon = float(protocol["epsilon"])
    counts = [len(store["tuple_ids"]) for store in stores]
    uniform = [np.ones(count, dtype=np.float64) for count in counts]
    year_weight = np.full(counts[2], epsilon, dtype=np.float64)
    years = stores[2]["production_year"]
    valid_year = years != YEAR_NULL_SENTINEL
    year_weight[valid_year] = epsilon + (1.0 - epsilon) * (
        (years[valid_year].astype(np.float64) - 1880.0) / (2019.0 - 1880.0)
    )
    if bool(np.any((year_weight < epsilon) | (year_weight > 1.0))):
        raise AssertionError("Title year weight outside [epsilon,1]")

    rank_weight = np.full(counts[3], epsilon, dtype=np.float64)
    ranks = stores[3]["top_rank"]
    valid_rank = ranks > 0
    rank_weight[valid_rank] = epsilon + (1.0 - epsilon) / ranks[valid_rank].astype(np.float64)
    if int(np.count_nonzero(valid_rank)) != 250:
        raise AssertionError("expected exactly 250 top-rank source rows")

    single = [values.copy() for values in uniform]
    single[2] = year_weight.copy()
    product = [values.copy() for values in uniform]
    product[2] = year_weight.copy()
    product[3] = rank_weight.copy()
    shuffled: List[np.ndarray] = []
    seed_base = int(protocol["shuffle_seed_base_hex"], 16)
    for rid, values in enumerate(product):
        as_list = values.tolist()
        random.Random(seed_base + rid).shuffle(as_list)
        shuffled_values = np.asarray(as_list, dtype=np.float64)
        if not np.array_equal(np.sort(shuffled_values), np.sort(values)):
            raise AssertionError(f"relation {rid}: shuffled multiset changed")
        shuffled.append(shuffled_values)
    return {
        "uniform": uniform,
        "single-anchor": single,
        "product-real": product,
        "product-shuffled": shuffled,
    }


def format_ld(value: np.longdouble) -> str:
    return np.format_float_scientific(value, precision=21, unique=False, trim="k")


def write_numeric_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        batch: List[str] = []
        for row in rows:
            batch.append("\t".join(str(value) for value in row) + "\n")
            if len(batch) >= 8192:
                handle.writelines(batch)
                batch.clear()
        if batch:
            handle.writelines(batch)


def write_shared_tsv(
    query_dir: Path, stores: Sequence[Mapping[str, Any]], query: Mapping[str, Any],
    cardinalities: Mapping[int, int],
) -> None:
    structure = [
        EXPECTED_PREPARED_FORMAT, f"query\t{query['id']}",
        f"m\t{len(query['relations'])}", f"verification_root\t{query['verification_root']}",
    ]
    for relation in query["relations"]:
        rid = int(relation["id"])
        structure.append("\t".join([
            "relation", str(rid), relation["alias"], relation["table"],
            str(len(stores[rid]["tuple_ids"])), f"keys/r{rid}.tsv", f"features/r{rid}.tsv",
        ]))
    for edge in query["edges"]:
        eid = int(edge["id"])
        structure.append("\t".join([
            "edge", str(eid), str(edge["left"]), str(edge["right"]), str(cardinalities[eid]),
            ",".join(edge["left_columns"]), ",".join(edge["right_columns"]),
        ]))
    (query_dir / "structure.tsv").write_text("\n".join(structure) + "\n", encoding="utf-8")

    neighbors = tree_neighbors(query)
    for rid, store in enumerate(stores):
        edge_ids = sorted(eid for _, eid in neighbors[rid])
        write_numeric_tsv(
            query_dir / "keys" / f"r{rid}.tsv",
            ["tuple_id"] + [f"edge_{eid}_key" for eid in edge_ids],
            (
                [int(store["tuple_ids"][index])] + [int(store["edge_keys"][eid][index]) for eid in edge_ids]
                for index in range(len(store["tuple_ids"]))
            ),
        )
        if rid == 2:
            write_numeric_tsv(
                query_dir / "features" / "r2.tsv", ["tuple_id", "production_year"],
                ((int(store["tuple_ids"][i]), "\\N" if store["production_year"][i] == YEAR_NULL_SENTINEL else int(store["production_year"][i]))
                 for i in range(len(store["tuple_ids"]))),
            )
        elif rid == 3:
            write_numeric_tsv(
                query_dir / "features" / "r3.tsv", ["tuple_id", "top_rank_positive_int"],
                ((int(store["tuple_ids"][i]), "\\N" if store["top_rank"][i] == 0 else int(store["top_rank"][i]))
                 for i in range(len(store["tuple_ids"]))),
            )
        else:
            write_numeric_tsv(
                query_dir / "features" / f"r{rid}.tsv", ["tuple_id"],
                ((int(value),) for value in store["tuple_ids"]),
            )


def write_weight_set(
    query_dir: Path, relative: Path, stores: Sequence[Mapping[str, Any]], weights: Sequence[np.ndarray]
) -> None:
    for rid, store in enumerate(stores):
        write_numeric_tsv(
            query_dir / relative / f"r{rid}.tsv", ["tuple_id", "weight"],
            ((int(store["tuple_ids"][i]), format(float(weights[rid][i]), ".17g"))
             for i in range(len(store["tuple_ids"]))),
        )


def tiny_witness_check(stores: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build a deterministic bounded induced instance and enumerate all ID vectors."""
    # Encoded keys suffice because both endpoints of every edge share an interner.
    valid_ct_keys = set(int(value) for value in stores[0]["edge_keys"][0])
    valid_it_keys = set(int(value) for value in stores[4]["edge_keys"][3])
    mc_by_movie: Dict[int, List[int]] = {}
    for i in range(len(stores[1]["tuple_ids"])):
        if int(stores[1]["edge_keys"][0][i]) in valid_ct_keys:
            mc_by_movie.setdefault(int(stores[1]["edge_keys"][1][i]), []).append(i)
    mi_by_movie: Dict[int, List[int]] = {}
    for i in range(len(stores[3]["tuple_ids"])):
        if int(stores[3]["edge_keys"][3][i]) in valid_it_keys:
            mi_by_movie.setdefault(int(stores[3]["edge_keys"][2][i]), []).append(i)
    # The two movie edges use separate interners, so bridge through the title row.
    candidates: List[Tuple[int, int, int, int]] = []
    for title_index in range(len(stores[2]["tuple_ids"])):
        left_key = int(stores[2]["edge_keys"][1][title_index])
        right_key = int(stores[2]["edge_keys"][2][title_index])
        if left_key in mc_by_movie and right_key in mi_by_movie:
            candidates.append((int(stores[2]["tuple_ids"][title_index]), title_index, left_key, right_key))
    if not candidates:
        raise AssertionError("JOB1a full join unexpectedly empty")
    _, title_index, left_key, right_key = min(candidates)
    subset_indices = [
        list(range(min(3, len(stores[0]["tuple_ids"])))),
        sorted(mc_by_movie[left_key], key=lambda i: int(stores[1]["tuple_ids"][i]))[:3],
        [title_index],
        sorted(mi_by_movie[right_key], key=lambda i: int(stores[3]["tuple_ids"][i]))[:3],
        list(range(min(3, len(stores[4]["tuple_ids"])))),
    ]
    outputs = []
    for i0 in subset_indices[0]:
        for i1 in subset_indices[1]:
            if stores[0]["edge_keys"][0][i0] != stores[1]["edge_keys"][0][i1]:
                continue
            for i2 in subset_indices[2]:
                if stores[1]["edge_keys"][1][i1] != stores[2]["edge_keys"][1][i2]:
                    continue
                for i3 in subset_indices[3]:
                    if stores[2]["edge_keys"][2][i2] != stores[3]["edge_keys"][2][i3]:
                        continue
                    for i4 in subset_indices[4]:
                        if stores[3]["edge_keys"][3][i3] != stores[4]["edge_keys"][3][i4]:
                            continue
                        outputs.append(tuple(
                            int(stores[rid]["tuple_ids"][index])
                            for rid, index in enumerate((i0, i1, i2, i3, i4))
                        ))
    if not outputs:
        raise AssertionError("tiny induced witness set is empty")
    digest = hashlib.sha256()
    for output in sorted(outputs):
        digest.update((",".join(map(str, output)) + "\n").encode("ascii"))
    return {
        "status": "passed",
        "row_counts": [len(values) for values in subset_indices],
        "witness_count": len(outputs),
        "seed_witness_ids": list(min(outputs)),
        "witness_sha256": digest.hexdigest(),
        "enumeration_semantics": "All five original occurrence IDs identify each witness",
    }


def duckdb_raw_oracle(
    raw_dir: Path, query: Mapping[str, Any], duckdb_python: Path, temp_dir: Path
) -> Dict[str, Any]:
    """Run an independent raw-CSV SQL oracle in a separate DuckDB process."""
    temp_dir.mkdir(parents=True, exist_ok=True)
    helper = temp_dir / "job_duckdb_oracle.py"
    output = temp_dir / "job_duckdb_oracle.json"
    script = r'''import duckdb, json, pathlib, sys
raw = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
con = duckdb.connect(":memory:")
con.execute("SET threads=1")
con.execute("SET memory_limit='6GB'")
common = "header=false, delim=',', quote='\"', escape='\\', nullstr='', allow_quoted_nulls=false, strict_mode=true"
schemas = {
  "company_type": "{'id':'BIGINT','kind':'VARCHAR'}",
  "info_type": "{'id':'BIGINT','info':'VARCHAR'}",
  "movie_companies": "{'id':'BIGINT','movie_id':'BIGINT','company_id':'BIGINT','company_type_id':'BIGINT','note':'VARCHAR'}",
  "movie_info_idx": "{'id':'BIGINT','movie_id':'BIGINT','info_type_id':'BIGINT','info':'VARCHAR','note':'VARCHAR'}",
  "title": "{'id':'BIGINT','title':'VARCHAR','imdb_index':'VARCHAR','kind_id':'BIGINT','production_year':'BIGINT','imdb_id':'BIGINT','phonetic_code':'VARCHAR','episode_of_id':'BIGINT','season_nr':'BIGINT','episode_nr':'BIGINT','series_years':'VARCHAR','md5sum':'VARCHAR'}"
}
for table, columns in schemas.items():
    path = str(raw / (table + ".csv")).replace("'", "''")
    con.execute(f"CREATE VIEW {table} AS SELECT * FROM read_csv('{path}', columns={columns}, {common})")
sql = """SELECT COUNT(*) FROM company_type AS ct, info_type AS it, movie_companies AS mc, movie_info_idx AS mi_idx, title AS t WHERE ct.kind = 'production companies' AND it.info = 'top 250 rank' AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%' AND (mc.note LIKE '%(co-production)%' OR mc.note LIKE '%(presents)%') AND ct.id = mc.company_type_id AND t.id = mc.movie_id AND t.id = mi_idx.movie_id AND mc.movie_id = mi_idx.movie_id AND it.id = mi_idx.info_type_id"""
nulls = {}
for table, column, is_text in [("movie_companies","note",True),("movie_info_idx","note",True),("title","production_year",False)]:
    if is_text:
        null_count, empty_count = con.execute(f"SELECT count(*) FILTER (WHERE {column} IS NULL), count(*) FILTER (WHERE {column} = '') FROM {table}").fetchone()
    else:
        null_count = con.execute(f"SELECT count(*) FILTER (WHERE {column} IS NULL) FROM {table}").fetchone()[0]
        empty_count = 0
    nulls[f"{table}.{column}"] = {"null_count": null_count, "empty_string_count": empty_count}
payload = {"duckdb_version": duckdb.__version__, "count": con.execute(sql).fetchone()[0], "null_audit": nulls,
           "csv_options": {"quote": "double quote", "escape_within_quote": "backslash", "nullstr": "", "allow_quoted_nulls": False, "strict_mode": True}}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
'''
    helper.write_text(script, encoding="utf-8")
    subprocess.run(
        [str(duckdb_python), str(helper), str(raw_dir), str(output)],
        check=True, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
    )
    return load_json(output)


def hash_files(root: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append({
            "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def read_case_order(query_dir: Path) -> List[Dict[str, Any]]:
    lines = (query_dir / "cases.tsv").read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    return [dict({"index": index}, **dict(zip(header, line.split("\t")))) for index, line in enumerate(lines[1:])]


def finalize_manifests(query_dir: Path, validation_out: Path, manifest_out: Path) -> Dict[str, Any]:
    validation = load_json(validation_out)
    case_order = read_case_order(query_dir)
    prepared_files = hash_files(query_dir)
    validation["prepared_root"] = str(query_dir)
    validation["case_order"] = case_order
    validation["prepared_files"] = prepared_files
    validation_out.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "format": "job-prepared-manifest-v1",
        "preparation_version": PREPARATION_VERSION,
        "query": validation["query"],
        "prepared_root": str(query_dir),
        "validation_path": str(validation_out),
        "source_inventory_sha256": validation["source_inventory_sha256"],
        "case_order": case_order,
        "prepared_files": prepared_files,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def enumerate_prepared_witnesses(query_dir: Path) -> List[Tuple[int, int, int, int, int]]:
    def rows(rid: int) -> Iterable[List[str]]:
        with (query_dir / "keys" / f"r{rid}.tsv").open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
            expected = ["tuple_id"] + [f"edge_{eid}_key" for eid in ([0] if rid == 0 else [0, 1] if rid == 1 else [1, 2] if rid == 2 else [2, 3] if rid == 3 else [3])]
            if header != expected:
                raise ValueError(f"r{rid} key header drift: {header} != {expected}")
            for line in handle:
                yield line.rstrip("\n").split("\t")

    ct = next(iter(rows(0)))
    it = next(iter(rows(4)))
    ct_id, ct_key = int(ct[0]), int(ct[1])
    it_id, it_key = int(it[0]), int(it[1])
    mc_by_movie: Dict[int, List[int]] = {}
    for row in rows(1):
        if int(row[1]) == ct_key:
            mc_by_movie.setdefault(int(row[2]), []).append(int(row[0]))
    mi_by_movie: Dict[int, List[int]] = {}
    for row in rows(3):
        if int(row[2]) == it_key:
            mi_by_movie.setdefault(int(row[1]), []).append(int(row[0]))
    outputs: List[Tuple[int, int, int, int, int]] = []
    for row in rows(2):
        title_id, left_key, right_key = int(row[0]), int(row[1]), int(row[2])
        for mc_id in mc_by_movie.get(left_key, []):
            for mi_id in mi_by_movie.get(right_key, []):
                outputs.append((ct_id, mc_id, title_id, mi_id, it_id))
    outputs.sort()
    if len(outputs) != len(set(outputs)):
        raise AssertionError("prepared witness enumeration produced duplicate occurrence-ID vectors")
    return outputs


def verify_serialized_moments(query_dir: Path) -> Dict[str, Any]:
    getcontext().prec = 80
    witnesses = enumerate_prepared_witnesses(query_dir)
    case_order = read_case_order(query_dir)
    needed = [set(output[rid] for output in witnesses) for rid in range(5)]
    fields = ("actual_mu", "sum_p2", "max_p")
    results = []
    maxima = {field: {"absolute": Decimal(0), "relative": Decimal(0)} for field in fields}
    for case in case_order:
        captured: List[Dict[int, float]] = []
        for rid in range(5):
            values: Dict[int, float] = {}
            path = query_dir / case["weight_set"] / f"r{rid}.tsv"
            with path.open("r", encoding="utf-8") as handle:
                header = handle.readline().rstrip("\n")
                if header != "tuple_id\tweight":
                    raise ValueError(f"{path}: header drift")
                for line in handle:
                    tuple_text, weight_text = line.rstrip("\n").split("\t")
                    tuple_id = int(tuple_text)
                    if tuple_id in needed[rid]:
                        values[tuple_id] = float(weight_text)
            if set(values) != needed[rid]:
                raise AssertionError(f"case {case['index']} relation {rid}: missing needed tuple weights")
            captured.append(values)
        probabilities: List[Decimal] = []
        for output in witnesses:
            probability = Decimal(1)
            for rid, tuple_id in enumerate(output):
                probability *= Decimal.from_float(captured[rid][tuple_id])
            probabilities.append(probability)
        actual = {
            "actual_mu": sum(probabilities, Decimal(0)),
            "sum_p2": sum((value * value for value in probabilities), Decimal(0)),
            "max_p": max(probabilities, default=Decimal(0)),
        }
        differences = {}
        for field in fields:
            recorded = Decimal(case[field])
            absolute = abs(actual[field] - recorded)
            relative = absolute / abs(actual[field]) if actual[field] != 0 else absolute
            maxima[field]["absolute"] = max(maxima[field]["absolute"], absolute)
            maxima[field]["relative"] = max(maxima[field]["relative"], relative)
            differences[field] = {
                "recorded": case[field], "recomputed_exact_from_binary64": str(actual[field]),
                "absolute_difference": str(absolute), "relative_difference": str(relative),
            }
        results.append({
            "index": int(case["index"]), "mode": case["mode"],
            "requested_targets": case["requested_targets"], "weight_set": case["weight_set"],
            "witness_count": len(witnesses), "differences": differences,
        })
    digest = hashlib.sha256()
    for output in witnesses:
        digest.update((",".join(map(str, output)) + "\n").encode("ascii"))
    return {
        "format": "job-serialized-moment-validation-v1",
        "query": "JOB1a", "case_count": len(case_order), "witness_count": len(witnesses),
        "witness_sha256": digest.hexdigest(),
        "arithmetic": "Decimal.from_float on each parsed TSV binary64 with 80 decimal digits and products over five tuple weights",
        "max_differences": {
            field: {key: str(value) for key, value in values.items()}
            for field, values in maxima.items()
        },
        "cases": results,
    }


def main() -> None:
    args = parse_args()
    source_manifest = load_json(args.source_manifest)
    query_manifest = load_json(args.query_manifest)
    if args.stage == "oracle":
        if args.duckdb_python is None or args.duckdb_temp_dir is None:
            raise ValueError("oracle stage requires --duckdb-python and --duckdb-temp-dir")
        payload = duckdb_raw_oracle(
            args.raw_dir, query_manifest["queries"][0], args.duckdb_python, args.duckdb_temp_dir
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.stage == "finalize":
        if args.output_dir is None or args.validation_out is None or args.manifest_out is None:
            raise ValueError("finalize stage requires --output-dir, --validation-out, and --manifest-out")
        query_dir = args.output_dir / query_manifest["queries"][0]["id"]
        manifest = finalize_manifests(query_dir, args.validation_out, args.manifest_out)
        print(json.dumps({
            "case_count": len(manifest["case_order"]),
            "prepared_file_count": len(manifest["prepared_files"]),
            "manifest": str(args.manifest_out),
        }, sort_keys=True))
        return
    if args.stage == "verify-moments":
        if args.output_dir is None or args.moment_validation_out is None:
            raise ValueError("verify-moments requires --output-dir and --moment-validation-out")
        query_dir = args.output_dir / query_manifest["queries"][0]["id"]
        payload = verify_serialized_moments(query_dir)
        args.moment_validation_out.parent.mkdir(parents=True, exist_ok=True)
        args.moment_validation_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "case_count": payload["case_count"], "witness_count": payload["witness_count"],
            "max_differences": payload["max_differences"],
            "output": str(args.moment_validation_out),
        }, sort_keys=True))
        return
    result = inventory(args.raw_dir, source_manifest, query_manifest)
    args.inventory_out.parent.mkdir(parents=True, exist_ok=True)
    with args.inventory_out.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.stage == "inventory":
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.output_dir is None or args.validation_out is None:
        raise ValueError("prepare stage requires --output-dir and --validation-out")
    if args.duckdb_python is None or args.duckdb_temp_dir is None:
        raise ValueError("prepare stage requires --duckdb-python and --duckdb-temp-dir")
    protocol = query_manifest["probability_protocol"]
    if np.longdouble(str(protocol["root_near_tie_relative_delta"])) != ROOT_NEAR_TIE_RELATIVE_DELTA:
        raise ValueError("root near-tie rule drift")
    if protocol["row_domain"] != "original local filters with all surviving source occurrences retained":
        raise ValueError("row-domain convention drift")

    query = query_manifest["queries"][0]
    query_dir = args.output_dir / query["id"]
    if query_dir.exists():
        shutil.rmtree(query_dir)
    query_dir.mkdir(parents=True)
    stores, relation_stats = build_prepared_rows(args.raw_dir, source_manifest, query_manifest)
    cardinalities = intern_edge_keys(stores, query)
    write_shared_tsv(query_dir, stores, query, cardinalities)
    exact_count = exact_tree_count(stores, cardinalities)
    tiny_check = tiny_witness_check(stores)

    all_modes = build_weight_modes(stores, query_manifest)
    targets = [float(value) for value in protocol["targets"]]
    cases: List[List[Any]] = []
    root_score_rows: List[List[Any]] = []
    modes_result: Dict[str, Any] = {}
    anchor = int(query["anchor_relation"])
    for mode_name in protocol["modes"]:
        weights = all_modes[mode_name]
        write_weight_set(query_dir, Path("weights") / mode_name / "unscaled", stores, weights)
        metrics = all_root_metrics_compact(stores, query, cardinalities, weights, exact_count)
        roots = metrics["roots"]
        chosen_root = int(metrics["chosen_root"])
        chosen = roots[chosen_root]
        mu0 = chosen["mu"]
        for item in roots:
            root_score_rows.append([
                mode_name, item["root"], format_ld(item["lambda"]), format_ld(item["kappa"]),
                format_ld(item["mu"]), int(item["root"] == metrics["raw_min_root"]),
                int(item["root"] in metrics["near_tie_eligible_roots"]),
                int(item["root"] == chosen_root),
            ])
        scale_groups: Dict[float, List[float]] = {}
        for target in targets:
            scale = 0.0 if mu0 <= 0 else min(1.0, target / float(mu0))
            scale_groups.setdefault(scale, []).append(target)
        weight_sets = []
        for scale_index, (scale, requested) in enumerate(sorted(scale_groups.items())):
            scaled = list(weights)
            scaled[anchor] = weights[anchor] * scale
            relative = Path("weights") / mode_name / f"scale-{scale_index:03d}"
            write_weight_set(query_dir, relative, stores, scaled)
            actual_mu = mu0 * np.longdouble(scale)
            actual_sum_p2 = chosen["sum_p2"] * np.longdouble(scale) ** 2
            actual_max = chosen["max_p"] * np.longdouble(scale)
            requested_label = ",".join(format(value, ".17g") for value in requested)
            cases.append([
                mode_name, requested_label, relative.as_posix(), exact_count, format(scale, ".17g"),
                format_ld(mu0), format_ld(actual_mu), format_ld(actual_sum_p2),
                format_ld(actual_max), chosen_root,
            ])
            weight_sets.append({
                "requested_targets": requested, "weight_set": relative.as_posix(),
                "a": format(scale, ".17g"), "actual_mu": format_ld(actual_mu),
                "sum_p2": format_ld(actual_sum_p2), "max_p": format_ld(actual_max),
            })
        modes_result[mode_name] = {
            "join_count": str(exact_count), "raw_min_root": metrics["raw_min_root"],
            "near_tie_eligible_roots": metrics["near_tie_eligible_roots"],
            "chosen_root": chosen_root, "mu0": format_ld(mu0),
            "root_scores": [{
                "root": item["root"], "lambda": format_ld(item["lambda"]),
                "kappa": format_ld(item["kappa"]), "mu": format_ld(item["mu"]),
                "max_p": format_ld(item["max_p"]), "sum_p2": format_ld(item["sum_p2"]),
                "raw_min": item["root"] == metrics["raw_min_root"],
                "near_tie_eligible": item["root"] in metrics["near_tie_eligible_roots"],
                "guarded_chosen": item["root"] == chosen_root,
            } for item in roots],
            "weight_sets": weight_sets,
        }

    write_numeric_tsv(
        query_dir / "cases.tsv",
        ["mode", "requested_targets", "weight_set", "join_size", "a", "mu0", "actual_mu", "sum_p2", "max_p", "chosen_root"],
        cases,
    )
    write_numeric_tsv(
        query_dir / "root_scores.tsv",
        ["mode", "root", "lambda", "kappa", "mu", "raw_min", "near_tie_eligible", "guarded_chosen"],
        root_score_rows,
    )

    # The raw oracle reparses the original CSVs using DuckDB, with its default
    # quoted-NULL behavior explicitly disabled to match PostgreSQL COPY CSV.
    oracle = duckdb_raw_oracle(args.raw_dir, query, args.duckdb_python, args.duckdb_temp_dir)
    if int(oracle["count"]) != exact_count:
        raise AssertionError(f"DuckDB raw SQL {oracle['count']} != tree DP {exact_count}")
    expected_nulls = {
        "movie_companies.note": next(x for x in result["tables"] if x["table"] == "movie_companies")["null_counts"]["note"],
        "movie_info_idx.note": next(x for x in result["tables"] if x["table"] == "movie_info_idx")["null_counts"]["note"],
        "title.production_year": next(x for x in result["tables"] if x["table"] == "title")["null_counts"]["production_year"],
    }
    for field, expected in expected_nulls.items():
        observed = int(oracle["null_audit"][field]["null_count"])
        empty_count = int(oracle["null_audit"][field]["empty_string_count"])
        if observed != expected or empty_count != 0:
            raise AssertionError(
                f"DuckDB NULL audit {field}: null={observed}, empty={empty_count}, expected null={expected}"
            )

    validation = {
        "format": "job-loader-validation-v1",
        "preparation_version": PREPARATION_VERSION,
        "source_inventory_sha256": sha256_file(args.inventory_out),
        "source_archive": source_manifest["archive"],
        "query": query["id"],
        "query_source": source_manifest["query_source"],
        "row_domain": protocol["row_domain"],
        "csv_semantics": {
            "quote": "double quote", "escape": "backslash only inside quoted fields",
            "outside_quote_backslash": "literal", "null": "unquoted empty field",
            "quoted_empty": "non-NULL empty string",
            "duckdb_allow_quoted_nulls": False,
        },
        "feature_rules": {
            "epsilon": protocol["epsilon"],
            "title_year": "eps+(1-eps)*(year-1880)/(2019-1880). Source NULL/invalid => eps",
            "top_rank": "only info_type_id=112 positive integral info: eps+(1-eps)/rank. All other rows => eps",
            "rank_valid_rows": int(np.count_nonzero(stores[3]["top_rank"] > 0)),
            "shuffle_seed_formula": protocol["shuffle_seed_formula"],
            "shuffle_seed_base_hex": protocol["shuffle_seed_base_hex"],
            "scale": "Title relation weights scale downward to targets 1/100/10000. Identical a=1 cases deduplicate",
        },
        "relation_stats": relation_stats,
        "edge_key_cardinalities": {str(key): value for key, value in cardinalities.items()},
        "tree_dp_count": str(exact_count),
        "duckdb_raw_sql_count": str(oracle["count"]),
        "duckdb_tree_dp_match": True,
        "duckdb_version": oracle["duckdb_version"],
        "duckdb_null_audit": oracle["null_audit"],
        "tiny_induced": tiny_check,
        "root_near_tie_rule": {
            "relative_delta": "1e-12",
            "positive_minimum": "eligible iff Lambda <= Lmin*(1+delta), then smallest manifest relation ID",
            "zero_minimum": "eligible iff Lambda == 0, then smallest manifest relation ID",
            "status": "deterministic practical guard, not a certified exact-minimizer error bound",
        },
        "modes": modes_result,
    }
    validation["prepared_root"] = str(query_dir)
    validation["case_order"] = [
        dict({"index": index}, **dict(zip(
            ["mode", "requested_targets", "weight_set", "join_size", "a", "mu0", "actual_mu", "sum_p2", "max_p", "chosen_root"],
            [str(value) for value in row],
        )))
        for index, row in enumerate(cases)
    ]
    validation["prepared_files"] = hash_files(query_dir)
    args.validation_out.parent.mkdir(parents=True, exist_ok=True)
    args.validation_out.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.manifest_out is not None:
        finalize_manifests(query_dir, args.validation_out, args.manifest_out)
    print(json.dumps({
        "query": query["id"], "tree_dp_count": exact_count, "duckdb_count": oracle["count"],
        "prepared_file_count": len(validation["prepared_files"]),
        "validation": str(args.validation_out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
