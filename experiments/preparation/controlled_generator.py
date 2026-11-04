#!/usr/bin/env python3
"""Controlled acyclic-join data generation and independent preparation validation.

Writes strict stats-undirected-v1 TSV inputs for the formal driver.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import csv
import gc
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


FORMAT = "stats-undirected-v1"
PREPARATION_VERSION = "controlled-scaling-preparation-v1"
ROOT_NEAR_TIE_RELATIVE_DELTA = np.longdouble("1e-12")
WEIGHT_SEED_BASE = 0x0929272000000000
SEED_GOLDEN_XOR = 0x9E3779B97F4A7C15
SCAN_MAX_J = 1_000_000
RANK_MAX_J = 1 << 53
TARGET_MU = 100.0


_LIBM_NAME = ctypes.util.find_library("m")
_LIBM = ctypes.CDLL(_LIBM_NAME) if _LIBM_NAME else None
if _LIBM is not None and hasattr(_LIBM, "exp2"):
    _C_EXP2 = _LIBM.exp2
    _C_EXP2.argtypes = [ctypes.c_double]
    _C_EXP2.restype = ctypes.c_double
else:
    _C_EXP2 = None


def exp2(value: float) -> float:
    # On the Linux preparation host this calls the same libm exp2 entry point
    # used by the C++17 pilot.  The fallback keeps the self-test portable.
    return float(_C_EXP2(value)) if _C_EXP2 is not None else 2.0 ** value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hashed_file(path: Path, relative_to: Optional[Path] = None) -> Dict[str, Any]:
    resolved = path.resolve(strict=True)
    label = str(resolved if relative_to is None else resolved.relative_to(relative_to.resolve()))
    return {"path": label, "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def write_tsv(path: Path, header: Sequence[Any], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def format_ld(value: np.longdouble) -> str:
    return np.format_float_scientific(value, precision=21, unique=False, trim="k")


class MT19937_64:
    """Small direct implementation of std::mt19937_64's specified engine."""

    N = 312
    M = 156
    MASK = (1 << 64) - 1
    LOWER = (1 << 31) - 1
    UPPER = MASK ^ LOWER

    def __init__(self, seed: int):
        self.state = [0] * self.N
        self.state[0] = seed & self.MASK
        for index in range(1, self.N):
            previous = self.state[index - 1]
            self.state[index] = (
                6364136223846793005 * (previous ^ (previous >> 62)) + index
            ) & self.MASK
        self.index = self.N

    def _twist(self) -> None:
        for index in range(self.N):
            value = (self.state[index] & self.UPPER) | (
                self.state[(index + 1) % self.N] & self.LOWER
            )
            shifted = value >> 1
            if value & 1:
                shifted ^= 0xB5026F5AA96619E9
            self.state[index] = self.state[(index + self.M) % self.N] ^ shifted
        self.index = 0

    def next_u64(self) -> int:
        if self.index >= self.N:
            self._twist()
        value = self.state[self.index]
        self.index += 1
        value ^= (value >> 29) & 0x5555555555555555
        value ^= (value << 17) & 0x71D67FFFEDA60000
        value ^= (value << 37) & 0xFFF7EEE000000000
        value ^= value >> 43
        return value & self.MASK

    def uniform_open(self) -> float:
        # Exact 52-bit midpoints, matching the C++ reference's strict (0,1) law.
        return ((self.next_u64() >> 12) + 0.5) * (2.0 ** -52)


@dataclass(frozen=True)
class Edge:
    edge_id: int
    left: int
    right: int


@dataclass(frozen=True)
class Row:
    tuple_id: int
    keys: Tuple[int, ...]


@dataclass
class Topology:
    name: str
    edges: List[Edge]
    incident: List[List[int]]
    rows: List[List[Row]]

    @property
    def m(self) -> int:
        return len(self.rows)

    @property
    def n(self) -> int:
        return sum(len(relation) for relation in self.rows)

    def key(self, relation: int, row: Row, edge_id: int) -> int:
        return row.keys[self.incident[relation].index(edge_id)]


def incident_from_edges(m: int, edges: Sequence[Edge]) -> List[List[int]]:
    incident: List[List[int]] = [[] for _ in range(m)]
    for edge in edges:
        incident[edge.left].append(edge.edge_id)
        incident[edge.right].append(edge.edge_id)
    for values in incident:
        values.sort()
    return incident


def reused_separator_topology(shape: str, m: int, key_count: int, degree: int) -> Topology:
    if m not in (4, 6):
        raise ValueError("controlled reused-separator topology supports only m=4 or m=6")
    if shape not in ("chain", "branch") or key_count < 1 or degree < 1:
        raise ValueError("invalid reused-separator topology")
    if shape == "chain":
        edges = [Edge(v - 1, v - 1, v) for v in range(1, m)]
    else:
        edges = [Edge(v - 1, 0, v) for v in range(1, m)]
    incident = incident_from_edges(m, edges)
    rows: List[List[Row]] = [[] for _ in range(m)]
    next_id = 0

    def add(relation: int, keys_by_edge: Mapping[int, int]) -> None:
        nonlocal next_id
        rows[relation].append(Row(next_id, tuple(keys_by_edge[e] for e in incident[relation])))
        next_id += 1

    if shape == "chain":
        for key in range(key_count):
            for _ in range(degree):
                add(0, {0: key})
        for relation in range(1, m):
            parent_edge = relation - 1
            child_edge = relation if relation + 1 < m else None
            for parent_key in range(key_count):
                for local in range(degree):
                    values = {parent_edge: parent_key}
                    if child_edge is not None:
                        values[child_edge] = (
                            parent_key + local * (2 * relation + 1)
                        ) % key_count
                    add(relation, values)
    else:
        for anchor in range(key_count):
            for local in range(degree):
                add(
                    0,
                    {
                        child - 1: (
                            anchor + local * (2 * child + 1) + 7 * child
                        ) % key_count
                        for child in range(1, m)
                    },
                )
        for child in range(1, m):
            for key in range(key_count):
                for _ in range(degree):
                    add(child, {child - 1: key})
    return Topology(f"reused-{shape}-m{m}-k{key_count}-d{degree}", edges, incident, rows)


def triangular_topology(key_count: int) -> Topology:
    if key_count < 2:
        raise ValueError("triangular topology requires K>=2")
    edges = [Edge(0, 0, 1), Edge(1, 1, 2), Edge(2, 2, 3)]
    incident = incident_from_edges(4, edges)
    rows: List[List[Row]] = [[] for _ in range(4)]
    next_id = 0

    def add(relation: int, values: Mapping[int, int]) -> None:
        nonlocal next_id
        rows[relation].append(Row(next_id, tuple(values[e] for e in incident[relation])))
        next_id += 1

    for x in range(key_count):
        add(0, {0: x})
    for x in range(key_count):
        for y in range(x, key_count):
            add(1, {0: x, 1: y})
    for y in range(key_count):
        for z in range(y, key_count):
            add(2, {1: y, 2: z})
    for z in range(key_count):
        add(3, {2: z})
    return Topology(f"triangular-path-k{key_count}", edges, incident, rows)


def neighbors(topology: Topology) -> List[List[Tuple[int, int]]]:
    result: List[List[Tuple[int, int]]] = [[] for _ in range(topology.m)]
    for edge in topology.edges:
        result[edge.left].append((edge.right, edge.edge_id))
        result[edge.right].append((edge.left, edge.edge_id))
    for values in result:
        values.sort()
    return result


Metric = Tuple[int, np.longdouble, np.longdouble, np.longdouble]


def all_root_metrics(topology: Topology, weights: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Independent count/Z/max/Z2 message DP over the undirected tree."""

    adjacent = neighbors(topology)
    edge_positions = [
        {edge_id: position for position, edge_id in enumerate(values)}
        for values in topology.incident
    ]
    memo: Dict[Tuple[int, int], Dict[int, Metric]] = {}

    def key_of(node: int, row: Row, edge_id: int) -> int:
        return row.keys[edge_positions[node][edge_id]]

    def message(node: int, excluded: int) -> Dict[int, Metric]:
        cache_key = (node, excluded)
        if cache_key in memo:
            return memo[cache_key]
        separator = next(edge for other, edge in adjacent[node] if other == excluded)
        grouped: Dict[int, List[Any]] = {}
        for row_index, row in enumerate(topology.rows[node]):
            count = 1
            z = np.longdouble(weights[node][row_index])
            z2 = z * z
            maximum = z
            for other, edge_id in adjacent[node]:
                if other == excluded:
                    continue
                incoming = message(other, node).get(key_of(node, row, edge_id))
                if incoming is None:
                    count = 0
                    z = z2 = maximum = np.longdouble(0)
                    break
                count *= incoming[0]
                z *= incoming[1]
                maximum *= incoming[2]
                z2 *= incoming[3]
            group = key_of(node, row, separator)
            aggregate = grouped.setdefault(
                group, [0, np.longdouble(0), np.longdouble(0), np.longdouble(0)]
            )
            aggregate[0] += count
            aggregate[1] += z
            if maximum > aggregate[2]:
                aggregate[2] = maximum
            aggregate[3] += z2
        result = {key: (x[0], x[1], x[2], x[3]) for key, x in grouped.items()}
        memo[cache_key] = result
        return result

    roots: List[Dict[str, Any]] = []
    join_counts = set()
    for root in range(topology.m):
        total_count = 0
        mu = np.longdouble(0)
        maximum = np.longdouble(0)
        sum_p2 = np.longdouble(0)
        envelope = np.longdouble(0)
        for row_index, row in enumerate(topology.rows[root]):
            count = 1
            weight = np.longdouble(weights[root][row_index])
            z = weight
            z2 = weight * weight
            row_max = weight
            for other, edge_id in adjacent[root]:
                incoming = message(other, root).get(key_of(root, row, edge_id))
                if incoming is None:
                    count = 0
                    z = z2 = row_max = np.longdouble(0)
                    break
                count *= incoming[0]
                z *= incoming[1]
                row_max *= incoming[2]
                z2 *= incoming[3]
            total_count += count
            mu += z
            maximum = max(maximum, row_max)
            sum_p2 += z2
            envelope += np.longdouble(count) * row_max
        join_counts.add(total_count)
        roots.append(
            {
                "root": root,
                "join_count": total_count,
                "mu": mu,
                "max_p": maximum,
                "sum_p2": sum_p2,
                "lambda": envelope,
                "kappa": envelope / mu if mu > 0 else np.longdouble(0),
            }
        )
    if len(join_counts) != 1:
        raise AssertionError(f"root-dependent join counts: {join_counts}")
    raw_min = min(roots, key=lambda item: (item["lambda"], item["root"]))
    minimum = raw_min["lambda"]
    ceiling = minimum * (np.longdouble(1) + ROOT_NEAR_TIE_RELATIVE_DELTA)
    eligible = [item["root"] for item in roots if item["lambda"] <= ceiling]
    return {
        "join_count": next(iter(join_counts)),
        "roots": roots,
        "raw_min_root": raw_min["root"],
        "near_tie_eligible_roots": eligible,
        "chosen_root": min(eligible),
    }


def root_mu(topology: Topology, weights: Sequence[Sequence[float]], root: int = 0) -> np.longdouble:
    adjacent = neighbors(topology)
    positions = [
        {edge_id: position for position, edge_id in enumerate(values)}
        for values in topology.incident
    ]
    memo: Dict[Tuple[int, int], Dict[int, np.longdouble]] = {}

    def key_of(node: int, row: Row, edge_id: int) -> int:
        return row.keys[positions[node][edge_id]]

    def message(node: int, parent: int) -> Dict[int, np.longdouble]:
        cache_key = (node, parent)
        if cache_key in memo:
            return memo[cache_key]
        separator = next(edge for other, edge in adjacent[node] if other == parent)
        grouped: Dict[int, np.longdouble] = {}
        for index, row in enumerate(topology.rows[node]):
            value = np.longdouble(weights[node][index])
            for child, edge_id in adjacent[node]:
                if child == parent:
                    continue
                value *= message(child, node).get(key_of(node, row, edge_id), np.longdouble(0))
            group = key_of(node, row, separator)
            grouped[group] = grouped.get(group, np.longdouble(0)) + value
        memo[cache_key] = grouped
        return grouped

    total = np.longdouble(0)
    for index, row in enumerate(topology.rows[root]):
        value = np.longdouble(weights[root][index])
        for child, edge_id in adjacent[root]:
            value *= message(child, root).get(key_of(root, row, edge_id), np.longdouble(0))
        total += value
    return total


def independent_weights(topology: Topology, seed: int) -> List[List[float]]:
    rng = MT19937_64(seed ^ SEED_GOLDEN_XOR)
    return [
        [exp2(-3.0 * rng.uniform_open()) for _ in relation]
        for relation in topology.rows
    ]


def triangular_weights(topology: Topology, key_count: int, scale_a: float) -> List[List[float]]:
    epsilon = key_count ** -2
    result = [[1.0 for _ in relation] for relation in topology.rows]
    for index in range(key_count):
        base = 1.0 if index == 0 else epsilon
        result[0][index] = float(base * scale_a)
        result[3][index] = 1.0 if index == key_count - 1 else epsilon
    return result


def dense_weights(topology: Topology, p: float) -> List[List[float]]:
    result = [[1.0 for _ in relation] for relation in topology.rows]
    result[0] = [p for _ in topology.rows[0]]
    return result


def write_topology(case_dir: Path, topology: Topology) -> None:
    lines = [
        FORMAT,
        f"query\t{topology.name}",
        f"m\t{topology.m}",
        "verification_root\t0",
    ]
    for relation, rows in enumerate(topology.rows):
        lines.append(
            "\t".join(
                [
                    "relation",
                    str(relation),
                    f"r{relation}",
                    f"synthetic_r{relation}",
                    str(len(rows)),
                    f"keys/r{relation}.tsv",
                    f"features/r{relation}.tsv",
                ]
            )
        )
    for edge in topology.edges:
        keys = {
            topology.key(edge.left, row, edge.edge_id)
            for row in topology.rows[edge.left]
        }
        lines.append(
            "\t".join(
                [
                    "edge",
                    str(edge.edge_id),
                    str(edge.left),
                    str(edge.right),
                    str(len(keys)),
                    f"e{edge.edge_id}",
                    f"e{edge.edge_id}",
                ]
            )
        )
    (case_dir / "structure.tsv").parent.mkdir(parents=True, exist_ok=True)
    (case_dir / "structure.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for relation, rows in enumerate(topology.rows):
        write_tsv(
            case_dir / "keys" / f"r{relation}.tsv",
            ["tuple_id"] + [f"edge_{edge}_key" for edge in topology.incident[relation]],
            ([row.tuple_id, *row.keys] for row in rows),
        )
        write_tsv(
            case_dir / "features" / f"r{relation}.tsv",
            ["tuple_id"],
            ([row.tuple_id] for row in rows),
        )


def write_weights(case_dir: Path, relative: Path, topology: Topology,
                  weights: Sequence[Sequence[float]]) -> None:
    for relation, rows in enumerate(topology.rows):
        write_tsv(
            case_dir / relative / f"r{relation}.tsv",
            ["tuple_id", "weight"],
            (
                (row.tuple_id, format(weights[relation][index], ".17g"))
                for index, row in enumerate(rows)
            ),
        )


def read_prepared(case_dir: Path, relative: Path) -> Tuple[Topology, List[List[float]]]:
    structure = (case_dir / "structure.tsv").read_text(encoding="utf-8").splitlines()
    if not structure or structure[0] != FORMAT:
        raise ValueError("prepared structure format mismatch")
    name = ""
    m = -1
    relation_rows: Dict[int, int] = {}
    key_paths: Dict[int, str] = {}
    edges: List[Edge] = []
    for line in structure[1:]:
        fields = line.split("\t")
        if fields[0] == "query":
            name = fields[1]
        elif fields[0] == "m":
            m = int(fields[1])
        elif fields[0] == "relation":
            relation_rows[int(fields[1])] = int(fields[4])
            key_paths[int(fields[1])] = fields[5]
        elif fields[0] == "edge":
            edges.append(Edge(int(fields[1]), int(fields[2]), int(fields[3])))
    if m <= 0 or len(relation_rows) != m or len(edges) != m - 1:
        raise ValueError("incomplete prepared structure")
    incident = incident_from_edges(m, edges)
    rows: List[List[Row]] = [[] for _ in range(m)]
    weights: List[List[float]] = [[] for _ in range(m)]
    seen_ids = set()
    for relation in range(m):
        with (case_dir / key_paths[relation]).open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            expected = ["tuple_id"] + [f"edge_{edge}_key" for edge in incident[relation]]
            if next(reader) != expected:
                raise ValueError("prepared key header mismatch")
            for fields in reader:
                tuple_id = int(fields[0])
                if tuple_id in seen_ids:
                    raise ValueError("prepared tuple IDs are not globally unique")
                seen_ids.add(tuple_id)
                rows[relation].append(Row(tuple_id, tuple(int(value) for value in fields[1:])))
        with (case_dir / relative / f"r{relation}.tsv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.reader(stream, delimiter="\t")
            if next(reader) != ["tuple_id", "weight"]:
                raise ValueError("prepared weight header mismatch")
            for index, fields in enumerate(reader):
                if int(fields[0]) != rows[relation][index].tuple_id:
                    raise ValueError("prepared key/weight tuple identity mismatch")
                value = float(fields[1])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError("invalid serialized probability")
                weights[relation].append(value)
        if len(rows[relation]) != relation_rows[relation] or len(weights[relation]) != len(rows[relation]):
            raise ValueError("prepared row count mismatch")
    return Topology(name, edges, incident, rows), weights


def enumerate_witnesses(topology: Topology, weights: Sequence[Sequence[float]],
                        root: int = 0) -> Dict[Tuple[int, ...], float]:
    adjacent = neighbors(topology)
    positions = [
        {edge_id: position for position, edge_id in enumerate(values)}
        for values in topology.incident
    ]
    parent = [-2] * topology.m
    parent_edge = [-1] * topology.m
    parent[root] = -1
    order = [root]
    for node in order:
        for other, edge_id in adjacent[node]:
            if parent[other] == -2:
                parent[other] = node
                parent_edge[other] = edge_id
                order.append(other)
    lookups: Dict[int, Dict[int, List[int]]] = {}
    for node in order[1:]:
        grouped: Dict[int, List[int]] = {}
        edge_id = parent_edge[node]
        for index, row in enumerate(topology.rows[node]):
            grouped.setdefault(row.keys[positions[node][edge_id]], []).append(index)
        lookups[node] = grouped

    def subtree(node: int, row_index: int) -> List[Tuple[Dict[int, int], float]]:
        row = topology.rows[node][row_index]
        partials: List[Tuple[Dict[int, int], float]] = [
            ({node: row.tuple_id}, weights[node][row_index])
        ]
        for child, edge_id in adjacent[node]:
            if parent[child] != node:
                continue
            key = row.keys[positions[node][edge_id]]
            child_parts: List[Tuple[Dict[int, int], float]] = []
            for child_index in lookups[child].get(key, []):
                child_parts.extend(subtree(child, child_index))
            combined: List[Tuple[Dict[int, int], float]] = []
            for left, right in itertools.product(partials, child_parts):
                ids = dict(left[0])
                ids.update(right[0])
                combined.append((ids, left[1] * right[1]))
            partials = combined
        return partials

    result: Dict[Tuple[int, ...], float] = {}
    for row_index in range(len(topology.rows[root])):
        for ids, probability in subtree(root, row_index):
            witness = tuple(ids[relation] for relation in range(topology.m))
            if witness in result:
                raise AssertionError("duplicate occurrence witness")
            result[witness] = probability
    return result


def tiny_oracle_checks() -> List[Dict[str, Any]]:
    specs = [
        ("chain-m4", reused_separator_topology("chain", 4, 3, 2), 0x101),
        ("chain-m6", reused_separator_topology("chain", 6, 2, 2), 0x102),
        ("branch-m6", reused_separator_topology("branch", 6, 2, 2), 0x103),
    ]
    checks = []
    for label, topology, seed in specs:
        weights = independent_weights(topology, seed)
        oracle = enumerate_witnesses(topology, weights)
        metrics = all_root_metrics(topology, weights)
        values = list(oracle.values())
        expected = len(oracle)
        if metrics["join_count"] != expected:
            raise AssertionError(f"{label}: witness count mismatch")
        reference = metrics["roots"][0]
        comparisons = {
            "mu": (float(reference["mu"]), math.fsum(values)),
            "sum_p2": (float(reference["sum_p2"]), math.fsum(x * x for x in values)),
            "max_p": (float(reference["max_p"]), max(values)),
        }
        max_relative_error = 0.0
        for observed, exact in comparisons.values():
            max_relative_error = max(
                max_relative_error, abs(observed - exact) / max(abs(exact), 1e-300)
            )
        if max_relative_error > 2e-14:
            raise AssertionError(f"{label}: DP/oracle numerical mismatch")
        checks.append(
            {
                "label": label,
                "witnesses": expected,
                "unique_witness_ids": len(oracle),
                "max_relative_moment_error": max_relative_error,
            }
        )
    topology = triangular_topology(4)
    weights = triangular_weights(topology, 4, 1.0)
    oracle = enumerate_witnesses(topology, weights)
    metrics = all_root_metrics(topology, weights)
    expected = 4 * 5 * 6 // 6
    if len(oracle) != expected or metrics["join_count"] != expected:
        raise AssertionError("triangular tiny witness count mismatch")
    checks.append(
        {
            "label": "triangular-k4",
            "witnesses": expected,
            "unique_witness_ids": len(oracle),
            "max_relative_moment_error": abs(
                float(metrics["roots"][0]["mu"]) - math.fsum(oracle.values())
            ) / math.fsum(oracle.values()),
        }
    )
    return checks


def triangular_exact_from_serialized(topology: Topology, weights: Sequence[Sequence[float]],
                                      key_count: int) -> Dict[str, Any]:
    getcontext().prec = 80
    K = key_count
    J = K * (K + 1) * (K + 2) // 6
    a0 = Decimal(format(weights[0][0], ".17g"))
    ao = Decimal(format(weights[0][1], ".17g"))
    do = Decimal(format(weights[3][0], ".17g"))
    d1 = Decimal(format(weights[3][-1], ".17g"))
    if any(value != weights[0][1] for value in weights[0][1:]):
        raise AssertionError("triangular A non-special weights differ")
    if any(value != weights[3][0] for value in weights[3][:-1]):
        raise AssertionError("triangular D non-special weights differ")
    half = K * (K - 1) // 2
    low = J - K * K
    classes = [(K, a0 * d1), (half, a0 * do), (half, ao * d1), (low, ao * do)]
    mu = sum(Decimal(count) * probability for count, probability in classes)
    sum_p2 = sum(Decimal(count) * probability * probability for count, probability in classes)
    maximum = max(probability for count, probability in classes if count)
    heavy = sum(count for count, probability in classes if probability > Decimal("0.5"))
    return {
        "mu": str(mu),
        "sum_p2": str(sum_p2),
        "max_p": str(maximum),
        "heavy_outputs": heavy,
        "class_counts": [count for count, _ in classes],
        "class_probabilities": [str(probability) for _, probability in classes],
    }


def close_decimal_to_ld(decimal_text: str, observed: np.longdouble,
                        relative: np.longdouble = np.longdouble("2e-15")) -> bool:
    expected = np.longdouble(decimal_text)
    return abs(observed - expected) <= relative * max(np.longdouble(1), abs(expected))


def case_definitions() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for shape in ("chain", "branch"):
        for key_count in (128, 512, 2048, 8192, 32768):
            cases.append(
                {
                    "family": "input-size-fixed-degree",
                    "name": f"input-{shape}-m6-d5-k{key_count}",
                    "shape": shape,
                    "m": 6,
                    "degree": 5,
                    "key_count": key_count,
                    "schedule_class": "target100",
                }
            )
    for degree in (2, 4, 8, 16, 32):
        cases.append(
            {
                "family": "local-degree-fixed-input",
                "name": f"degree-chain-m4-n65536-d{degree}",
                "shape": "chain",
                "m": 4,
                "degree": degree,
                "key_count": 16384 // degree,
                "schedule_class": "target100",
            }
        )
    for key_count in (16, 32, 64, 128, 256, 512, 1024):
        cases.append(
            {
                "family": "triangular-envelope",
                "name": f"triangular-k{key_count}-unscaled",
                "key_count": key_count,
                "triangular_variant": "unscaled",
                "requested_labels": "unscaled,target100" if key_count < 128 else "unscaled",
                "schedule_class": "unscaled",
            }
        )
        if key_count >= 128:
            cases.append(
                {
                    "family": "triangular-envelope",
                    "name": f"triangular-k{key_count}-target100",
                    "key_count": key_count,
                    "triangular_variant": "target100",
                    "requested_labels": "target100",
                    "schedule_class": "target100",
                }
            )
    for label, probability in (
        ("1over1024", 1 / 1024),
        ("1over16", 1 / 16),
        ("1over2", 1 / 2),
        ("3over4", 3 / 4),
        ("1", 1.0),
    ):
        cases.append(
            {
                "family": "dense-uniform-scope",
                "name": f"dense-triangular-k64-p{label}",
                "key_count": 64,
                "dense_p": probability,
                "requested_labels": label,
                "schedule_class": "dense",
            }
        )
    if len(cases) != 31:
        raise AssertionError(f"controlled case count {len(cases)} !=31")
    for index, item in enumerate(cases):
        item["case_index"] = index
        item["directory"] = f"c{index:03d}_{item['name']}"
    return cases


def build_case(case: Mapping[str, Any], prepared_root: Path) -> Dict[str, Any]:
    case_index = int(case["case_index"])
    case_dir = prepared_root / str(case["directory"])
    case_dir.mkdir(parents=True, exist_ok=False)
    family = str(case["family"])
    relative = Path("weights") / "controlled" / "scale-000"
    seed: Optional[int] = None
    exact_formula: Optional[Dict[str, Any]] = None

    if family in ("input-size-fixed-degree", "local-degree-fixed-input"):
        topology = reused_separator_topology(
            str(case["shape"]), int(case["m"]), int(case["key_count"]), int(case["degree"])
        )
        expected_n = int(case["m"]) * int(case["key_count"]) * int(case["degree"])
        expected_j = int(case["key_count"]) * int(case["degree"]) ** int(case["m"])
        if topology.n != expected_n:
            raise AssertionError("reused-separator N formula mismatch")
        seed = (WEIGHT_SEED_BASE + case_index * 0x100) & ((1 << 64) - 1)
        weights = independent_weights(topology, seed)
        mu0_ld = root_mu(topology, weights)
        mu0 = float(mu0_ld)
        scale = min(1.0, TARGET_MU / mu0)
        if not 0 < scale <= 1 or mu0 < TARGET_MU:
            raise AssertionError("independent workload cannot be normalized downward to target100")
        weights[0] = [float(value * scale) for value in weights[0]]
        requested_labels = "target100"
        structural = {
            "formula_n": f"m*K*d={expected_n}",
            "formula_j": f"K*d^m={expected_j}",
            "expected_n": expected_n,
            "expected_j": expected_j,
        }
    elif family == "triangular-envelope":
        K = int(case["key_count"])
        topology = triangular_topology(K)
        expected_n = K * K + 3 * K
        expected_j = K * (K + 1) * (K + 2) // 6
        epsilon = K ** -2
        mu0 = K + epsilon * K * (K - 1) + epsilon * epsilon * (expected_j - K * K)
        if case["triangular_variant"] == "target100":
            scale = min(1.0, TARGET_MU / mu0)
            if not scale < 1.0:
                raise AssertionError("distinct triangular target100 case failed deduplication")
        else:
            scale = 1.0
        weights = triangular_weights(topology, K, scale)
        requested_labels = str(case["requested_labels"])
        structural = {
            "formula_n": f"K^2+3K={expected_n}",
            "formula_j": f"K(K+1)(K+2)/6={expected_j}",
            "expected_n": expected_n,
            "expected_j": expected_j,
            "epsilon": format(epsilon, ".17g"),
        }
    elif family == "dense-uniform-scope":
        K = int(case["key_count"])
        topology = triangular_topology(K)
        expected_n = K * K + 3 * K
        expected_j = K * (K + 1) * (K + 2) // 6
        probability = float(case["dense_p"])
        mu0 = float(expected_j)
        scale = probability
        weights = dense_weights(topology, probability)
        requested_labels = str(case["requested_labels"])
        structural = {
            "formula_n": f"K^2+3K={expected_n}",
            "formula_j": f"K(K+1)(K+2)/6={expected_j}",
            "expected_n": expected_n,
            "expected_j": expected_j,
        }
    else:
        raise ValueError(f"unknown family {family}")

    write_topology(case_dir, topology)
    write_weights(case_dir, relative, topology, weights)
    del topology, weights
    gc.collect()

    topology, serialized = read_prepared(case_dir, relative)
    metrics = all_root_metrics(topology, serialized)
    if topology.n != expected_n or metrics["join_count"] != expected_j:
        raise AssertionError("serialized topology N/J mismatch")
    root0 = metrics["roots"][0]
    for root_metric in metrics["roots"]:
        if root_metric["join_count"] != expected_j:
            raise AssertionError("all-root join cardinality mismatch")
        if abs(root_metric["mu"] - root0["mu"]) > np.longdouble("2e-12") * max(
            np.longdouble(1), abs(root0["mu"])
        ):
            raise AssertionError("all-root weighted mass mismatch")
        if abs(root_metric["sum_p2"] - root0["sum_p2"]) > np.longdouble("2e-12") * max(
            np.longdouble(1), abs(root0["sum_p2"])
        ):
            raise AssertionError("all-root second moment mismatch")
        if abs(root_metric["max_p"] - root0["max_p"]) > np.longdouble("2e-15"):
            raise AssertionError("all-root maximum probability mismatch")

    if family == "triangular-envelope":
        exact_formula = triangular_exact_from_serialized(
            topology, serialized, int(case["key_count"])
        )
        if not close_decimal_to_ld(exact_formula["mu"], root0["mu"]):
            raise AssertionError("triangular exact mu formula mismatch")
        if not close_decimal_to_ld(exact_formula["sum_p2"], root0["sum_p2"]):
            raise AssertionError("triangular exact sum_p2 formula mismatch")
        if not close_decimal_to_ld(exact_formula["max_p"], root0["max_p"]):
            raise AssertionError("triangular exact max formula mismatch")
        ideal_lambda = np.longdouble(scale) * (
            np.longdouble(int(case["key_count"]) * (int(case["key_count"]) + 1) // 2)
            + np.longdouble(int(case["key_count"]) ** -2)
            * np.longdouble(
                int(case["key_count"])
                * (int(case["key_count"]) - 1)
                * (int(case["key_count"]) + 1)
                // 6
            )
        )
        for root_metric in metrics["roots"]:
            if abs(root_metric["lambda"] - ideal_lambda) > np.longdouble("3e-12") * max(
                np.longdouble(1), abs(ideal_lambda)
            ):
                raise AssertionError("triangular all-root lambda formula mismatch")
        if metrics["near_tie_eligible_roots"] != [0, 1, 2, 3]:
            raise AssertionError("triangular roots should all satisfy near-tie guard")
    elif family == "dense-uniform-scope":
        getcontext().prec = 80
        probability = Decimal(format(serialized[0][0], ".17g"))
        exact_mu = Decimal(expected_j) * probability
        exact_sum_p2 = Decimal(expected_j) * probability * probability
        if not close_decimal_to_ld(str(exact_mu), root0["mu"]):
            raise AssertionError("dense exact mu formula mismatch")
        if not close_decimal_to_ld(str(exact_sum_p2), root0["sum_p2"]):
            raise AssertionError("dense exact sum_p2 formula mismatch")
        if metrics["near_tie_eligible_roots"] != [0, 1, 2, 3]:
            raise AssertionError("dense roots should all satisfy near-tie guard")
        exact_formula = {
            "mu": str(exact_mu),
            "sum_p2": str(exact_sum_p2),
            "max_p": str(probability),
            "heavy_outputs": expected_j if probability > Decimal("0.5") else 0,
        }

    actual_mu = root0["mu"]
    if family in ("input-size-fixed-degree", "local-degree-fixed-input"):
        relative_error = abs(actual_mu - np.longdouble(TARGET_MU)) / np.longdouble(TARGET_MU)
        if relative_error > np.longdouble("2e-14"):
            raise AssertionError(f"serialized relation0 normalization missed target: {relative_error}")

    schedule_class = str(case["schedule_class"])
    if schedule_class == "target100":
        queries, warmup = 300, 5
    elif schedule_class == "unscaled":
        queries, warmup = 100, 2
    elif schedule_class == "dense":
        queries, warmup = 20, 2
    else:
        raise AssertionError("unknown schedule class")
    rank_eligible = family == "dense-uniform-scope" and expected_j <= RANK_MAX_J
    scan_eligible = expected_j <= SCAN_MAX_J

    mode = str(case["name"])
    write_tsv(
        case_dir / "cases.tsv",
        [
            "case_index", "family", "mode", "requested_labels", "schedule_class",
            "weight_set", "join_size", "a", "mu0", "actual_mu", "sum_p2",
            "max_p", "chosen_root", "queries", "warmup", "rank_eligible",
            "scan_eligible", "weight_seed",
        ],
        [
            [
                case_index, family, mode, requested_labels, schedule_class,
                relative.as_posix(), expected_j, format(scale, ".17g"),
                format(mu0, ".17g"), format_ld(root0["mu"]),
                format_ld(root0["sum_p2"]), format_ld(root0["max_p"]),
                metrics["chosen_root"], queries, warmup, int(rank_eligible),
                int(scan_eligible), "" if seed is None else f"0x{seed:016x}",
            ]
        ],
    )
    write_tsv(
        case_dir / "root_scores.tsv",
        [
            "mode", "root", "lambda", "kappa", "mu", "join_count", "raw_min",
            "near_tie_eligible", "guarded_chosen",
        ],
        (
            (
                mode, item["root"], format_ld(item["lambda"]), format_ld(item["kappa"]),
                format_ld(item["mu"]), item["join_count"],
                int(item["root"] == metrics["raw_min_root"]),
                int(item["root"] in metrics["near_tie_eligible_roots"]),
                int(item["root"] == metrics["chosen_root"]),
            )
            for item in metrics["roots"]
        ),
    )

    weight_hashes = [
        hashed_file(case_dir / relative / f"r{relation}.tsv", case_dir)
        for relation in range(topology.m)
    ]
    result = {
        **dict(case),
        "query_directory": str(case_dir.resolve()),
        "weight_set": relative.as_posix(),
        "weight_seed": None if seed is None else f"0x{seed:016x}",
        "normalization_relation": 0,
        "normalization_a": format(scale, ".17g"),
        "mu0": format(mu0, ".17g"),
        "n": topology.n,
        "join_size": expected_j,
        "actual_mu": format_ld(root0["mu"]),
        "sum_p2": format_ld(root0["sum_p2"]),
        "max_p": format_ld(root0["max_p"]),
        "raw_min_root": metrics["raw_min_root"],
        "near_tie_eligible_roots": metrics["near_tie_eligible_roots"],
        "chosen_root": metrics["chosen_root"],
        "root_scores": [
            {
                "root": item["root"],
                "lambda": format_ld(item["lambda"]),
                "kappa": format_ld(item["kappa"]),
                "mu": format_ld(item["mu"]),
            }
            for item in metrics["roots"]
        ],
        "structural": structural,
        "serialized_weight_hashes": weight_hashes,
        "serialized_weights_reloaded_for_metrics": True,
        "exact_formula": exact_formula,
        "queries": queries,
        "warmup": warmup,
        "rank_eligible": rank_eligible,
        "scan_eligible": scan_eligible,
    }
    del topology, serialized, metrics
    gc.collect()
    return result

