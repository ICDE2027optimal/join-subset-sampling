#!/usr/bin/env python3
"""Blackbox checks for the compiled product_sample executable using Python's stdlib."""

import argparse
import collections
import itertools
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest


BINARY = None
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "tiny"


def write_query(directory, relations, edges):
    """Write prepared rows as (tuple_id, {edge_id: key}, weight)."""
    (directory / "keys").mkdir(parents=True)
    (directory / "weights").mkdir()
    structure = ["stats-undirected-v1", "query\ttest", f"m\t{len(relations)}",
                 "verification_root\t0"]
    for relation, rows in enumerate(relations):
        structure.append(
            f"relation\t{relation}\tr{relation}\tR{relation}\t{len(rows)}\t"
            f"keys/r{relation}.tsv\tunused")
        incident = sorted(edge for edge, left, right in edges
                          if relation in (left, right))
        keys = ["\t".join(["tuple_id"] + [f"edge_{edge}_key" for edge in incident])]
        weights = ["tuple_id\tweight"]
        for tuple_id, key_map, weight in rows:
            keys.append("\t".join([str(tuple_id)] + [str(key_map[edge]) for edge in incident]))
            weights.append(f"{tuple_id}\t{weight}")
        (directory / "keys" / f"r{relation}.tsv").write_text("\n".join(keys) + "\n", encoding="utf-8")
        (directory / "weights" / f"r{relation}.tsv").write_text("\n".join(weights) + "\n", encoding="utf-8")
    for edge, left, right in edges:
        structure.append(f"edge\t{edge}\t{left}\t{right}\t1\tkey\tkey")
    (directory / "structure.tsv").write_text("\n".join(structure) + "\n", encoding="utf-8")


class SamplingCLI(unittest.TestCase):
    def invoke(self, query=EXAMPLE, root=0, seed=246813579, calls=1, extra=()):
        command = [str(BINARY), "--query-dir", str(query), "--weights", "weights",
                   "--root", str(root), "--seed", str(seed), "--calls", str(calls)]
        return subprocess.run(command + list(extra), capture_output=True, text=True, timeout=60)

    def parse(self, process, count, relation_count, mu, allowed):
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        lines = process.stdout.splitlines()
        self.assertEqual(len(lines), count)
        subsets = []
        for call, line in enumerate(lines):
            record = json.loads(line)
            self.assertEqual(set(record), {"call", "mu", "sample"})
            self.assertEqual(record["call"], call)
            self.assertTrue(math.isfinite(record["mu"]))
            self.assertAlmostEqual(record["mu"], mu, places=12)
            vectors = record["sample"]
            self.assertIsInstance(vectors, list)
            self.assertEqual(vectors, sorted(vectors))
            for ids in vectors:
                self.assertIsInstance(ids, list)
                self.assertEqual(len(ids), relation_count)
                self.assertTrue(all(type(value) is int for value in ids))
            subset = frozenset(tuple(ids) for ids in vectors)
            self.assertEqual(len(vectors), len(subset))
            self.assertTrue(subset.issubset(allowed), subset)
            subsets.append(subset)
        return subsets

    def assert_rejected(self, process):
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(process.stdout, "")
        self.assertIn("product_sample:", process.stderr)

    def test_help_and_repeatability(self):
        help_result = subprocess.run([str(BINARY), "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--query-dir", help_result.stdout)
        first = self.invoke(calls=20)
        second = self.invoke(calls=20)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)

    def test_example_product_law_and_shared_record_independence(self):
        probabilities = {(101, 201): 0.8, (101, 202): 0.4,
                         (102, 201): 0.25, (102, 202): 0.125}
        allowed = frozenset(probabilities)
        count = 8000
        for root in (0, 1):
            with self.subTest(root=root):
                samples = self.parse(self.invoke(root=root, calls=count), count, 2, 63 / 40, allowed)
                observations = collections.Counter(samples)
                self.assertGreater(len({len(sample) for sample in samples}), 2)
                for output, probability in probabilities.items():
                    observed = sum(output in sample for sample in samples) / count
                    tolerance = 6 * math.sqrt(probability * (1 - probability) / count) + 0.002
                    self.assertLess(abs(observed - probability), tolerance)
                for left, right in (((101, 201), (101, 202)), ((101, 201), (102, 201))):
                    expected = probabilities[left] * probabilities[right]
                    observed = sum(left in sample and right in sample for sample in samples) / count
                    tolerance = 6 * math.sqrt(expected * (1 - expected) / count) + 0.002
                    self.assertLess(abs(observed - expected), tolerance)
                # All sixteen subsets exercise the joint product law.
                chi_squared = 0.0
                for flags in itertools.product((False, True), repeat=len(probabilities)):
                    subset = frozenset(output for output, included in zip(probabilities, flags) if included)
                    probability = math.prod(p if included else 1 - p
                                            for p, included in zip(probabilities.values(), flags))
                    expected = count * probability
                    chi_squared += (observations[subset] - expected) ** 2 / expected
                self.assertLess(chi_squared, 65.0)

    def test_deterministic_endpoints_and_occurrence_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            query = Path(temporary)
            write_query(query, [
                [(-7, {0: 1}, 1), (2147483647, {0: 1}, 1), (0, {0: 2}, 0)],
                [(-7, {0: 1}, 1), (-2147483648, {0: 1}, 1), (9, {0: 2}, 1)],
            ], [(0, 0, 1)])
            expected = frozenset(itertools.product((-7, 2147483647), (-7, -2147483648)))
            samples = self.parse(self.invoke(query, root=1, calls=12), 12, 2, 4, expected)
            self.assertEqual(samples, [expected] * 12)

    def test_single_relation_half_boundary_and_maximum_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            query = Path(temporary)
            write_query(query, [[(-1, {}, 0), (0, {}, 1), (1, {}, 0.5)]], [])
            count = 4000
            samples = self.parse(self.invoke(query, seed=18446744073709551615, calls=count),
                                 count, 1, 1.5, frozenset({(0,), (1,)}))
            self.assertTrue(all((0,) in sample for sample in samples))
            observed = sum((1,) in sample for sample in samples) / count
            self.assertLess(abs(observed - 0.5), 0.05)

    def test_empty_and_zero_mass_joins(self):
        cases = [
            ([[(1, {0: 10}, 1)], [(2, {0: 20}, 1)]], [(0, 0, 1)]),
            ([[(1, {0: 10}, 0)], [(2, {0: 10}, 1)]], [(0, 0, 1)]),
            ([[], [(2, {0: 10}, 1)]], [(0, 0, 1)]),
            ([[]], []),
        ]
        for relations, edges in cases:
            with self.subTest(relations=relations), tempfile.TemporaryDirectory() as temporary:
                query = Path(temporary)
                write_query(query, relations, edges)
                samples = self.parse(self.invoke(query, calls=3), 3, len(relations), 0, frozenset())
                self.assertEqual(samples, [frozenset()] * 3)

    def test_option_validation(self):
        base = [str(BINARY), "--query-dir", str(EXAMPLE), "--weights", "weights",
                "--root", "0", "--seed", "1", "--calls", "1"]
        standalone = [[], ["--query-dir"], ["--query-dir", str(EXAMPLE)],
                      ["--query-dir", "--weights"]]
        for arguments in standalone:
            with self.subTest(arguments=arguments):
                self.assert_rejected(subprocess.run([str(BINARY)] + arguments,
                                                    capture_output=True, text=True, timeout=10))
        for arguments in (["--unknown", "1"], ["--calls", "1"], ["--seed", "1"], ["--root", "0"]):
            with self.subTest(arguments=arguments):
                self.assert_rejected(subprocess.run(base + arguments, capture_output=True, text=True, timeout=10))
        for name, invalid in (
            ("--root", "-1"), ("--root", "2"), ("--root", "2147483648"),
            ("--seed", "-1"), ("--seed", "+1"), ("--seed", "18446744073709551616"),
            ("--calls", "0"), ("--calls", "-1"), ("--calls", "1x"),
            ("--calls", "18446744073709551616"), ("--root", " 0"),
            ("--weights", "../weights"), ("--weights", str(EXAMPLE.resolve())),
            ("--query-dir", ""),
        ):
            with self.subTest(name=name, value=invalid):
                command = base.copy()
                command[command.index(name) + 1] = invalid
                self.assert_rejected(subprocess.run(command, capture_output=True, text=True, timeout=10))

    def test_malformed_prepared_input(self):
        cases = [
            ("weights/r0.tsv", "tuple_id\tweight\n2\t1\n"),
            ("weights/r0.tsv", "tuple_id\tweight\n1\tnan\n"),
            ("weights/r0.tsv", "tuple_id\tweight\n1\tinf\n"),
            ("weights/r0.tsv", "tuple_id\tweight\n1\t-0.1\n"),
            ("weights/r0.tsv", "tuple_id\tweight\n1\t1.1\n"),
            ("weights/r0.tsv", "tuple_id\tweight\n1\t0.5x\n"),
            ("weights/r0.tsv", "tuple_id\tweight\n1\t\n"),
            ("weights/r0.tsv", "tuple_id\tweight\n"),
            ("keys/r0.tsv", "tuple_id\tedge_0_key\n2147483648\t7\n"),
            ("keys/r0.tsv", "tuple_id\tedge_0_key\n1\t9223372036854775808\n"),
            ("keys/r0.tsv", "tuple_id\tedge_0_key\n1\t7\n2\t7\n"),
            ("keys/r0.tsv", "tuple_id\twrong_header\n1\t7\n"),
        ]
        for filename, contents in cases:
            with self.subTest(file=filename, contents=contents), tempfile.TemporaryDirectory() as temporary:
                query = Path(temporary)
                write_query(query, [[(1, {0: 7}, 1)], [(2, {0: 7}, 1)]], [(0, 0, 1)])
                (query / filename).write_text(contents, encoding="utf-8")
                self.assert_rejected(self.invoke(query))
        for old, replacement in (
            ("stats-undirected-v1", "unknown-format"),
            ("verification_root\t0", "verification_root\t2"),
            ("keys/r0.tsv", "../r0.tsv"),
            ("R0\t1\t", "R0\t-1\t"),
            ("relation\t1\t", "relation\t0\t"),
        ):
            with self.subTest(old=old, replacement=replacement), tempfile.TemporaryDirectory() as temporary:
                query = Path(temporary)
                write_query(query, [[(1, {0: 7}, 1)], [(2, {0: 7}, 1)]], [(0, 0, 1)])
                structure = query / "structure.tsv"
                structure.write_text(structure.read_text(encoding="utf-8").replace(old, replacement), encoding="utf-8")
                self.assert_rejected(self.invoke(query))

    def test_duplicate_ids_and_cyclic_tree(self):
        cases = [
            ([[(1, {0: 7}, 1), (1, {0: 7}, 1)], [(2, {0: 7}, 1)]], [(0, 0, 1)]),
            ([[(1, {0: 7, 2: 7}, 1)], [(2, {0: 7, 1: 7}, 1)],
              [(3, {1: 7, 2: 7}, 1)], [(4, {}, 1)]], [(0, 0, 1), (1, 1, 2), (2, 2, 0)]),
        ]
        for relations, edges in cases:
            with self.subTest(edges=edges), tempfile.TemporaryDirectory() as temporary:
                query = Path(temporary)
                write_query(query, relations, edges)
                self.assert_rejected(self.invoke(query))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="compiled product_sample executable")
    arguments, remaining = parser.parse_known_args()
    BINARY = arguments.binary.resolve()
    if not BINARY.is_file():
        parser.error(f"binary does not exist: {BINARY}")
    unittest.main(argv=[__file__] + remaining)
