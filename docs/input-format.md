# Input and output

An input describes a full equality join through a tree on `m` relation occurrences. Relation indices
are contiguous integers `0,...,m-1`. A self-join uses a separate relation index for each position in
the query. Every tree edge compares the separator key of its two endpoints.

For a general full acyclic query, start from a valid join tree. Encode the shared attributes on each
edge into one key, consistently on both endpoints. Composite separator values can be mapped to
signed 64-bit integer IDs. Retaining every shared attribute in its separator preserves the query's
equality constraints. Local filters are applied before preparing the files.

## Directory

```text
query/
  structure.tsv
  keys/r0.tsv
  keys/r1.tsv
  weights/r0.tsv
  weights/r1.tsv
```

Fields are tab-separated, and each file begins with the indicated header. The example directory
contains complete files that can be copied and edited.

`structure.tsv` begins with `stats-undirected-v1`, followed by these records:

| Record | Fields after its name |
| --- | --- |
| `query` | descriptive query name |
| `m` | number of relation occurrences |
| `verification_root` | valid relation index used in preparation |
| `relation` | relation index, alias, source relation name, row count, relative key-file path, metadata field |
| `edge` | edge ID, left relation index, right relation index, key cardinality, left separator name, right separator name |

Aliases, source names, separator names, and the last relation field describe the prepared data.
The sampler uses the declared row counts, file paths, relation indices, and edge endpoints.
It reads the root from `--root`. The example uses `unused` for the optional metadata field.
Each relation record has seven fields including `relation`, and each edge record has seven including
`edge`. A valid tree has `m-1` distinct edges and connects all relation occurrences.

For a relation incident to edges 0 and 2, the key file starts with:

```text
tuple_id	edge_0_key	edge_2_key
```

Write incident edges in increasing edge-ID order. Each following row contains a signed 32-bit tuple
ID and one signed 64-bit key per incident edge. IDs are unique within a relation occurrence.
Different relation occurrences may use the same ID. Two records with equal attribute values retain
different IDs when their occurrences should remain distinct.

A weight file has header `tuple_id<TAB>weight`. It contains exactly the same IDs in the same order as
the relation's key file. Weights are finite decimal values in `[0,1]`. Files for all relation indices
are named `r0.tsv`, `r1.tsv`, and so on within the relative directory passed as `--weights`.
An empty relation has header-only key and weight files and a declared row count of zero.

## Output and reuse

```json
{"call":0,"mu":1.575,"sample":[[101,201],[102,202]]}
```

Vector position `v` contains the ID selected from relation occurrence `v`, independently of the
chosen root. A sample contains each selected vector once. The CLI orders vectors lexicographically
to make output easier to inspect. Inclusion of a complete vector has target probability equal to
the product of its input weights. Sampling returns occurrence vectors even when projected attribute
values coincide.

`--calls` selects the number of sampling calls, each using fresh draws from the seeded generator.
The target distribution determines the number of vectors in each sample. The same executable and
standard-library implementation reproduce a run given the same files, root, and seed.
Construct a new index after changing the input or weights.

The loader checks file shape, IDs, numeric ranges, tree connectivity, and row alignment. Preparation
establishes that the encoded separators represent the intended query. The numerical operating
conditions of the binary64 prototype are described in [implementation.md](implementation.md).
