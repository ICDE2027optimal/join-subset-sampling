# Independent subset sampling over acyclic joins

Code and measurement records for the paper

>>**Optimal-Time Independent Subset Sampling over Acyclic Joins**.

The sampler builds an index over a static, full acyclic join. Each input tuple has a weight in
`[0,1]`. A complete join result is included with probability equal to the product of its tuple
weights, with independent inclusion decisions across results. Each returned result contains one
tuple-occurrence ID per relation occurrence. **The subset size is random. Its expected size is μ.**

The C++ implementation uses binary64 arithmetic. The paper establishes the distribution and
complexity guarantees in its exact real-RAM model. [Implementation notes](docs/implementation.md)
describe the numerical measures used by the prototype.

## Build

Use Linux, GCC 11 or newer, CMake 3.16 or newer, and Python 3.10 or newer.
The core and measurement drivers use the C++ standard library and the bundled AtCoder headers.
Python's standard library suffices for the CLI tests and inspection of supplied measurements.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

For a smaller build containing just the sampler, add
`-DBUILD_EXPERIMENTS=OFF -DBUILD_TESTING=OFF` to the configuration command.

## Sample a small join

The example joins `R0.key = R1.key`.

| R0 occurrence ID | key | weight | R1 occurrence ID | key | weight |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 101 | 7 | 0.8 | 201 | 7 | 1 |
| 102 | 7 | 0.25 | 202 | 7 | 0.5 |
| 103 | 8 | 0 | 203 | 8 | 1 |

The five complete results have inclusion probabilities `0.8`, `0.4`, `0.25`, `0.125`, and `0`.
Their sum is `μ = 1.575`. Results that share an input record still receive independent inclusion
decisions.

```sh
./build/product_sample --query-dir examples/tiny --weights weights \
  --root 0 --seed 2027 --calls 4
```

On the tested GCC 11.4 build, the first two lines are:

```json
{"call":0,"mu":1.5750000000000002,"sample":[[101,201]]}
{"call":1,"mu":1.5750000000000002,"sample":[[101,201],[101,202],[102,201]]}
```

Each output line reports one sample. For example, a line with
`"sample":[[101,201],[102,202]]` contains two complete results, each represented in relation order.
An empty subset appears as `"sample":[]`. The `call` field starts at zero, and `mu` reports the
index's computed expected sample size. The CLI sorts vectors for readable output.
The experiment drivers measure the sampler's output construction and destruction directly.

## Use your own input

Provide a join tree, a key file for each relation occurrence, and an aligned file of tuple weights.
The [input-format guide](docs/input-format.md) specifies every field and explains composite keys,
duplicate records, and output IDs. Replace `examples/tiny` and `weights` in the command above with
your prepared directory and its relative weight directory. Choose a relation index for `--root`.
The same index is reused across the requested calls.

## Locate the algorithm

| Paper component | Implementation |
| --- | --- |
| Join-tree index and total/max completion summaries | `include/core.hpp`, `WeightedJoinIndex::build` |
| `WeightedDraw` | `WeightedJoinIndex::draw_into` |
| Heavy witness enumeration | `enumerate_heavy` and `heavy_dfs` |
| Poisson proposals, independent acceptance, and deduplication | `sample_subset` |
| Input loading and tree orientation | `prepared_loader.hpp` and `orient.hpp` |
| Command-line use | `src/sample.cpp` |

## Test

```sh
ctest --test-dir build --output-on-failure
```

The tests cover input validation, tuple identities, tree orientation, probability endpoints,
shared-record independence, complete subset distributions, and the heavy/light boundary.
The adjacent-call diagnostic accounts for the covariance between overlapping pairs.

## Paper experiments

The [experiment guide](experiments/README.md) provides two entry points:

- Inspect the supplied per-process author measurements and recompute their statistics.
- Prepare workloads and run new measurements under `runs/`.

The guide covers real queries, controlled workloads, repeated sampling with materialization, and
numerical validation. [Baseline notes](docs/baselines.md) identify each implementation and its source.
[Source provenance](docs/source-provenance.json) records the correspondence with the measured code.

## License

The artifact code is available under the [MIT license](LICENSE). The vendored AtCoder Library retains
its [CC0 license](third_party/ac-library/LICENSE). External datasets retain their upstream terms.
