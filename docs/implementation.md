# Implementation and paper correspondence

`WeightedJoinIndex` keeps the input instance by reference. Keep that instance alive and unchanged
while using the index. Construct `std::mt19937_64` once and pass it to successive calls of
`sample_subset(index, rng, stats)`. Each returned vector owns its occurrence IDs.

## Shared index

`WeightedJoinIndex::build` groups tuples by their parent separator and computes subtree total and
maximum completion weights in reverse preorder. Total weights build alias tables for proportional
draws. Maximum weights order the group heaps used by heavy enumeration.

`draw_into` visits the join tree in preorder. Each selected parent tuple identifies the compatible
child group. The operation returns a complete ID vector and its product weight. Its proposal
distribution covers the whole join, including heavy witnesses.

`enumerate_heavy` and `heavy_dfs` combine tuple entries into complete witnesses. Heap thresholds use
maximum completion summaries to prune prefixes. Complete products above `1/2` receive direct
Bernoulli trials in `sample_subset`.

## Light path

`sample_subset` simulates exponential interarrival times and makes proportional draws up to the
Poisson horizon. It discards heavy proposals and independently accepts each light proposal with
the thinning probability. Accepted vectors enter the output hash set, which retains one copy of
each selected vector. This order implements acceptance before deduplication. Hash equality compares
the entire occurrence vector.

## Floating-point computation

The prototype uses binary64 weights, summaries, arithmetic, and a pseudorandom generator.
`log_survival_ratio` uses `log1p` for stable evaluation near zero. A small multiplier inflation
supports acceptance probabilities near their upper bound. Heavy search uses a slightly reduced
cutoff, then classifies each complete witness using the same preorder product used by proportional
draws. These measures support consistent threshold handling in the tested instances.

Finite mass and rate checks raise errors for unsupported numeric values. Underflow of extremely
small products and rounding of cumulative masses remain properties of the binary64 representation.
The paper's exact real-RAM proof specifies the mathematical target. `core_correctness` checks the
empirical complete-subset law, shared-tuple independence, and near-threshold behavior of the
implementation.

The prototype eagerly allocates `m`-entry scratch vectors during a positive-mass call. The proof
uses allocation on demand when making the dependence on `m` explicit. Both have the stated
fixed-query asymptotic cost.

## Source identity

The packaged algorithm and measurement translation units retain the executable tokens of the
paper's sources. Comments and whitespace have been edited for this standalone artifact.
`source-provenance.json` records both byte hashes and the token comparison. `src/sample.cpp`, the
CMake build, and portable experiment entry points provide the standalone interfaces.
The numerical suite includes the corrected adjacent-pair variance:

```text
n = T - 1
Var(sum of adjacent products) = n p² (1-p²) + 2(n-1) p³ (1-p)
```

The second term accounts for consecutive pair products sharing one Bernoulli variable.
Source hashes, binary hashes, seeds, and the author measurement fields are documented in the
[experiment guide](../experiments/README.md).
