# Baselines and dependencies

The methods below share input loading, query orientation, tuple IDs, compiler flags, and output
semantics. Their implementations were written for this study from the cited algorithmic descriptions.

| Paper name | Driver ID | Implementation and role |
| --- | --- | --- |
| Ours | `candidate` | `core.hpp`, the paper's shared-index heavy/light sampler |
| Score | `score` | `score_baseline_hybrid.hpp`, PRODUCT score buckets following [Esmailpour et al., Subset Sampling over Joins](https://doi.org/10.1145/3801913) |
| Global | `global` | `uniform_envelope.hpp`, one probability upper bound over all witnesses |
| Root | `root` | `root_envelope.hpp`, a separate upper bound for each root block, related to the conditioning mechanism in [Bekkers et al., Poisson Sampling over Acyclic Joins](https://doi.org/10.1145/3802101) |
| Rank | `rank` | `rank_scan_controls.hpp`, geometric skipping for uniform inclusion probabilities |
| Scan | `scan` | `rank_scan_controls.hpp`, stored witness vectors with independent Bernoulli trials |
| Materialized | `materialized-explicit` | `materialized_explicit.hpp` and `explicit_subset.hpp`, full enumeration followed by an explicit-subset index based on [Bringmann and Panagiotou](https://doi.org/10.1007/s00453-016-0205-0) |

`formal_driver` runs the first six methods. `materialized_explicit_driver` runs Materialized.
Both report construction cost, sampling cost including returned-vector lifetime, and storage
diagnostics. The paper applies Rank to uniform configurations, Scan up to one million witnesses,
and Materialized up to 25 million witnesses.

## Score details

The score step is `δ = 1/m`, with cutoff `L = ceil(2m log2(max(N,2))/δ)`.
A positive tuple weight contributes `floor(-log2(w)/δ)`. Below `L`, alias tables select the first
contributing bucket. A conditioned geometric gap selects its first candidate, and ordinary geometric
gaps select later candidates. Thinning converts the bucket bound `2^(-δs)` to each witness probability.
Scores at least `L` use geometric proposals over the full join under bound `2^(-δL)`, followed by
score and probability checks.

`score_convolution.hpp` chooses sparse, direct, or number-theoretic-transform convolution during
construction. Modular convolutions use the bundled AtCoder Library. Checked unsigned Chinese
remainder reconstruction recovers the integer counts. The AtCoder headers are the sole vendored
algorithm library, with release, commit, and license listed in `third_party/ac-library/PROVENANCE.md`.

## Materialized details

Construction computes completion counts, enumerates witnesses, stores their `mJ` IDs in a flat
array, and builds an independent-subset index over their probabilities. Counts use overflow checks.
The temporary counts and probability vector are released during construction.

Probability-one witnesses are stored separately. Other positive probabilities use dyadic upper
bounds. Alias tables select contributing buckets, and geometric skipping generates candidates within
them. Thinning uses the ratio of the witness probability to its bucket bound. Probabilities below
`(J+1)^(-2)` share a tail bucket with that bound. Zero-probability witnesses contribute no candidates.
Each sampling call copies the full selected occurrence vectors into its returned result.

## Attribution and source tracking

The cited papers define the algorithmic sources of these comparisons. This repository's C++
implementations and integration code are distributed under the root MIT license. AtCoder Library
retains CC0. Dataset preparation uses public STATS and JOB inputs identified by version and hashes
in `experiments/configs`. The public input repositories retain their own terms.

The measurement drivers and headers have source hashes in `source-provenance.json`.
The per-batch measurement manifests retain the original executable and source hashes in
`experiments/results/batch_metadata.json`. The portable CMake build regenerates executables for the
current compiler and environment.
