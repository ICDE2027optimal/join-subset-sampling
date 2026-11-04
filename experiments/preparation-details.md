# Preparation conventions

All methods in a comparison receive the same filtered rows, encoded separators, weights, and
rooted join tree. Preparation computes the unscaled expectation, chooses normalization, and
evaluates root envelope masses. The common root is the first relation index whose computed mass
is within relative `1e-12` of the minimum.

Feature mappings use `ε = 1/64`. Missing features receive `ε`. Valid constant date domains receive
weight one. Otherwise the original table's minimum and maximum dates map linearly to `[ε,1]`.
For nonnegative importance `x`, the weight is

```text
ε + (1-ε) min(1, log(1+x)/log(1+q99))
```

Here `q99` is the nearest-rank 99th percentile in the original table. If `q99=0`, positive values
receive weight one and zero receives `ε`. JOB1a combines title-year recency over 1880–2019 with
`ε+(1-ε)/r` for valid top-250 ranks. Other rank subtypes receive `ε`. JOB17a uses the same reciprocal
form for positive cast billing order, assigning `ε` otherwise.

Requested expected sample sizes are reached by scaling the designated relation's weights downward.
The comparison uses the actual computed expectation when a target exceeds the unscaled mass.
Identical unscaled cases are included once. The JSON query configs identify each feature, anchor,
target, and seed.

Size and degree workloads use tuple weights `2^(-u)` with `u` uniform on `(0,3)`, scaled to expectation
100. Triangular workloads use ordered-key constraints on a four-relation path, with selected endpoint
weights one and the remaining endpoint weights `K^-2`. Their unscaled and distinct target-100
instances retain the original case indices. Dense cases have constant witness probabilities
`1/1024`, `1/16`, `1/2`, `3/4`, and `1`.

Preparation and loading are outside `B+kq`. The recorded STATS preparation took approximately
88 seconds for the complete workload, excluding download. Its scope is the entire preparation,
whereas each reported method cost belongs to one configuration. Changed data or weights can require
repeating preparation. The individual preparation manifests record the new environment and elapsed
time when these steps are rerun.
