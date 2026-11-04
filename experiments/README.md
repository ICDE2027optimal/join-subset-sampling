# Paper experiments

The two entry points are inspection of the supplied author measurements and execution of new
measurements. Run the commands below from the repository root. New inputs and outputs use `data/`
and `runs/`, which are excluded from Git.

## Inspect the recorded results

```sh
python3 experiments/inspect_results.py
python3 experiments/inspect_results.py --output runs/author-summary.json
```

The script verifies file digests, reconstructs method medians and observed ranges, and recomputes
the original baseline comparisons and paired materialization ratios. It reports the 17/18 PRODUCT
sampling wins, 9/18 total-cost wins, and 52/67 paired total-cost wins. Its output file also contains
per-case comparator identities and estimated crossover counts. The extension comparison covers all
ten NEWbranch5 and JOB17a cases, with comparator identities selected from complete measurements.

| Batch ID | Coverage | Recorded outcomes | Paper location |
| --- | --- | ---: | --- |
| `stats_original` | 36 STATS configurations | 708 completed | VI-B, Fig. 2, Table III |
| `controlled` | 31 size, degree, triangular, and density configurations | 544 completed, 16 timeouts | VI-B, Fig. 3, Table III |
| `job1a` | 10 JOB1a configurations | 212 completed | VI-B, Table III |
| `materialized_paired` | 67 configurations measured in a separate paired batch | 536 completed | VI-C, Fig. 4, Table III |
| `extension` | Five configurations each for NEWbranch5 and JOB17a | 196 completed, 12 memory failures | VI-B, Table IV |

The five batches remain separate. Each case-method group has four terminal outcomes. For the
controlled timeouts, Global exceeded the limit in cases 21, 23, 24, and 25. In the extension, Score
encountered allocation failures in JOB17a's three PRODUCT configurations.

`results/measurements.jsonl` contains one normalized row per process, with exact seed, method,
repeat, query count, timing, output count, memory, and status. `records.jsonl` contains the original
driver payload and stdout plus the process wrapper with operational locations removed.
`aggregates.json` provides the per-method medians and endpoints. `batch_metadata.json` gives case
parameters and recorded source/binary hashes. `manifest.json` describes the fields, sanitization,
missing values, and SHA256 digests. Original source-record hashes cover the original bytes.
Distributed-file hashes cover the supplied files after location removal.

For an allocation failure, `raw_status` preserves the original process outcome and `status` gives
its evidence-based resource classification. Missing component measurements are `null`. Peak RSS
for the 16 timeouts is also unavailable. Historical executable identities are recorded as hashes.
The build commands regenerate executables using the current environment.

## Run a controlled experiment

Use Python 3.10–3.12 for experiment preparation. The tested version is Python 3.10.12.
Install the preparation dependencies in that environment:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r experiments/configs/requirements.txt
```

The requirements select NumPy 1.26.1 and DuckDB 1.5.6, matching the preparation environment.
The [NumPy release notes](https://numpy.org/doc/stable/release/1.26.1-notes.html)
specify its supported Python versions.
Execution uses Linux CPU affinity, address-space limits, and GNU `time`.

Start with a short end-to-end check:

```sh
python3 experiments/runners/run_controlled.py --group smoke \
  --batch all --output runs/smoke --execute
```

For one original degree case, use:

```sh
python3 experiments/runners/run_controlled.py --group paper-degree4 \
  --batch all --output runs/degree4 --execute
```

This case has four relations, `N = 65,536`, degree `4`, `J = 1,048,576`, and expected sample size
`100`. Each process performs five warmup calls and 300 measured calls. The command runs 16 primary
processes and eight paired processes with their original seeds and method order. The paired batch
rebuilds and remeasures Ours independently alongside Materialized.

Other groups are `paper-size`, `paper-degree`, `paper-triangular`, `paper-dense`, and `paper-all`.
Choose `--batch primary` or `--batch paired` to run either comparison alone. These larger groups use
the full paper schedules in `configs/controlled_groups.json`. The triangular family uses ordered
keys on an acyclic path. Its joins remain acyclic.

Omitting `--execute` prepares input and writes a measurement plan. Each execution requires a new
output directory and records its executable and prepared-input hashes. `--cpu` chooses an allowed
logical core. The default selects the first allowed core. Each process has an 8 GiB address-space
limit and a 120-second timeout. Timeouts are recorded and execution continues. Other unexpected
exits stop the controlled runner for inspection.

The output contains `preparation/`, `manifest.json`, and `run/`. Each process under `run/processes/`
has `record.json`, `stdout.jsonl`, `stderr.txt`, and `gnu_time.txt`. `run/summary.json` reports the
terminal status counts. Construction and call costs are `driver_record.method_build_ms` and
`driver_record.query_ms_mean` in each completed process record.

## Prepare real queries

Source manifests pin the STATS snapshot, JOB SQL, IMDb archive, and accepted source-file hashes.
The acquisition script verifies hashes before preparation. Reuse an existing IMDb archive with
`--archive PATH` to avoid downloading it again. Source files and prepared data can require substantial
disk space, particularly for JOB17a.
After an interrupted download or extraction, repeat the same command. Completed files are verified
and reused. Unfinished files are processed again using separate temporary files, with completed
downloads and extracted tables checked before replacing their destinations.

```sh
python3 experiments/preparation/acquire_data.py stats --output data/stats
python3 experiments/preparation/prepare_dataset.py stats \
  --raw-dir data/stats --output runs/stats-preparation

python3 experiments/preparation/prepare_dataset.py newbranch5 \
  --raw-dir data/stats --output runs/newbranch5-preparation

python3 experiments/preparation/acquire_data.py job --output data/job
python3 experiments/preparation/prepare_dataset.py job \
  --raw-dir data/job --output runs/job1a-preparation

python3 experiments/preparation/acquire_data.py job17a --output data/job17a
python3 experiments/preparation/prepare_dataset.py job17a \
  --raw-dir data/job17a --output runs/job17a-preparation
```

Use `acquire_data.py DATASET --output PATH --check-only` to verify existing source files.
Preparation applies the published filters, encodes separator keys, generates weights and target
scales, chooses a common root, and checks join counts and moments. Each resulting query directory
contains `structure.tsv`, key files, and case-specific relative weight directories.

The four original modes are uniform, one-relation, real PRODUCT, and shuffled PRODUCT. Shuffling
preserves the marginal weights in each relation while changing their association with join keys.
The configs retain the target schedules, normalization, features, and shuffle seeds. The extension
uses three real PRODUCT targets and uniform and one-relation target-100 configurations.
Feature mappings and their boundary conventions are listed in [preparation-details.md](preparation-details.md).

## Rerun a recorded real-data case

Supply the prepared query directory, batch ID, and case index. For example, original STATS case 0
uses Q010:

```sh
python3 experiments/runners/run_recorded_case.py --batch stats_original --case 0 \
  --query-dir runs/stats-preparation/prepared/Q010 \
  --output runs/stats-case0 --execute
```

This entry restores the case's original methods, seeds, root, warmups, measured-call counts, and
resource limits from the supplied records. `--methods candidate root` selects specific methods.
The same entry accepts `job1a`, `extension`, and `materialized_paired`. The paired case indices and
their original parent case indices appear in `batch_metadata.json`. JOB17a uses 32 GiB and 600-second
limits. Allocation failures identified by `std::bad_alloc` and timeouts are recorded as terminal
resource outcomes. Other unexpected exits stop execution for inspection.

## Numerical validation

```sh
./build/core_correctness
ctest --test-dir build -R sampling_cli --output-on-failure
```

The core suite uses 300,000 proportional draws and 250,000 subset calls on a small enumerable join.
It checks all subset probabilities and shared-record independence. The boundary suite includes
14,502 deterministic cases, a 100,000-call threshold example, and a 150,000-call complete-law check
on nine witnesses. Total variation is half the sum of absolute differences over all subsets.
Adjacent-call products use the covariance correction described in `docs/implementation.md`.
These diagnostics assess the floating-point implementation on the specified fixtures.
The author output is retained in `results/numerical_checks.txt`, with its source hashes in
`results/numerical_checks.json`.

## Statistics and timing scope

- `B` is method construction time after common preparation, loading, and tree orientation.
- `q` is the mean measured-call time within one process. Call timing includes constructing,
  populating, and destroying the returned subset.
- `k` is the scheduled number of measured calls. `B+kq` measures construction plus those calls.
  Warmup calls and common preparation/loading are separate recorded quantities.
- Tables report medians across four processes and the complete observed range `max-min` in
  parentheses. Ranges use unrounded endpoints. Sampling entries in Tables III–IV convert ms to μs.
  Bold entries identify the lowest median within the same batch and metric.
- For the original STATS, controlled, and JOB1a comparisons, sampling ratios divide method medians.
  Total-cost ratios use `median(B)+k median(q)` for each method. The denominator is the fastest
  applicable baseline with all four measurements, chosen separately for each case and metric.
- Extension comparisons use medians of the four per-process `B+kq` values. Sampling ratios divide
  method medians. Both comparisons select the fastest applicable baseline with four completed
  measurements. Ratios in the inspection output are Ours divided by that baseline.
- The paired study divides corresponding repeats first and then reports the median of the four
  ratios. For total cost, repeat `r` contributes `(B_ours,r+k q_ours,r)/(B_mat,r+k q_mat,r)`.
  Vertical segments in Fig. 4 show the minimum and maximum of these four observed ratios.
- Fig. 4 cost curves evaluate measured component medians as `median(B)+k median(q)`. Their estimated
  crossing is `k*=(B_mat-B_ours)/(q_ours-q_mat)` when the denominator is nonzero. These curves reuse
  the measured components across call counts. Observed ranges describe repeat variability.
- Resource failures leave component costs unavailable. A timeout duration supplies no estimate
  for `B` or `q`. NA denotes an inapplicable method, TO a timeout, and MEM memory exhaustion.
  A dash in Table III denotes a method outside that batch.

The author measurements used GCC 11.4 with `-O3 -DNDEBUG`, Ubuntu 22.04, and one pinned core of an
Intel Xeon Gold 6342. Other tasks could run on the host. Each configuration uses four process
repeats with distinct seeds and the recorded method order. The source record includes CPU 0.
The portable runners select a core permitted by the current system.
