# Phase 4 prior-comparison pilot results

**Status: complete — 15 of 15 runs.** The supervisor-approved design compares
B2 (uniform replay), B5 (embedding-similarity replay), and M3
(published-interference replay) on five frozen ten-language orders, all with
training seed 42.

All five submitted Kaggle output archives match the frozen protocol and share
the same protocol hash
(`8410f165679468b770528eb8083160570c374921d0f1a0fdb2935b87ba6b008b`), data
manifest hash
(`373852a7f1e09ab99da8a25f21023fedc1977ab1969e09d938874f7c645a721a`), and
source-bundle fingerprint
(`sha256:642db1b58cf5105a3801fe9ffa70431b845c3dad2f4a08873a40c7d5732d424d`).
Each archive contains three completed records for its corresponding order.
The unified aggregate validates all 15 expected run IDs and has no missing
runs. The five executed notebooks were configured for their matching orders
and report completed cells with no execution errors.

## Descriptive results

Metrics are means across the five order-level results at the **same training
seed**, not estimates across independent training seeds. ACC standard
deviation is across orders.

| Method | ACC mean ± SD | Mean BWT | Mean FWT | Mean F (forgetting) |
|---|---:|---:|---:|---:|
| B2 — Uniform replay | 0.6807 ± 0.0086 | 0.0674 | 0.2854 | 0.0056 |
| B5 — Embedding-similarity replay | 0.6632 ± 0.0137 | 0.0714 | 0.2486 | 0.0168 |
| M3 — Published-interference replay | 0.6858 ± 0.0104 | 0.0701 | 0.2823 | 0.0062 |

M3 has the highest mean ACC in this pilot, but the differences are descriptive
only. This single-seed, five-order comparison does not establish seed
robustness or statistical significance and should not be presented as a
confirmatory winner selection.

M3 logs confirm 500 current examples plus 500 replay examples at each later
language stage, a 250-example reservoir, and no replay at the first stage.
Arabic uses the frozen URIEL+ syntactic-distance `k=5` imputation.

## Runtime and files

The notebook run cells took 8.90–10.34 hours per order (mean 9.49 hours) for
the three methods, based on the submitted Papermill execution metadata.
Runtime depends on Kaggle hardware and scheduling.

- [`aggregate.json`](./aggregate.json): complete unified 15-run summary.
- [`results/`](./results/): one validated JSON record per run.
- [`protocol_manifest.json`](./protocol_manifest.json): frozen protocol and
  prepared-data checksums.
- [`priors/`](./priors/): the shared prepared interference matrix and metadata.
- [`input_archives/`](./input_archives/): original submitted order output ZIPs.
- [`provenance/execution_validation.json`](./provenance/execution_validation.json):
  notebook completion checks, execution durations, and archive hashes.

For any additional-seed study, create a new protocol ID and obtain approval
before training; do not combine future runs with this single-seed pilot.
